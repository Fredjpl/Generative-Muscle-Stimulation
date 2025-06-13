import re
import pandas as pd
from speech_engine import SpeechEngine
import sys
import time
import pandas as pd
from mocap_oscserver import MocapOSCServer
from oscservers import OSCServers
import math

output_gesture_response = '''
  left hand:
1. Maintain a steady grip on the upper section of the tripod’s legs, close to the central pivot  
<left>[grasp]
<right>[wrist flexion]

'''

test_skip = '''
  gestures: 
Left hand:
1. From the fully downward (locked) position, rotate your wrist so that the handle moves to the angle you want  
<left>[skip]

'''

single_gestures_response = '''
<left>[wrist][rotation][counterclockwise][90]
<left>[shoulder][rotation][counterclockwise][10]
<left>[grip]
'''

speech_engine = SpeechEngine()

sys.path.append('/Users/yunho/Documents/code-local-copy/ems-auto_detect_feature/ems')
sys.path.append('/Users/romainnith/Projects_local/Playground/python/libs/ems-libs/ems')
from core import EMS


class GestureProcessor:
    def __init__(self, skeleton=None, instruction_pause=False, debug=False):
        self.llm_gestures_csv = pd.read_csv("gesture_lists/llm-gesture-list.csv")
        self.ems_gestures_csv = pd.read_csv("gesture_lists/ems-gesture-list.csv")
        self.ems_joint_limits_csv = pd.read_csv("gesture_lists/ems-joint-limits.csv")

        self.default_pulse_count = 20
        self.default_delay = 0.0098
        self.EMS_ch = [[0, 200, 10],
            [1, 200, 6],
            [2, 200, 6],
            [3, 200, 6],
            [4, 200, 6],
            [5, 200, 6],
            [6, 200, 6],
            [7, 200, 6]]
        self.ems_power = 1                      # set as a percentage of calibrated power, where 1 is 100% of calibrated intensityq
        self.ems_device = EMS.autodetect()
        self.load_ems_calibration()
        self.instruction_pause = instruction_pause
        self.debug = debug
        self.sekeleton = skeleton
        self.osc_servers = OSCServers()

    def load_ems_calibration(self):
        try:
            with open("ems_calibration.txt", "r") as file:
                lines = file.readlines()
                for i in range(len(lines)):
                    line = lines[i].strip().split(",")
                    self.EMS_ch[i] = [int(line[0]), int(line[1]), int(line[2])]
            print("EMS_ch", self.EMS_ch)
        except FileNotFoundError:
            print("ems_cali.txt not found, using default values")

    # clamping power
    def set_ems_power(self, power):
        if not (0 < power < 1):
            raise("EMS power out of bound")
        else:
            self.ems_power = power

    def parse_instruction(self, instruction):
        """
        Parse the instruction to extract joint, direction, and target angle.
        Example: <left>[wrist][rotation][counterclockwise][180]
        Returns: handedness (str), joint (str), direction (str), target_angle (int)
        """
        # Default instruction format
        full_match = re.match(r"<(.*?)>\[(.*?)\]\[(.*?)\]\[(.*?)\]\[(.*?)\]", instruction)
        if full_match:
            handedness      = full_match.group(1)
            joint           = full_match.group(2)
            movement        = full_match.group(3)
            direction       = full_match.group(4)
            target_angle    = int(full_match.group(5))
            return handedness, joint, movement, direction, target_angle

        # other instruction format, mostly for "grip"
        simplified_match = re.match(r"<(.*?)>\[(.*?)\]", instruction)
        if simplified_match:
            handedness = simplified_match.group(1)
            joint = simplified_match.group(2)
            if joint == "grip":
                # For grip, we assume no movement and direction
                movement = "flexion"
                direction = "inward"
                target_angle = int(45)
            else:
                movement = None
                direction = None
                target_angle = None
            return handedness, joint, movement, direction, target_angle
    
        raise ValueError(f"Invalid instruction format: {instruction}")
    
    def get_parent_joint(self, joint, movement, direction):
        """
        Get the parent joint for the given joint, movement, and direction.
        Returns: Parent joint name (str).
        """
        parent = self.ems_joint_limits_csv[
            (self.ems_joint_limits_csv["joint"]     == joint) &
            (self.ems_joint_limits_csv["movement"]  == movement) &
            (self.ems_joint_limits_csv["direction"] == direction)
        ]
        if parent.empty:
            raise ValueError(f"No parent joint found for {joint} in {movement}")
        
        parent_joint        = parent.iloc[0]["parent joint"]
        parent_movement     = parent.iloc[0]["parent movement"]
        parent_direction    = parent.iloc[0]["parent direction"]

        return parent_joint, parent_movement, parent_direction
    
    # TODO: select the correct child
    def get_child_joint(self, joint, movement, direction):
        """
        Get the child joint for the given joint, movement, and direction.
        Returns: Parent joint name (str).
        """
        child = self.ems_joint_limits_csv[
            (self.ems_joint_limits_csv["parent joint"]     == joint) &
            (self.ems_joint_limits_csv["parent movement"]  == movement) &
            (self.ems_joint_limits_csv["parent direction"] == direction)
        ]
        if child.empty:
            raise ValueError(f"No child joint found for {joint} in {movement}")
        
        child_joint        = child.iloc[0]["joint"]
        child_movement     = child.iloc[0]["movement"]
        child_direction    = child.iloc[0]["direction"]

        # print(f"[child] joint: {child_joint}, movement: {child_movement}, direction: {child_direction}")
        print(child)

        return child_joint, child_movement, child_direction
        
    def get_joint_limits(self, joint, movement, direction):
        """
        Get the joint limits for the given joint, movement, and direction.
        Returns: Dictionary of joint limits.
        """
        joint_limits = self.ems_joint_limits_csv[
            (self.ems_joint_limits_csv["joint"]     == joint) &
            (self.ems_joint_limits_csv["movement"]  == movement) &
            (self.ems_joint_limits_csv["direction"] == direction)
        ]
        if joint_limits.empty:
            raise ValueError(f"No joint limits found for {joint} in {movement}")

        range_min = 0
        # range_min = joint_limits["range_min"].values[0]
        range_max = joint_limits["range_max"].values[0]

        return {
            "range_min": range_min,
            "range_max": range_max
        }
    
    def get_ems_channel(self, handedness, joint, movement, direction):
        """
        Validate the EMS channel is calibrated for the given joint, movement, and direction.
        Returns channel if calibrated, None otherwise.
        """
        ems_params = self.ems_joint_limits_csv[
            (self.ems_joint_limits_csv["joint"]     == joint) &
            (self.ems_joint_limits_csv["movement"]  == movement) &
            (self.ems_joint_limits_csv["direction"] == direction)
        ]
        if ems_params.empty:
            return None

        try:
            channel = int(ems_params.iloc[0]["channel-right"] if handedness == "right" else ems_params.iloc[0]["channel-left"])
            return channel

        except ValueError as e:
            return None    

    def validate_joint_limits(self, handedness, joint, movement, direction, target_angle):
        """
        Validate the target angle against the joint limits.
        Returns: True if valid, False otherwise.
        """
        if movement is None or direction is None:
            # If movement or direction is not provided, skip validation
            return True
        
        joint_limits = self.ems_joint_limits_csv[
            (self.ems_joint_limits_csv["joint"]     == joint) &
            (self.ems_joint_limits_csv["movement"]  == movement) &
            (self.ems_joint_limits_csv["direction"] == direction)
        ]
        if joint_limits.empty:
            raise ValueError(f"No joint limits found for {joint} in {movement}")

        range_min = 0
        # range_min = joint_limits["range_min"].values[0]
        range_max = joint_limits["range_max"].values[0]

        # Retrieve the current joint angle from skeleton
        current_joint_angles = self.sekeleton.get_joint_angles(handedness, joint, movement)
        if current_joint_angles is None:
            current_joint_angles = 0
            # raise ValueError(f"Current joint angles not found for {joint} -- set to {current_joint_angles}")
        
        print(f"Current joint angles for {joint}: {current_joint_angles}")

        # Validate the target angle against the joint limits
        if not (range_min <= target_angle + current_joint_angles <= abs(range_max)):
            print(f"Target angle {target_angle} for {joint} is out of range. Current angle: {current_joint_angles}, Range: [{range_min}, {range_max}]")
            return False
    
        else: 
            print(f"Target angle {target_angle} for {joint} is within range. Current angle: {current_joint_angles}, Range: [{range_min}, {range_max}]")
            return True

    def get_full_ems_tree(self, handedness, joint, movement, direction, target_angle):
        """
        Recursively retrieve EMS parameters for the given joint and its parent joints.
        Returns: List of EMS parameters for the full joint tree.
        """
        ems_params_tree = []
        # current_ems_params = {}
        # current_target_angle = 0
        if self.debug: print(f"\nProcessing joint: {joint}, movement: {movement}, direction: {direction}, target angle: {target_angle}")

        # Validate joint limits and if EMS channel is calibrated
        if self.validate_joint_limits(handedness, joint, movement, direction, target_angle) and self.get_ems_channel(handedness, joint, movement, direction):
            if self.debug: print(f"Target angle {target_angle} is within range for {joint} in {direction}.")

            # Get EMS parameters for the current joint
            current_ems_params = self.get_ems_parameters(handedness, joint, movement, direction, target_angle)
            current_ems_params["handedness"]    = handedness
            current_ems_params["joint"]         = joint
            current_ems_params["movement"]      = movement
            current_ems_params["direction"]     = direction
            current_ems_params["target_angle"]  = target_angle
            ems_params_tree.append(current_ems_params)

        else:
            print(f"EMS exist: {self.get_ems_channel(handedness, joint, movement, direction)} ")
            if self.get_ems_channel(handedness, joint, movement, direction):
                if self.debug: print(f"Target angle {target_angle} is not within range for {joint} in {direction}. Looking for parent joint.")
                joint_limits = self.get_joint_limits(joint, movement, direction)
                current_target_angle = int(joint_limits["range_max"] * 0.67) 
                parent_target_angle = int(target_angle - current_target_angle)
                if self.debug: print(f"Parent target angle: {parent_target_angle}")

                # Get EMS parameters for the current joint
                current_ems_params = self.get_ems_parameters(handedness, joint, movement, direction, target_angle)
                current_ems_params["handedness"]    = handedness
                current_ems_params["joint"]         = joint
                current_ems_params["movement"]      = movement
                current_ems_params["direction"]     = direction
                current_ems_params["target_angle"]  = current_target_angle
            else:
                if self.debug: print(f"EMS channel not calibrated for {joint} in {direction}. Skipping this joint.")
                parent_target_angle = target_angle

            # Check if the joint has a parent joint
            try:
                parent_joint, parent_movement, parent_direction = self.get_parent_joint(joint, movement, direction)
                if self.debug: print(f"Parent joint: {parent_joint}, movement: {parent_movement}, direction: {parent_direction}")

                # Recursively get EMS parameters for the parent joint.
                # TODO: find strategy to achieve target by combining multiple joints
                # TODO: if parent joint fails, propagate this info back and adjust the angles so total is equal to original target angle
                # parent_ems_params_tree = self.get_full_ems_tree(handedness, parent_joint, parent_movement, parent_direction, target_angle)
                parent_ems_params_tree = self.get_full_ems_tree(handedness, parent_joint, parent_movement, parent_direction, parent_target_angle)
               
                if not parent_ems_params_tree:
                    if self.debug: print(f"No parent joint found for {joint}. Updating target angle to {current_target_angle}.")
                    current_ems_params["target_angle"] = target_angle
                    ems_params_tree.append(current_ems_params)

                else:
                    ems_params_tree.extend(parent_ems_params_tree)
                
            except ValueError as e:
                # No parent joint found, stop recursion
                if current_ems_params != None:
                    current_ems_params["target_angle"] = target_angle
                    ems_params_tree.append(current_ems_params)
                    if self.debug: print(f"No parent joint found for {joint} in {movement}, updating target angle to {current_target_angle}.. Error: {e}")
                else:
                    if self.debug: print(f"No parent joint found for {joint} in {movement}. Error: {e}")

        # Check for child joints. TODO: fix child function first
        # try:
        #     child_joints = self.get_child_joint(joint, movement, direction)
        #     for child_joint, child_movement, child_direction in child_joints:
        #         if self.debug: print(f"Child joint: {child_joint}, movement: {child_movement}, direction: {child_direction}")

        #         # Recursively get EMS parameters for the child joint
        #         child_ems_params_tree = self.get_full_ems_tree(handedness, child_joint, child_movement, child_direction, target_angle)
        #         ems_params_tree.extend(child_ems_params_tree)

        # except ValueError as e:
        #     # No child joints found
        #     if self.debug: print(f"No child joints found for {joint} in {movement}. Error: {e}")
            
        return ems_params_tree

    def get_ems_parameters(self, handedness, joint, movement, direction, target_angle):
        """
        Retrieve EMS parameters for the given joint and direction.
        Returns: Dictionary of EMS parameters.
        """
        ems_params = self.ems_joint_limits_csv[
            (self.ems_joint_limits_csv["joint"]     == joint) &
            (self.ems_joint_limits_csv["movement"]  == movement)  &
            (self.ems_joint_limits_csv["direction"] == direction) 
        ]
        if ems_params.empty:
            raise ValueError(f"No EMS gesture found for {joint} in {direction}")
        
        # print("ems_params", ems_params)
        channel         = int(ems_params.iloc[0]["channel-right"] if handedness == "right" else ems_params.iloc[0]["channel-left"])
        pulse_width     = int(ems_params.iloc[0]["pulse_width"] if not pd.isna(ems_params.iloc[0]["pulse_width"]) else self.EMS_ch[channel][1])
        intensity       = int(ems_params.iloc[0]["intensity"] if not pd.isna(ems_params.iloc[0]["intensity"]) else self.EMS_ch[channel][2])
        pulse_count     = int(ems_params.iloc[0]["pulse_count"] if not pd.isna(ems_params.iloc[0]["pulse_count"]) else self.default_pulse_count)
        delay           = float(ems_params.iloc[0]["delay"] if not pd.isna(ems_params.iloc[0]["delay"]) else self.default_delay)

        if self.debug: print(f'{channel} {pulse_width} {intensity} {pulse_count} {delay}')
        return {
            "channel"       : channel,
            "pulse_width"   : pulse_width,
            "intensity"     : intensity,
            "pulse_count"   : pulse_count,
            "delay"         : delay,
        }

    # main processing function
    def process_instructions(self, instructions):
        """
        Process instructions to validate and execute EMS stimulation.
        Includes parent joints in the EMS parameters tree.
        """
        output_log = ""
        for instruction in instructions.split("\n"):
            try:
                # Parse the instruction
                handedness, joint, movement, direction, target_angle = self.parse_instruction(instruction)
                print(f"Parsed Instruction: Handedness={handedness}, Joint={joint}, Direction={direction}, Movement={movement}, Target Angle={target_angle}")
                output_log += f"Parsed Instruction: Handedness={handedness}, Joint={joint}, Direction={direction}, Movement={movement}, Target Angle={target_angle}\n"

                # Get EMS parameters for full joint tree according to end angle
                ems_params = self.get_full_ems_tree(handedness, joint, movement, direction, target_angle)
                ems_params.reverse() # flip the order, starting from child to parent
                
                # just for printing nice format
                ems_params_df = pd.DataFrame(ems_params)
                print(f"EMS Parameters Tree: {ems_params_df}")
                output_log += f"EMS Parameters Tree: {ems_params}\n"

                self.osc_servers.send_stim_status("stim_on")
                # Stimulate using EMS
                for params in ems_params:
                    self.osc_servers.send_ems_movements(params["handedness"], params["joint"], params["direction"])

                    adjusted_intensity = int(params["intensity"] * self.ems_power)

                    if adjusted_intensity <= 0: # lower bound always 1
                        adjusted_intensity = 1
                    elif adjusted_intensity > params["intensity"]: # clamp upper bound
                        adjusted_intensity = params["intensity"]

                    self.ems_device.pulsed_stimulate(
                        channel         = params["channel"],
                        pulse_width     = params["pulse_width"],
                        intensity       = adjusted_intensity,
                        pulse_count     = params["pulse_count"],
                        delay           = params["delay"],
                    )
                    print(f"Stimulated {params['joint']} {params['movement']} with target angle {params['target_angle']}.")
                    output_log += f"Stimulated {params['joint']} {params['movement']} with target angle {params['target_angle']}.\n\n"
                
                self.osc_servers.send_stim_status("stim_off")

            except ValueError as e:
                print(f"Error processing instruction: {e}")
                output_log += f"Error processing instruction: {e}\n"

        return output_log


    # old function
    def link_ems_control(self, gesture_response):
        found_gestures = re.findall(r"\<.*?\]", gesture_response)
        for i in range(len(found_gestures)):
            if i != 0:
                speech_engine.speak("next")

            current_full_gesture = found_gestures[i]
            print(f"gesture{i}:", current_full_gesture)

            # parse handedness
            handedness = re.findall(r"\<.*?\>", current_full_gesture)[0]
            print(f"handedness{i}", handedness)

            # parse gesture
            current_llm_gesture = re.findall(r"\[.*?\]", current_full_gesture)[0]
            current_llm_gesture = current_llm_gesture[1:-1]
            print("current llm gesture: ", current_llm_gesture)
            
            # handle "skip"
            if current_llm_gesture == "skip":
                voice_instruction = gesture_response.split("<")[0]
                speech_engine.speak("please do the following gesture by yourself"+voice_instruction)
            # other gestures link from llm-csv to ems-csv
            else:
                speech_engine.speak(current_full_gesture)
                user_comfirmation = ""
                while user_comfirmation == "":                    
                    if self.instruction_pause:
                        user_comfirmation = speech_engine.live_listening()
                
                    if "continue" in user_comfirmation or self.instruction_pause == False:
                        # match current_llm_gesture to the "gesture" column of llm-gesture-list.csv
                        llm_gesture = self.llm_gestures_csv[self.llm_gestures_csv["gesture"] == current_llm_gesture]
                        print("matched llm gesture: ", llm_gesture)
                        
                        # link to ems_gesture_list
                        ems_gesture = llm_gesture["EMS gesture"].values[0]
                        print("ems gesture: ", ems_gesture)

                        stimulation_parameters = self.ems_gestures_csv[self.ems_gestures_csv["gesture"]==ems_gesture]
                        print(stimulation_parameters)
                        
                        
                        # stimulate
                        time.sleep(1.5)
                        set_channel         = int(stimulation_parameters.iloc[0]["channel-right"] if handedness == "<right>" else stimulation_parameters.iloc[0]["channel-left"])
                        set_pulse_width     = stimulation_parameters.iloc[0]["pulse_width"] if not pd.isna(stimulation_parameters.iloc[0]["pulse_width"]) else self.EMS_ch[set_channel][1]
                        set_intensity       = stimulation_parameters.iloc[0]["intensity"] if not pd.isna(stimulation_parameters.iloc[0]["intensity"]) else self.EMS_ch[set_channel][2]
                        set_pulse_count     = stimulation_parameters.iloc[0]["pulse_count"] if not pd.isna(stimulation_parameters.iloc[0]["pulse_count"]) else self.default_pulse_count
                        set_delay           = stimulation_parameters.iloc[0]["delay"] if not pd.isna(stimulation_parameters.iloc[0]["delay"]) else self.default_delay
                        self.ems_device.pulsed_stimulate(channel=set_channel, pulse_width=set_pulse_width, intensity=set_intensity, pulse_count=set_pulse_count, delay=set_delay)
                        print("DEBUG", "channel: ", set_channel, "intensity", set_intensity)
                    
                    elif "don't" in user_comfirmation:
                        speech_engine.speak("skipping this gesture")

    def direct_stimulation(self, set_channel, set_intensity, set_pulse_count=20, set_delay=0.0098, set_pulse_width = 200):
        self.ems_device.pulsed_stimulate(channel=set_channel, pulse_width=set_pulse_width, intensity=set_intensity, pulse_count=set_pulse_count, delay=set_delay)
                        
# Example usage
if __name__ == "__main__":
    skeleton = MocapOSCServer()
    gesture_processor = GestureProcessor(skeleton=skeleton, instruction_pause=False, debug=True)
    # gesture_processor.link_ems_control(test_skip)
    # gesture_processor.process_instructions(single_gestures_response)
    # print(" ----- ")
    # gesture_processor.process_instructions("<left>[grip]")
    # gesture_processor.process_instructions("<right>[wrist][rotation][outward][180]")
    # gesture_processor.process_instructions("<right>[grip][flexion][forward][90]")
    gesture_processor.process_instructions("<right>[wrist][abduction (twist)][inward][45]")

    # gesture_processor.validate_joint_limits("wrist", "fel", direction, target_angle):
