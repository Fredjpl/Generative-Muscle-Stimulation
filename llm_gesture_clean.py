import base64
import time
import openai
import pandas as pd
import cv2
import geocoder
from speech_engine import SpeechEngine
# from gesture_processor import GestureProcessor
from mocap_oscserver import MocapOSCServer
import os
import sys




llm_model = "o1"



class LLMGesture:
    def __init__(self):
        self.ems_joint_limits = pd.read_csv('gesture_lists/ems-joint-limits-clean.csv')
        # keep the first three columns
        self.joint_list = self.ems_joint_limits.iloc[:, :3]
        print(self.joint_list)

        # read user profile
        f = open("user_profile/user_profile_1.txt", "r")
        # user_location = f'I am located in {geocoder.ip('me').city}, {geocoder.ip("me").country}'
        user_location = f'I am located in Berlin, Germany'

        self.user_profile = f.readlines() + [user_location]
        f.close()
        print("user profile: ", self.user_profile)
        
        # lab open ai key: TODO: remove it
        self.api_key = "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"
        openai.api_key = self.api_key

        self.speech_engine = SpeechEngine()
        self.skeleton = MocapOSCServer()
        # self.gesture_processor = GestureProcessor(skeleton = self.skeleton, instruction_pause=False)


    # Main LLM reasoning function for the whole pipeline
    def process_image_and_task(self, path, image, task, handedness, hands_obj, skeleton_direction, checkpoints=True, load_cache=False):
        with open(filename, 'a', encoding='utf-8') as f :
            f.write(f"Input user request: \n {task}\n\n")
        
        with open(filename, 'a', encoding='utf-8') as f :
            f.write(f"Handedness: \n {handedness}\n\n")
        
        with open(filename, 'a', encoding='utf-8') as f :
            f.write("LLM output")
        
        if load_cache:
            last_cached_output = self.load_cache_output("test-results/full-demo/")
            print("Try again with prior failed reasoning")
        else:
            last_cached_output = "No prior failed reasoning result. Start fresh."
            print(last_cached_output)

        print("Recognition:")
        recognition_response = self.recognize_object(image, task, handedness, hands_obj,last_cached_output)
        print(recognition_response)
        with open(filename, 'a', encoding='utf-8') as f :
            f.write(f"LLM output (recognition): \n {recognition_response}\n\n")

        print("Movements:")
        movements_response = self.generate_movements(recognition_response, limb_directions=skeleton_direction, cached_output = last_cached_output)
        print(movements_response)
        with open(filename, 'a', encoding='utf-8') as f :
            f.write(f"LLM output (movement): \n {movements_response}\n\n")

        print("Gesture:")
        gestures_response = self.generate_gestures_oneshot(movements_response)
        print(gestures_response)

        with open(filename, 'a', encoding='utf-8') as f :
            f.write(f"input pose (skeleton directions): \n {skeleton_direction}\n\n")

        with open(filename, 'a', encoding='utf-8') as f :
            f.write(f"LLM output (gesture without checkpoint): \n {gestures_response}\n\n")


        if checkpoints:
            user_comfirmation = ""
            while user_comfirmation == "":
                user_comfirmation = self.speech_engine.live_listening()
            
            if "continue" in user_comfirmation:
                control = True
                count = 2
            elif "dont" in user_comfirmation:
                control = False
                self.speech_engine.speak("Bye bye")

            prev = ""
            
            steps = list(map(str.strip, movements_response.strip().split("\n"))) 
            steps = [step.split(". ")[-1] for step in steps]

            while control:
                self.speech_engine.speak("Checkpoint in 3")
                time.sleep(1)
                self.speech_engine.speak("2")
                time.sleep(1)
                self.speech_engine.speak("1")
                time.sleep(1)
                
                if path != "images/image1.jpg":
                    path = "images/image" + str(count) + ".jpg"
                    path = self.capture_image_oneshot(path)
                    # path = self.capture_image(path)
                    count += 1
                    with open(path, "rb") as img_file:
                        image = base64.b64encode(img_file.read()).decode('utf-8')
                else:
                    with open(path, "rb") as img_file:
                        image = base64.b64encode(img_file.read()).decode('utf-8')
                    path = "images/image2.jpg"

                print("Checkpoints:")

                checkpoints_response = self.generate_checkpoints(image, task, movements_response, handedness)
                print(checkpoints_response)

                if prev == checkpoints_response:
                    step_index = steps.index(checkpoints_response.strip().split("\n")[-1])
                    print("step index: ", step_index)
                    if step_index + 1 == len(steps):
                        checkpoints_response = "done"
                    elif not steps[step_index + 1]:
                        checkpoints_response  = steps[step_index + 5]
                    else:
                        checkpoints_response  = steps[step_index + 1]

                print(checkpoints_response)

                with open(filename, 'a', encoding='utf-8') as f :
                    f.write(f"LLM output (checkpoint): \n {checkpoints_response}\n\n")

                prev =  checkpoints_response           
                
                checkpoints_response = checkpoints_response.lower()
                if (checkpoints_response.strip() == "done") or "done" in checkpoints_response:
                    control = False
                    return False
                
                print("Gesture:")
                gestures_response = self.generate_gestures(checkpoints_response)
                print(gestures_response)

                with open(filename, 'a', encoding='utf-8') as f :
                    f.write(f"LLM output (gesture): \n {gestures_response}\n\n")
                

                ems_params = self.gesture_processor.process_instructions(gestures_response)
                print("EMS params", ems_params)
                with open(filename, 'a', encoding='utf-8') as f :
                    f.write("END OF LLM")
                with open(filename, 'a', encoding='utf-8') as f :
                    f.write(f"EMS params: \n {ems_params}\n\n")


        else:
            ems_params = self.gesture_processor.process_instructions(gestures_response)
            print("EMS params", ems_params)
            with open(filename, 'a') as f:
                f.write("END OF LLM \n\n")
            with open(filename, 'a') as f:
                f.write(f"EMS params: \n {ems_params}\n\n")

        return recognition_response, movements_response, gestures_response
        
    
    def capture_image_oneshot(self, output_path="image.jpg"):
        """Capture an image from the webcam and save it to a file."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open webcam")
    
        print("Capturing image...")
        ret, frame = cap.read()
        if not ret:
            raise Exception("Failed to capture frame from webcam")

        cv2.imwrite(output_path, frame)
        print(f"Image captured and saved to {output_path}")
    
        cap.release()
        cv2.destroyAllWindows()
        return output_path
    
    def capture_image(self, output_path="image.jpg"):
     """Capture an image from the webcam and save it to a file."""
     cap = cv2.VideoCapture(0)
     if not cap.isOpened():
         raise Exception("Could not open webcam")
 
     print("Press 'Space' to capture an image or 'Esc' to exit.")
     while True:
         ret, frame = cap.read()
         if not ret:
             raise Exception("Failed to capture frame from webcam")
 
         cv2.imshow("Webcam", frame)
         key = cv2.waitKey(1)
 
         if key == 27:  # Esc key to exit
             print("Exiting without capturing.")
             cap.release()
             cv2.destroyAllWindows()
             return None
         elif key == 32:  # Space key to capture
             cv2.imwrite(output_path, frame)
             print(f"Image captured and saved to {output_path}")
             break
 
     cap.release()
     cv2.destroyAllWindows()
     return output_path


    def recognize_object(self, image, task, handedness, hands_obj, cached_output= ""):
        print("LLM recognizing object...")
        recognition = openai.chat.completions.create(
            model=llm_model,
            reasoning_effort="medium",
            messages=[
                {
                    "role": "system", 
                    "content": 
                    f'''
                    You will be provided:
                    "the handedness that the user is holding or touching the object": {handedness} and {hands_obj}. Use this information to decide which hand/limb to use in the movement instructions. If the object in the image is related to the user's goal, use the provided handedness for movement instructions. If the object in the image is unrelated to the user's goal, use the other hand.
                    the given handedness: {handedness}, and the user's customized preference in the user profile: {self.user_profile} to determine which limb and handedness should be used in the movement instructions. Prioritize the image and the given handedness: {handedness}. If the handedness is unclear, or both hands will work, refer to customized preference in the user profile to determine the handedness.
                    1. Analyze the provided image where hands are interacting with an object. 
                    2. Identify the object being held or manipulated by the hands.
                    3. If the user profile is not "None,” use the location information in the user profile: {self.user_profile} for your reasoning to determine the object/mechanism and functionality of the object.
                    4. Describe the object in detail, including its physical characteristics such as shape, color, size, and notable features.
                    5. Explain the object’s functionality, detailing what it can do and how it is typically used.
                    6. Rule out the functionality or mechanism that does not work with prior failed reasoning result: {cached_output} (see LLM output (recognition)). Do not repeat the prior failed reasoning result. Try another mechanism or functionality.
                    7. If the object has multiple mechanisms or functions, propose only one that makes the most sense in this context. Do not list all the possible functions or optional functions. 
                    8. If the user’s hands do not interact with a specific object in the image, use the contextual information from the image and the user's request (task) to provide relevant information.
                    9. Review user profile:  {self.user_profile}, to give personalized information. 

                    '''
                },
                {
                    "role": "user",
                    "content": [
                        {  
                        "type": "text",
                        "text": "User Task: " + task + "\n" + "User Profile: " + str(self.user_profile) + "\n"
                        },
                        {
                        "type": "image_url",
                        "image_url": {
                            "url":  f"data:image/jpeg;base64,{image}"
                            } 
                        }
                    ],
                },
            ],
        )
        return recognition.choices[0].message.content

    # remove image and task?
    def generate_movements(self, recognition_response, limb_directions, cached_output= ""):
        print("LLM planning movement...")
        movements = openai.chat.completions.create(
            model=llm_model,
            reasoning_effort="medium",
            messages=[
                {
                    "role": "system", 
                    "content": 
                    f'''
                    You will be provided:
                    "descriptions of the object and its mechanism": {recognition_response}. Use this information to formulate the movement instructions.
                    "current body position described by where the limbs are pointing. Note that the wrist is referring to the thumb to point": {limb_directions}
                    "Prior failed reasoning result": {cached_output}

                    <instructions>
                        You are a physical assistant that can control the user's body. 
                        Your task is to analyze the descriptions of the object and its mechanism, and the user's goal to generate step-by-step movement instructions to interact with the object to achieve their goal. 
                        Follow these steps to complete the task:
                        1. Analyze the image
                        2. Use the current body position described by {limb_directions} to determine the starting position of each limb. Avoid redundant movements by considering the current orientation and position of the limbs. For example:
                        - If the limb is already in the required position, skip unnecessary steps.
                        - If the limb is pointing in the opposite direction, include a step to reorient it before proceeding with the task.
                        3. The image is depicting the user's current state. If you determine that the user has already achieved their goal, simply output "done.” If you see the user has already done part of the steps, do not repeat those steps.
                        4. Rule out the functionality or mechanism that does not work with prior failed reasoning result: {cached_output} (see LLM output (movement):). Do NOT repeat the prior failed reasoning result. Try another mechanism or functionality. 
                        5. Ensure your instructions are easy to follow and consider the user's perspective.
                        6. Ensure each movement instruction is an actionable movement instruction that is directly related to the user's goal. Remove all the adjectives that are not critical.
                        7. Be specific and reply in a formatted list as follows: 
                        - 'limb:' followed by one word that details which limb of the body the user is using to perform the task (e.g., foot or hand).
                        - 'handedness': followed by one word that details on which side of the body this limb is, with respect to the person's point of view in the photo (e.g., left or right). 
                        - 'movement': start with handness like "Left or Right hand: ..." and then describe in detail how the limb is moving step by step. (e.g., firstly, which joint, flexion, or extension, and secondly ...) Include the required angle of the movement to complete each step (e.g., turn the object for how many degrees.)
                        If it's a sequence of movements, list all the required movements in sequence; if it's a bimanual or bipedal task, output movements for each hand or each foot. 
                        8. Ensure your output contains no XML tags or additional formatting; it should be a plain text response.
                        9. Carefully consider the angle of the movement and the direction of the movement by explicitly stating the angle and direction of the movement if it is essential to achieve the goal.
                    </instructions>

                    Example:
                    <input>
                        Image: Left hand is holding a pill bottle. Right hand is free.
                        Task: Open this pill bottle.
                        Handedness: left hand
                    </input>
                    <output>
                        - limb: hand  
                        handedness: left
                        movement:
                            1. Left hand: Grip the bottle to stabilize it.

                        - limb: hand
                        handedness: right
                        movement:
                            1. Right hand: Place the palm and fingers over the white cap.
                            2. Right hand: Press down firmly with your palm, applying downward pressure.
                            3. Right hand: While pressing, twist the cap counterclockwise for 45 degrees.
                            4. Right hand: Continue turning until the cap loosens and can be lifted from the bottle.
                    </output>
                    '''
                },
                {
                    "role": "user",
                    "content": [
                        {  
                        "type": "text",
                        "text": "User Profile: " + str(self.user_profile) + "\n"
                        },

                    ],
                },
            ],

        )
        return movements.choices[0].message.content
    
    def generate_checkpoints(self, image, task, movements_response, handedness):
        print("LLM generating checkpoint...")
        checkpoints =  openai.chat.completions.create(
                model=llm_model,
                messages=[
                    {
                        "role": "system", 
                        "content": 
                        f'''
                         <instructions>
                            You will receive an image depicting the user's current state, a description of the user's task, and a sequence of movements the user needs to make. 
                            Your objective is to analyze the provided information and choose the next step from the movements list with the handedness. 

                            Follow these steps to complete the task:
                            1. Examine the image to understand the user's current state and context.
                            2. Follow the given handedness information: {handedness} to determine Left hand or Right hand. 
                            3. Read the description of the user's task to understand what they are trying to achieve.
                            4. Review the sequence of movements to identify what actions the user has already taken.
                            5. Based on your analysis, choose the next step from the movements list for the user to continue towards their goal.
                            6. If you determine that the user has already achieved their goal, simply output "done".
                            7. Ensure that your output contains no XML tags or additional formatting; it should be a plain text response.
                        </instructions>
                        <example>
                                <input>
                                    Image: Left hand is holding a pill bottle. Right hand is free.
                                    Task: Open this pill bottle.
                                    Movements: 
                                    - limb: hand  
                                    handedness: left
                                    movement:
                                        1. Left hand: Wrap the fingers around the orange bottle, letting the bottle rest in the palm.
                                        2. Left hand: Stabilize the bottle with a gentle, steady grip—avoid squeezing too tightly.
                                        3. Left hand: Maintain this firm but comfortable hold throughout the opening process.

                                    - limb: hand
                                    handedness: right
                                    movement:
                                        1. Right hand: Place the palm and fingers over the white cap.
                                        2. Right hand: Press down firmly with your palm, applying downward pressure.
                                        3. Right hand: While pressing, twist the cap counterclockwise.
                                        4. Right hand: Continue turning until the cap loosens and can be lifted away from the bottle.
                                </input>
                                <output>
                                    Right hand: Place the palm and fingers over the white cap.
                                </output>
                        </example>
                        '''
                },
                {
                    "role": "user",
                    "content": [
                        {
                        "type": "image_url",
                        "image_url": {
                            "url":  f"data:image/jpeg;base64,{image}"
                            } 
                        },
                        {  
                        "type": "text",
                        "text": 
                        f'''
                        Task: {task}
                        Movements: {movements_response}
                        '''
                        }
                    ],
                },
                ]
            )
      
        return "checkpoint: \n" + checkpoints.choices[0].message.content

    def generate_gestures(self, checkpoints_response, limb_directions=None):
        print("LLM analyzing gesture...")
        gestures = openai.chat.completions.create(
        model=llm_model,
        reasoning_effort="medium",
        messages=[
                {
                    "role": "system", 
                    "content": 
                    f'''
                    You will be provided:
                    "a movement for operating an object or achieving a goal": {checkpoints_response}
                    "a joint_list (the movements of the joints, and the direction that the joints can turn relative to the neutral pose)”: {self.joint_list}
                    "the neutral pose”: it is a person standing upright with the arms resting beside the thighs. Use this neutral pose as a baseline (0 degrees) to calculate the direction and angle of the movements.
                    "current body position described by where the limbs are pointing. Note that the wrist is referring to the thumb to point": {limb_directions}

                    <instructions>
                        You are tasked with translating a list of movement instructions into specific muscle movements using a provided joint list. Follow these steps to complete the task:

                        1. Carefully read the provided movement instructions, which include details about the limb, handedness, and specific actions to be performed.
                        2. For each movement, use the joint_list: {self.joint_list} to describe the the movement in the following format: <handedness. i.e., left, right>[the joint that needs move. e.g., wrist, elbow, shoulder, heel.][the movement of that joint. e.g., extension, flexion, abduction.][the direction of that movement relative to the neutral pose][how many degrees the joint needs to move in that direction you described. Ranging from 0-360. e.g., 180, 90, 45.] Ensure that each movement is directly linked to the action it corresponds to. Ensure you only use the [joint][movement][direction] pairs in the joint_list.
                        3.1 Review the movement instruction again to determine if the movement instruction should be translated into "grip" or "grasp" in muscle movement. If it should, choose  <handedness>[grip][flexion][direction][angle], instead of 
                        <right>[index][flexion][inward][15]  
                        <right>[middle][flexion][inward][15]  
                        <right>[ring][flexion][inward][15]  
                        <right>[pinky][flexion][inward][15]. Same for "release": instead of choosing multiple finger joint movements, choose <handedness>[release][extension][direction][angle]
                        3.2 Review the movement instruction again to determine if the movement instruction should be translated into "pull" in muscle movement. If it should, choose  <handedness>[pull][extension][backward][angle], instead of
                        <right>[wrist][pronation][inward][25]  
                        <right>[elbow][flexion][upward][30]. Same for "push": instead of choosing multiple arm joint movements, choose <handedness>[arm][push][flexion][angle]
                        4. Structure your output for each movement in the following format: <handedness>[joint][movement][direction][angle]. If more than one joint movement is required, use the following format: <handedness>[joint][movement][direction][angle] \n <handedness>[joint][movement][direction][angle]. 
                        6. Make sure each joint movement has all five parameters in the format of  <handedness>[joint][movement][direction][angle]. 
                        7. Ensure that your output does not contain any XML tags but keep the "<left>" or "<right>" tag to indicate the handedness of the movement and should not be modified.
                        8. Provide a clear separation between different limbs and movements in your output for better readability.
                        9. Review your output to ensure that you only choose the joint movement from the given list of joints. Also ensure accuracy and completeness before finalizing it.
                        10. If no direct mapping is found, return <skip>.
                        11. Consider the current body position described by {limb_directions} to determine which body part should move in which direction.
                    </instructions> 

                    <examples>
                        <example>
                            <input>
                                Right hand: Twist the knob counterclockwise.  
                            </input>
                            <output>
                                <right>[abduction (twist)][inward][5]
                            </output>
                        </example>
                        
                        <example>
                            <input>
                                Left hand: Slightly bend your left elbow so your left hand is positioned in front of you.  
                            </input>
                            <output>
                                <left>[elbow][flexion][upward][45]
                            </output>
                        </example>
                   </examples>

                    '''
                },
                {
                    "role": "user",
                    "content": [
                        {  
                        "type": "text",
                        "text": 
                        f'''
                        Describe the movement in the following format: <handedness. i.e., left, right>[the joint that needs move. e.g., wrist, elbow, shoulder, heel.][the movement of that joint. e.g., extension, flexion, abduction.][the direction of that movement relative to the neutral pose][how many degrees the joint needs to move in that direction you described. Ranging from 0-360. e.g., 180, 90, 45.] to achieve the following movements {checkpoints_response}. 
                        If it's a bimanual or bipedal task, then output movement description for each hand or each foot.

                        '''
                        },
                    ],
                },
            ]
        )
        
        return "llm output gestures: \n" + gestures.choices[0].message.content


    def generate_gestures_oneshot(self, movement_response, limb_directions=None):
        print("LLM analyzing gesture (without checkpoint)...")
        gestures = openai.chat.completions.create(
        reasoning_effort="medium",
        model=llm_model,
        messages=[
                {
                    "role": "system", 
                    "content": 
                    f'''
                    You will be provided:
                  You will be provided:
                    "a movement for operating an object or achieving a goal": {movement_response}
                    "a joint_list (the movements of the joints, and the direction that the joints can turn relative to the neutral pose)”: {self.joint_list}
                    "the neutral pose”: it is a person standing upright with the arms resting beside the thighs. Use this neutral pose as a baseline (0 degrees) to calculate the direction and angle of the movements.
                    "current body position described by where the limbs are pointing. Note that the wrist is referring to the thumb to point": {limb_directions}
                    
                    <instructions>
                        You are tasked with translating a list of movement instructions into specific muscle movements using a provided joint list. Follow these steps to complete the task:

                        1. Carefully read the provided movement instructions, which include details about the limb, handedness, and specific actions to be performed.
                        2. For each movement, use the joint_list: {self.joint_list} to describe the the movement in the following format: <handedness. i.e., left, right>[the joint that needs move. e.g., wrist, elbow, shoulder, heel.][the movement of that joint. e.g., extension, flexion, abduction.][the direction of that movement relative to the neutral pose][how many degrees the joint needs to move in that direction you described. Ranging from 0-360. e.g., 180, 90, 45.] Ensure that each movement is directly linked to the action it corresponds to. Ensure you only use the [joint][movement][direction] pairs in the joint_list.
                        3.1 Review the movement instruction again to determine if the movement instruction should be translated into "grip" or "grasp" in muscle movement. If it should, choose  <handedness>[grip][flexion][direction][angle], instead of 
                        <right>[index][flexion][inward][15]  
                        <right>[middle][flexion][inward][15]  
                        <right>[ring][flexion][inward][15]  
                        <right>[pinky][flexion][inward][15]. Same for "release": instead of choosing multiple finger joint movements, choose <handedness>[release][extension][direction][angle]
                        3.2 Review the movement instruction again to determine if the movement instruction should be translated into "pull" in muscle movement. If it should, choose  <handedness>[pull][extension][backward][angle], instead of
                        <right>[wrist][pronation][inward][25]  
                        <right>[elbow][flexion][upward][30]. Same for "push": instead of choosing multiple arm joint movements, choose <handedness>[arm][push][flexion][angle]
                        4. Structure your output for each movement in the following format: <handedness>[joint][movement][direction][angle]. If more than one joint movement is required, use the following format: <handedness>[joint][movement][direction][angle] \n <handedness>[joint][movement][direction][angle]. 
                        6. Make sure each joint movement has all five parameters in the format of  <handedness>[joint][movement][direction][angle]. 
                        7. Ensure that your output does not contain any XML tags but keep the "<left>" or "<right>" tag to indicate the handedness of the movement and should not be modified.
                        8. Provide a clear separation between different limbs and movements in your output for better readability.
                        9. Review your output to ensure that you only choose the joint movement from the given list of joints. Also ensure accuracy and completeness before finalizing it.
                        10. If no direct mapping is found, return <skip>.
                        11. Consider the current body position described by {limb_directions} to determine which body part should move in which direction.
                    </instructions> 

                        <example>
                            <input>
                                - limb: hand
                                handedness: right
                                movement: 
                                1. Twist the knob counterclockwise.

                                - limb: hand
                                handedness: left
                                movement:
                                1. Lift handle upward.
                                2. Curl your thumb and fingers around the sides of the handle, maintaining a comfortable grip.
                            </input>
                            <output>
                                Right hand:
                                1. Twist the knob counterclockwise.
                                <right>[abduction (twist)][inward][5]   
                               
                                Left hand:
                                1. Lift handle upward.
                                <left>[elbow][flexion][upward][30]  
                                <left>[shoulder][flexion][upward][15]
                                2. Curl your thumb and fingers around the sides of the handle, maintaining a comfortable grip.
                                <left>[grip][flexion][inward][15]
                               
                            </output>
                        </example>
                    '''
                },
                {
                    "role": "user",
                    "content": [
                        {  
                        "type": "text",
                        "text": 
                        f'''
                        Describe the movement in the following format: <handedness. i.e., left, right>[the joint that needs move. e.g., wrist, elbow, shoulder, heel.][the movement of that joint. e.g., extension, flexion, abduction.][the direction of that movement relative to the neutral pose][how many degrees the joint needs to move in that direction you described. Ranging from 0-360. e.g., 180, 90, 45.] to achieve the following movements {movement_response}. 
                        If it's a bimanual or bipedal task, then output movement description for each hand or each foot.
                        '''
                        },
                    ],
                },
            ]
        )
        
        return "llm output gestures (without checkpoint): \n" + gestures.choices[0].message.content
    
    
    def load_cache_output(self, path):
        files = os.listdir(path)
        files.sort()
        latest_file = files[-2] # exclude current file
        print("latest file: ", latest_file)
        # read the last file in the folder
        with open(os.path.join(path, latest_file), 'r') as file:
            lines = file.readlines()
            if "END OF LLM \n" in lines:
                # find index of the line that contains "END OF LLM"
                end_of_llm_index = lines.index("END OF LLM \n")
                # get the lines before "END OF LLM"
                llm_cache_ouput = lines[:end_of_llm_index]
                # concatenate the lines to a string
                llm_cache_ouput = '\n'.join(llm_cache_ouput)
            else:
                llm_cache_ouput = '\n'.join(lines)
        return llm_cache_ouput


# Example usage
if __name__ == "__main__":

    benchmark = LLMGesture()
    

    example = sys.argv[1]
    image_path = 'images/rebuttal/' + example +'.png'
    user_request = sys.argv[2]

    handedness = sys.argv[3] #"right hand"
    hands_obj = sys.argv[4] #"object in right hand"
    skeleton_direction = sys.argv[5]

    # datalog
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # save the input and ouptut to a file
    filename = f"test-results/ego4d-examples/{example}_{timestamp}.txt"
    # create a file and write the input and output to it
    f = open(filename, "x")
    with open(filename, 'a', encoding='utf-8') as f :
        f.write(f"LLM model: \n {llm_model}\n\n")

    
   
    with open(filename, 'a', encoding='utf-8') as f :
            f.write("clean, new architecture \n")

    with open(filename, 'a', encoding='utf-8') as f :
            f.write(f"hands_obj: \n {hands_obj}\n\n")
    
    
        
            

    
    print("testing full pipeline")
    path = benchmark.capture_image_oneshot('images/image1.jpg')
    

    
    with open(image_path, "rb") as img_file:
        image = base64.b64encode(img_file.read()).decode('utf-8')
    
    benchmark.process_image_and_task(path, image, user_request,handedness, hands_obj, skeleton_direction, checkpoints=False, load_cache=False)

    
    
    
    
