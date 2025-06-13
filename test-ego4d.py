import subprocess
# {'example image': ['spoken request','handedness', 'hands_obj','skeleton_direction']}

skeleton_direction_1 = "Directions: {'Skeleton': 'center', 'Hips': 'center', 'Spine': 'center', 'LeftUpLeg': 'left', 'LeftLeg': 'left', 'LeftFoot': 'left', 'RightUpLeg': 'right', 'RightLeg': 'right', 'RightFoot': 'right', 'Neck': 'center', 'Head': 'center', 'LeftShoulder': 'down', 'LeftArm': 'down', 'LeftForeArm': 'down', 'LeftHand': 'down', 'LeftHandIndex1': 'down', 'LeftHandMiddle1': 'down', 'LeftHandPinky1': 'down', 'LeftHandRing1': 'down', 'LeftHandThumb1': 'down', 'RightShoulder': 'down', 'RightArm': 'backward', 'RightForeArm': 'backward', 'RightHand': 'upward', 'RightHandIndex1': 'upward', 'RightHandMiddle1': 'upward', 'RightHandPinky1': 'upward', 'RightHandRing1': 'upward', 'RightHandThumb1': 'upward'}"


test_examples = {
    'chop-onion': ['Help me chop the onion', 'none', 'none',skeleton_direction_1],
    'chop-saw': ['Help me cut this wood', 'none', 'none', skeleton_direction_1],
    'trimming': ['Help me', 'none', 'none', skeleton_direction_1],
    'open-bucket': ['Help me', 'none', 'none',skeleton_direction_1]
}
# iterate through the dictionary
for example, inputs in test_examples.items():
    spoken_request = inputs[0]
    handedness = inputs[1]
    hands_obj = inputs[2]
    skeleton_direction = inputs[3]
    print(f"testing example: {example}, spoken request: {spoken_request}, handedness: {handedness}, hands_obj: {hands_obj}")
    subprocess.run(['python', 'llm_gesture_clean.py', example, spoken_request, handedness, hands_obj, skeleton_direction])