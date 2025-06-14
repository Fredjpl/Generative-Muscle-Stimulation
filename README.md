# Generative Muscle Stimulation: Physical Assistance by Constraining Multimodal-AI with Biomechanical Knowledge

This repository contains the implementation of a system that assists users with electrical muscle stimulation (EMS) by generating contextual movement instructions using multimodal AI constrained by biomechanical knowledge.

## Overview

Unlike traditional EMS systems that rely on fixed, pre-programmed movements, our system generates muscle stimulation instructions dynamically based on:
- User's spoken requests
- Point-of-view (POV) images
- Current body pose
- Contextual information (location, object detection)
- Biomechanical constraints

The system uses computer vision and large language models (LLMs) to understand the user's context and generate appropriate movement instructions, which are then translated into safe and executable EMS commands.

## Key Features

- **Context-aware assistance**: Generates movement instructions based on visual context and user requests
- **Biomechanical safety**: Constrains AI outputs using joint limits, kinematic chains, and current body pose
- **Flexible interaction**: Supports various objects and tasks without task-specific programming
- **Multi-modal input**: Combines speech, vision, and motion tracking
- **Checkpoint-based execution**: Monitors progress and adjusts instructions as needed

## System Architecture

The system consists of several interconnected modules:

1. **User Interaction Module** (`speech_engine.py`)
   - Handles speech recognition and text-to-speech
   - Manages user commands and system feedback

2. **Context-aware Tutorial Generation** (`llm_gesture_clean.py`)
   - Uses multimodal AI to understand objects and situations
   - Generates step-by-step movement descriptions

3. **Movement Instruction Generation** (`llm_gesture_clean.py`)
   - Converts tutorial descriptions into specific movement instructions
   - Considers current body pose and user preferences

4. **Stimulation Instruction Generation** (`gesture_processor.py`)
   - Maps movement instructions to EMS actuations
   - Uses knowledge base of feasible EMS movements

5. **Biomechanical Constraints** (`gesture_processor.py`)
   - Validates movements against joint limits
   - Implements kinematic chain adjustments
   - Ensures safe stimulation parameters

6. **Motion Tracking** (`mocap_oscserver.py`)
   - Receives and processes skeletal data via OSC
   - Tracks joint angles and limb positions

## Requirements

See `requirements.txt` for a complete list of dependencies. Key requirements include:
- Python 3.x
- OpenAI API access
- Various computer vision and audio processing libraries
- OSC communication libraries

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/generative-muscle-stimulation.git
cd generative-muscle-stimulation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key in `llm_gesture_clean.py`

4. Configure your EMS device and motion tracking system

## Usage

### Basic Operation

1. Start the motion tracking server:
```bash
python mocap_oscserver.py
```

2. Run the main system:
```bash
python llm_gesture_clean.py [example_name] [user_request] [handedness] [hands_obj] [skeleton_direction]
```


### Wake Word and Commands

- Use "EMS" as the wake word to activate the system
- Commands:
  - "EMS, help me [task]" - Request assistance
  - "EMS, continue" - Confirm and execute movements
  - "EMS, stop" - Stop current execution
  - "EMS, user-settings: [preference]" - Set user preferences

## Configuration Files

### Gesture Lists
- `gesture_lists/llm-gesture-list.csv` - Maps LLM gestures to EMS gestures
- `gesture_lists/ems-gesture-list.csv` - EMS stimulation parameters
- `gesture_lists/ems-joint-limits-clean.csv` - Biomechanical joint limits

### User Profiles
- `user_profile/user_profile_1.txt` - User preferences and customization

### Calibration
- `ems_calibration.txt` - EMS channel calibration values
- `poses/I-pose.json` - Neutral pose calibration

## Key Components

### gesture_processor.py
Handles the core EMS processing:
- Parses movement instructions
- Validates against biomechanical constraints
- Generates EMS stimulation patterns
- Implements kinematic chain logic

### llm_gesture_clean.py
Manages the AI pipeline:
- Object recognition from images
- Movement planning using LLMs
- Gesture translation
- Checkpoint management

### mocap_oscserver.py
Provides real-time skeleton tracking:
- Receives OSC data from motion capture
- Calculates joint angles
- Tracks limb positions and directions

### speech_engine.py
Handles audio interaction:
- Speech recognition with wake word detection
- Text-to-speech feedback
- User command parsing

## Safety Considerations

The system implements multiple safety layers:
1. **Biomechanical validation** - Prevents movements beyond joint limits
2. **User confirmation** - Requires explicit consent before stimulation
3. **Emergency stop** - Voice commands or opposing movements stop execution
4. **Calibrated intensities** - Per-user, per-muscle calibration
5. **Progressive execution** - Monitors movement success and adjusts accordingly

## Limitations

- **Latency**: LLM API calls introduce ~10 second delays
- **EMS constraints**: Limited to muscles accessible via surface electrodes
- **Motion tracking**: Requires external tracking system
- **Context understanding**: May misinterpret complex scenes or requests

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{ho2025generative,
  title={Generative Muscle Stimulation: Physical Assistance by Constraining Multimodal-AI with Biomechanical Knowledge},
  author={Ho, Yun and Nith, Romain and Jiang, Peili and Teng, Shan-Yuan and Lopes, Pedro},
  booktitle={Proceedings of the CHI Conference on Human Factors in Computing Systems},
  year={2025}
}
```

## License


## Contact

For questions or collaborations, please contact the authors at the University of Chicago.