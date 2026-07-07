# Robot sign_language

A ROS (Noetic) package for **Malaysian Sign Language (MSL)** recognition. It reads a
USB camera stream, detects hand landmarks with MediaPipe, classifies the gesture with
a trained 3D-CNN, speaks the recognised word aloud, and can also transcribe spoken
Malay words back to on-screen text — a two-way sign ↔ speech bridge.

Recognised vocabulary (7 words):

| Word     | Meaning (EN) |
|----------|--------------|
| `AWAK`   | you          |
| `MAAF`   | sorry        |
| `MAKAN`  | eat          |
| `MINUM`  | drink        |
| `SALAH`  | wrong        |
| `SAYA`   | I / me       |
| `TOLONG` | help         |

## How it works

```
 USB camera ──► /camera/image_raw
                     │
                     ▼
          ┌──────────────────────┐
          │  msl_detector.py     │  MediaPipe hand landmarks → 3D-CNN
          │                      │  (10-frame sequence, smoothed)
          └──────────────────────┘
                     │ /msl/prediction (String)
                     │ /msl/confidence (Float32)
                     │ /msl/image      (Image, annotated GUI)
                     ▼
          ┌──────────────────────┐
          │  msl_sound.py        │  plays sounds/<word>.wav
          │                      │  (falls back to festival TTS)
          └──────────────────────┘
                     │ /msl/spoken_word (String)

          ┌──────────────────────┐
          │  msl_voice_to_text.py│  microphone → speech recognition
          │                      │  → matched MSL word
          └──────────────────────┘
                     │ /msl/voice_text (String) ──► shown on detector GUI
```

## ROS nodes & topics

### `msl_detector.py`
Subscribes to a camera image topic, runs MediaPipe hand detection + the trained MSL
model, and renders an annotated preview window with per-class probability bars.

- **Subscribes:** `/camera/image_raw` (`sensor_msgs/Image`), `/msl/voice_text` (`std_msgs/String`)
- **Publishes:** `/msl/prediction` (`String`), `/msl/confidence` (`Float32`), `/msl/image` (`Image`)
- **Params:** `~camera_topic` (default `/camera/image_raw`), `~show_window` (default `true`)

### `msl_sound.py`
Speaks each recognised word. Plays a matching `.wav`/`.ogg` from `src/sounds/`, or falls
back to `festival` text-to-speech. Requires `soundplay_node` running.

- **Subscribes:** `/msl/prediction` (`String`)
- **Publishes:** `/msl/spoken_word` (`String`)

### `msl_voice_to_text.py`
Listens to the microphone, transcribes speech (Google Web Speech API, with optional
Vosk/Sphinx offline fallback), and publishes only recognised MSL words.

- **Publishes:** `/msl/voice_text` (`String`), `/msl/voice_event` (`String`, JSON)
- **Params:** `~language` (default `ms-MY`), `~energy_threshold` (`300`), `~pause_threshold` (`0.8`), `~phrase_timeout` (`3.0`), `~use_offline` (`false`), `~vosk_model_path` (`""`)

## Requirements

- **ROS Noetic** on Ubuntu 20.04 (catkin workspace)
- Python 3 packages:
  ```bash
  pip install tensorflow mediapipe opencv-python numpy SpeechRecognition pyaudio
  # optional offline speech recognition:
  pip install vosk pocketsphinx
  ```
- ROS packages:
  ```bash
  sudo apt install ros-noetic-usb-cam ros-noetic-sound-play ros-noetic-cv-bridge
  ```
- The trained model **`src/script/models/msl_3dcnn.h5`** must be present.
  The hand-landmark model (`hand_landmarker.task`) is downloaded automatically on first run.

## Installation

Clone into the `src/` folder of your catkin workspace and build:

```bash
cd ~/catkin_ws/src
git clone <repo-url> sign_language
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

Make the node scripts executable:

```bash
chmod +x src/sign_language/src/script/*.py
```

## Usage

Launch everything (camera + detector + sound + voice) at once:

```bash
roslaunch sign_language sign_language.launch
```

Or run pieces individually:

```bash
# camera only
roslaunch sign_language camera.launch

# detector node
rosrun sign_language msl_detector.py

# audio output node (needs soundplay_node running)
rosrun sign_language msl_sound.py
```

Show a hand gesture to the camera; the recognised word appears in the preview window
and is spoken aloud. Speak one of the MSL words into the mic to have it shown on screen.

Press `q` or `Esc` in the preview window to quit.

## Custom sounds

Drop audio files in `src/sounds/` named after each word (case-insensitive), e.g.
`makan.wav`, `minum.wav`. Supported extensions: `.wav`, `.ogg`. Words without a file
fall back to `festival` TTS.

## Project layout

```
sign_language/
├── package.xml                    # catkin package manifest
├── CMakeLists.txt
└── src/
    ├── launch/
    │   ├── sign_language.launch    # full system
    │   └── camera.launch           # usb_cam only
    ├── script/
    │   ├── msl_detector.py         # hand-gesture → text
    │   ├── msl_sound.py            # text → speech
    │   ├── msl_voice_to_text.py    # speech → text
    │   └── models/
    │       ├── msl_3dcnn.h5        # trained classifier
    │       └── hand_landmarker.task
    └── sounds/                      # per-word audio clips
        ├── awak.wav  maaf.wav  makan.wav  minum.wav
        └── salah.wav saya.wav  tolong.wav
```

## Notes

- Confidence threshold for accepting a prediction is `0.25`; predictions are smoothed
  over the last 8 frames and a gesture uses a 10-frame sequence.
- `camera.launch` expects a USB camera at `/dev/video0` at 1280×720 @ 30 fps; adjust
  the params for your device.
