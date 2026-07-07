#!/usr/bin/env python3
"""
MSL Sound Node - ROS Noetic
Subscribes to /msl/prediction (String) published by msl_detector.py.
Speaks each recognised Malaysian Sign Language word aloud via the
ROS sound_play package (soundplay_node must be running).

Custom audio files go in:
    <package>/sounds/          e.g.  msl_project/sounds/makan.wav

If a .wav file exists for the predicted word it is played directly.
If not, the node falls back to festival TTS.

Publishes:
  /msl/spoken_word  (std_msgs/String)  — word at the moment it is spoken

Dependencies:
    sudo apt install ros-noetic-sound-play

Usage:
    roslaunch msl_project msl_detector.launch
"""

import os
import time
from pathlib import Path

import rospy
from std_msgs.msg import String
from sound_play.libsoundplay import SoundClient

# ── Configuration ──────────────────────────────────────────────────────────────

CLASSES = ['AWAK', 'MAAF', 'MAKAN', 'MINUM', 'SALAH', 'SAYA', 'TOLONG']

# ── Folder that holds custom .wav / .ogg files ────────────────────────────────
# Structure expected:
#   sounds/
#     awak.wav
#     maaf.wav
#     makan.wav
#     minum.wav
#     salah.wav
#     saya.wav
#     tolong.wav
#
# Change SOUNDS_DIR to point anywhere you like, e.g.:
#   SOUNDS_DIR = Path('/home/mustar/my_sounds')
SOUNDS_DIR = Path(__file__).parent.parent / "sounds"   # <package>/sounds/

# Supported audio extensions, tried in order.
SOUND_EXTENSIONS = ['.wav', '.ogg']

# ── TTS fallback (used when no sound file is found) ───────────────────────────
TTS_VOICE = 'voice_kal_diphone'   # run: festival --list-voices

WORD_TO_SPOKEN = {
    'AWAK':   'awak',
    'MAAF':   'maaf',
    'MAKAN':  'makan',
    'MINUM':  'minum',
    'SALAH':  'salah',
    'SAYA':   'saya',
    'TOLONG': 'tolong',
}

# ── Trigger tuning ────────────────────────────────────────────────────────────
AUDIO_HOLD_FRAMES  = 12    # consecutive frames before speaking
AUDIO_COOLDOWN_SEC = 2.0   # minimum gap between words


# ── MSL Sound Node ─────────────────────────────────────────────────────────────

class MSLSoundNode:

    def __init__(self):
        rospy.init_node('msl_sound', anonymous=False)
        rospy.loginfo("[msl_sound] Initialising...")

        # sound_play client — blocking=False keeps the callback non-blocking
        self.sound_client = SoundClient(blocking=False)
        rospy.sleep(0.5)
        rospy.loginfo("[msl_sound] SoundClient ready.")

        # Build sound-file map from the sounds/ folder
        self.sound_map = self._build_sound_map()

        # Audio state
        self._hold_count = 0
        self._last_word  = None
        self._last_time  = 0.0

        # Publisher
        self.pub_spoken = rospy.Publisher('/msl/spoken_word', String, queue_size=1)

        # Subscriber
        rospy.Subscriber('/msl/prediction', String,
                         self.prediction_callback, queue_size=1)

        rospy.loginfo(f"[msl_sound] sounds folder: {SOUNDS_DIR}")
        rospy.loginfo(f"[msl_sound] Loaded sound files: {list(self.sound_map.keys())}")
        rospy.loginfo("[msl_sound] Listening on /msl/prediction")

    # ── Sound-file discovery ──────────────────────────────────────────────────

    def _build_sound_map(self) -> dict:
        """
        Scan SOUNDS_DIR and map each MSL class label to its audio file path.
        File must be named exactly like the label (case-insensitive),
        e.g.  makan.wav  or  MAKAN.wav  both match 'MAKAN'.
        Returns a dict: { 'MAKAN': '/full/path/makan.wav', ... }
        """
        sound_map = {}

        if not SOUNDS_DIR.exists():
            rospy.logwarn(f"[msl_sound] sounds folder not found: {SOUNDS_DIR}")
            rospy.logwarn("[msl_sound] Falling back to TTS for all words.")
            return sound_map

        for cls in CLASSES:
            for ext in SOUND_EXTENSIONS:
                # Try exact case, then lower-case filename
                for name in [cls + ext, cls.lower() + ext]:
                    candidate = SOUNDS_DIR / name
                    if candidate.exists():
                        sound_map[cls] = str(candidate)
                        rospy.loginfo(f"[msl_sound]   {cls} -> {candidate.name}")
                        break
                if cls in sound_map:
                    break

            if cls not in sound_map:
                rospy.loginfo(f"[msl_sound]   {cls} -> (no file, will use TTS)")

        return sound_map

    # ── Callback ──────────────────────────────────────────────────────────────

    def prediction_callback(self, msg: String):
        word = msg.data

        if word not in CLASSES:
            self._hold_count = 0
            return

        if word == self._last_word:
            self._hold_count += 1
        else:
            self._hold_count = 1
            self._last_word  = word

        now         = time.monotonic()
        hold_ok     = self._hold_count >= AUDIO_HOLD_FRAMES
        cooldown_ok = (now - self._last_time) >= AUDIO_COOLDOWN_SEC

        if hold_ok and cooldown_ok:
            self._play(word)
            self.pub_spoken.publish(String(data=word))
            self._last_time  = now
            self._hold_count = 0

    # ── Playback ──────────────────────────────────────────────────────────────

    def _play(self, word: str):
        """Play custom file if available, otherwise fall back to TTS."""
        if word in self.sound_map:
            path = self.sound_map[word]
            rospy.loginfo(f"[msl_sound] Playing file: {Path(path).name}")
            try:
                self.sound_client.playWave(path)
            except Exception as e:
                rospy.logwarn(f"[msl_sound] playWave error: {e}")
        else:
            text = WORD_TO_SPOKEN.get(word, word.lower())
            rospy.loginfo(f"[msl_sound] TTS fallback: '{text}'")
            try:
                self.sound_client.say(text, TTS_VOICE)
            except Exception as e:
                rospy.logwarn(f"[msl_sound] TTS error: {e}")

    # ── Spin ──────────────────────────────────────────────────────────────────

    def run(self):
        rospy.spin()


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        node = MSLSoundNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
