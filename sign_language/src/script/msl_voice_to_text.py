#!/usr/bin/env python3
"""
MSL Voice-to-Text Node - ROS Noetic
Listens to the microphone, transcribes speech to text using SpeechRecognition
(Google Web Speech API by default, with a Vosk offline fallback),
filters result to only publish recognised MSL words,
and publishes to /msl/voice_text so msl_detector.py shows it on the GUI.

Publishes:
  /msl/voice_text   (std_msgs/String) — matched MSL word (upper-case)
                      ONLY published when transcript contains a valid MSL word.
  /msl/voice_event  (std_msgs/String) — JSON: {"text","matched","source","timestamp"}

Parameters:
  ~energy_threshold   (int,   default 300)
  ~pause_threshold    (float, default 0.8)
  ~language           (str,   default "ms-MY")
  ~use_offline        (bool,  default false)
  ~vosk_model_path    (str,   default "")
  ~phrase_timeout     (float, default 3.0)

Dependencies:
    pip install SpeechRecognition pyaudio
"""

import json
import time

import rospy
from std_msgs.msg import String

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

# ── MSL word filter ────────────────────────────────────────────────────────────
MSL_WORDS = {'awak', 'maaf', 'makan', 'minum', 'salah', 'saya', 'tolong'}

def snap_to_msl(text: str):
    """Return first MSL word found in transcript (upper-case), or None."""
    for token in text.lower().split():
        clean = token.strip(".,!?-")
        if clean in MSL_WORDS:
            return clean.upper()
    return None


# ── MSL Voice-to-Text Node ────────────────────────────────────────────────────

class MSLVoiceToTextNode:

    def __init__(self):
        rospy.init_node('msl_voice_to_text', anonymous=False)
        rospy.loginfo("[msl_vtt] Initialising MSL Voice-to-Text node...")

        if not SR_AVAILABLE:
            rospy.logfatal("[msl_vtt] pip install SpeechRecognition pyaudio")
            rospy.signal_shutdown("Missing dependency: SpeechRecognition")
            return

        self._energy_threshold = rospy.get_param('~energy_threshold', 300)
        self._pause_threshold  = rospy.get_param('~pause_threshold',  0.8)
        self._language         = rospy.get_param('~language',         'ms-MY')
        self._use_offline      = rospy.get_param('~use_offline',      False)
        self._vosk_model_path  = rospy.get_param('~vosk_model_path',  '')
        self._phrase_timeout   = rospy.get_param('~phrase_timeout',   3.0)

        self._pub_text  = rospy.Publisher('/msl/voice_text',  String, queue_size=5)
        self._pub_event = rospy.Publisher('/msl/voice_event', String, queue_size=5)

        self._recogniser = sr.Recognizer()
        self._recogniser.energy_threshold         = self._energy_threshold
        self._recogniser.pause_threshold          = self._pause_threshold
        self._recogniser.dynamic_energy_threshold = True

        self._vosk_model = None
        if self._use_offline:
            self._vosk_model = self._load_vosk_model()

        try:
            self._mic = sr.Microphone()
        except OSError as e:
            rospy.logfatal(f"[msl_vtt] Microphone not accessible: {e}")
            rospy.signal_shutdown("No microphone")
            return

        rospy.loginfo(f"[msl_vtt] language     : {self._language}")
        rospy.loginfo(f"[msl_vtt] MSL filter   : {sorted(MSL_WORDS)}")
        rospy.loginfo("[msl_vtt] Adjusting for ambient noise (1 s)...")

        with self._mic as source:
            self._recogniser.adjust_for_ambient_noise(source, duration=1)

        rospy.loginfo("[msl_vtt] Listening for MSL words...")

        self._stop_listening = self._recogniser.listen_in_background(
            self._mic,
            self._audio_callback,
            phrase_time_limit=self._phrase_timeout,
        )

    def _load_vosk_model(self):
        try:
            from vosk import Model
        except ImportError:
            rospy.logwarn("[msl_vtt] vosk not installed; using Google.")
            return None
        if not self._vosk_model_path:
            return None
        try:
            return Model(self._vosk_model_path)
        except Exception as e:
            rospy.logwarn(f"[msl_vtt] Vosk load failed: {e}")
            return None

    def _audio_callback(self, recogniser, audio):
        text = source = None

        if self._use_offline and self._vosk_model:
            text, source = self._recognise_vosk(recogniser, audio)
        if text is None:
            text, source = self._recognise_google(recogniser, audio)
        if text is None:
            text, source = self._recognise_sphinx(recogniser, audio)

        if not text:
            return

        text = text.strip()
        rospy.loginfo(f"[msl_vtt] Heard [{source}]: '{text}'")

        matched = snap_to_msl(text)
        if matched:
            rospy.loginfo(f"[msl_vtt] MSL match -> '{matched}'")
            self._pub_text.publish(String(data=matched))
            self._pub_event.publish(String(data=json.dumps({
                "text": text.lower(), "matched": matched,
                "source": source, "timestamp": time.time(),
            })))
        else:
            rospy.loginfo(f"[msl_vtt] No MSL word — ignored.")

    def _recognise_google(self, rec, audio):
        try:
            return rec.recognize_google(audio, language=self._language), "google"
        except sr.UnknownValueError:
            return None, None
        except sr.RequestError as e:
            rospy.logwarn(f"[msl_vtt] Google error: {e}")
            return None, None

    def _recognise_vosk(self, rec, audio):
        try:
            result = json.loads(rec.recognize_vosk(audio, model=self._vosk_model))
            phrase = result.get("text", "").strip()
            return (phrase, "vosk") if phrase else (None, None)
        except Exception as e:
            rospy.logwarn(f"[msl_vtt] Vosk error: {e}")
            return None, None

    def _recognise_sphinx(self, rec, audio):
        try:
            return rec.recognize_sphinx(audio), "sphinx"
        except Exception:
            return None, None

    def run(self):
        rospy.on_shutdown(self._shutdown)
        rospy.spin()

    def _shutdown(self):
        rospy.loginfo("[msl_vtt] Shutting down microphone listener.")
        if hasattr(self, '_stop_listening') and callable(self._stop_listening):
            self._stop_listening(wait_for_stop=False)


if __name__ == '__main__':
    try:
        node = MSLVoiceToTextNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
