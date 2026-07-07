#!/usr/bin/env python3
"""
MSL Detector - ROS Noetic Node
Subscribes to a camera topic, detects hand landmarks via MediaPipe,
classifies Malaysian Sign Language using the trained .h5 model,
and publishes predictions as ROS topics.

Audio is handled by the separate msl_sound.py node which subscribes
to /msl/prediction.

Usage:
    roslaunch msl_project msl_detector.launch
    or
    rosrun msl_project msl_detector.py
"""

import os
import collections
import urllib.request
import time
from pathlib import Path

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import tensorflow as tf

# ── Configuration ──────────────────────────────────────────────────────────────
CLASSES      = ['AWAK', 'MAAF', 'MAKAN', 'MINUM', 'SALAH', 'SAYA', 'TOLONG']
CONF_THRESH  = 0.25
SMOOTH_N     = 8
SEQUENCE_LEN = 10

SCRIPT_DIR      = Path(__file__).parent
MODELS_DIR      = SCRIPT_DIR / "models"
HAND_MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
                   "hand_landmarker/hand_landmarker/float16/latest/"
                   "hand_landmarker.task")
HAND_MODEL_PATH = str(MODELS_DIR / "hand_landmarker.task")
MSL_MODEL_PATH  = str(MODELS_DIR / "msl_3dcnn.h5")

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

COL_GREEN  = (50,  220,  50)
COL_BLUE   = (220, 150,  50)
COL_WHITE  = (255, 255, 255)
COL_BLACK  = (0,     0,   0)
COL_YELLOW = (0,   220, 220)
COL_RED    = (50,   50, 220)
COL_CYAN   = (220, 220,  50)


# ── Helpers ────────────────────────────────────────────────────────────────────

def ensure_hand_model():
    if not os.path.exists(HAND_MODEL_PATH):
        rospy.loginfo("[msl_detector] Downloading hand_landmarker.task (~25 MB)...")
        os.makedirs(os.path.dirname(HAND_MODEL_PATH), exist_ok=True)
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        rospy.loginfo(f"[msl_detector] Saved: {HAND_MODEL_PATH}")
    else:
        rospy.loginfo(f"[msl_detector] Hand model found: {HAND_MODEL_PATH}")


def landmarks_to_features(landmarks) -> np.ndarray:
    return np.array([[lm.x, lm.y] for lm in landmarks],
                    dtype='float32').flatten()


def fill_rect(img, p1, p2, color, alpha=0.6):
    x1, y1 = p1; x2, y2 = p2
    x1, x2 = max(0, x1), min(img.shape[1]-1, x2)
    y1, y2 = max(0, y1), min(img.shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)


def shadow_text(img, text, org, scale, color, thick=2):
    f = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(img, text, (org[0]+2, org[1]+2), f, scale, COL_BLACK, thick+1, cv2.LINE_AA)
    cv2.putText(img, text, org, f, scale, color, thick, cv2.LINE_AA)


def prob_bar(img, x, y, w, p, color):
    fill_rect(img, (x, y), (x+w, y+14), (50, 50, 50), 0.8)
    bw = max(0, int(w * p))
    if bw > 0:
        fill_rect(img, (x, y), (x+bw, y+14), color, 0.9)


# ── MSL Detector Node ──────────────────────────────────────────────────────────

class MSLDetectorNode:

    def __init__(self):
        rospy.init_node('msl_detector', anonymous=False)
        rospy.loginfo("[msl_detector] Initialising...")

        camera_topic     = rospy.get_param('~camera_topic', '/camera/image_raw')
        self.show_window = rospy.get_param('~show_window',  True)

        # Publishers
        self.pub_label = rospy.Publisher('/msl/prediction', String,  queue_size=1)
        self.pub_conf  = rospy.Publisher('/msl/confidence', Float32, queue_size=1)
        self.pub_viz   = rospy.Publisher('/msl/image',      Image,   queue_size=1)

        self.bridge = CvBridge()

        ensure_hand_model()
        self._load_msl_model()
        self._load_hand_detector()

        # State
        self.landmark_buf = collections.deque(maxlen=SEQUENCE_LEN)
        self.pred_buf     = collections.deque(maxlen=SMOOTH_N)
        self.pred         = "—"
        self.conf         = 0.0
        self.probs        = np.zeros(len(CLASSES))

        self.sub = rospy.Subscriber(camera_topic, Image,
                                    self.image_callback, queue_size=1,
                                    buff_size=2**24)

        # Voice word display state
        self._voice_word      = ""
        self._voice_expire_at = 0.0
        rospy.Subscriber('/msl/voice_text', String,
                         self._voice_callback, queue_size=5)

        rospy.loginfo(f"[msl_detector] Ready — listening on: {camera_topic}")

    # ── Voice callback ────────────────────────────────────────────────────────

    def _voice_callback(self, msg: String):
        word = msg.data.strip().upper()
        if word:
            self._voice_word      = word
            self._voice_expire_at = time.monotonic() + 4.0

    # ── Model loaders ─────────────────────────────────────────────────────────

    def _load_msl_model(self):
        if not os.path.exists(MSL_MODEL_PATH):
            rospy.logfatal(f"[msl_detector] Model not found: {MSL_MODEL_PATH}")
            rospy.logfatal("[msl_detector] Run train_msl.py then convert_to_h5.py first!")
            rospy.signal_shutdown("Model file missing")
            return
        rospy.loginfo(f"[msl_detector] Loading .h5 model: {MSL_MODEL_PATH}")
        self.model = tf.keras.models.load_model(MSL_MODEL_PATH)
        rospy.loginfo("[msl_detector] MSL model loaded.")

    def _load_hand_detector(self):
        base_opts = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector = mp_vision.HandLandmarker.create_from_options(opts)
        rospy.loginfo("[msl_detector] HandLandmarker ready.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, seq: np.ndarray) -> np.ndarray:
        inp = seq.reshape(1, SEQUENCE_LEN, 21, 2, 1)
        return self.model.predict(inp, verbose=0)[0]

    # ── Camera callback ───────────────────────────────────────────────────────

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"[msl_detector] cv_bridge error: {e}")
            return

        h, w = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        has_hand = bool(result.hand_landmarks)

        if has_hand:
            lms = result.hand_landmarks[0]
            self._draw_hand(frame, lms, h, w)
            self.landmark_buf.append(landmarks_to_features(lms))

            if len(self.landmark_buf) == SEQUENCE_LEN:
                seq        = np.array(list(self.landmark_buf), dtype='float32')
                p_arr      = self.predict(seq)
                idx        = int(np.argmax(p_arr))
                self.conf  = float(p_arr[idx])
                self.probs = p_arr
                self.pred_buf.append(idx)
                smooth     = collections.Counter(self.pred_buf).most_common(1)[0][0]
                self.pred  = CLASSES[smooth] if self.conf >= CONF_THRESH else "Low confidence"

            self.pub_label.publish(String(data=self.pred))
            self.pub_conf.publish(Float32(data=self.conf))
        else:
            self.landmark_buf.clear()
            self.pred_buf.clear()
            self.pred  = "No hand"
            self.conf  = 0.0
            self.probs = np.zeros(len(CLASSES))

        self._draw_overlay(frame, h, w, has_hand)

        try:
            self.pub_viz.publish(
                self.bridge.cv2_to_imgmsg(frame, encoding='bgr8'))
        except Exception as e:
            rospy.logerr(f"[msl_detector] publish image error: {e}")

        if self.show_window:
            cv2.namedWindow("MSL Detector", cv2.WINDOW_NORMAL)
            cv2.imshow("MSL Detector", frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                rospy.signal_shutdown("User quit")

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_hand(self, frame, landmarks, h, w):
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2, cv2.LINE_AA)
        for i, pt in enumerate(pts):
            cv2.circle(frame, pt, 5,
                       (0, 255, 180) if i == 0 else COL_WHITE, -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 5, COL_BLACK, 1, cv2.LINE_AA)

    def _draw_overlay(self, frame, h, w, has_hand):
        fill_rect(frame, (0, 0), (w, 52), (20, 20, 20), 0.7)
        shadow_text(frame, "MSL Detector", (12, 36), 0.9, COL_WHITE)

        # ── Top-right: voice word ─────────────────────────────────────────────
        now = time.monotonic()
        if self._voice_word and now < self._voice_expire_at:
            label  = f"MIC: {self._voice_word}"
            vscale = 0.85
            vthick = 2
            font   = cv2.FONT_HERSHEY_DUPLEX
            (tw, th), baseline = cv2.getTextSize(label, font, vscale, vthick)
            pad    = 10
            box_x1 = w - tw - pad * 2 - 6
            box_y1 = 8
            box_x2 = w - 6
            box_y2 = box_y1 + th + baseline + pad * 2
            time_left = self._voice_expire_at - now
            alpha     = min(1.0, time_left) * 0.75
            fill_rect(frame, (box_x1, box_y1), (box_x2, box_y2), (20, 20, 20), alpha)
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), COL_CYAN, 1, cv2.LINE_AA)
            shadow_text(frame, label, (box_x1 + pad, box_y1 + pad + th), vscale, COL_CYAN, thick=vthick)

        px, py, pw = 12, h - 235, 310
        fill_rect(frame, (px-10, py-14), (px+pw+10, h-8), (15, 15, 15), 0.65)

        pc = COL_GREEN if (has_hand and self.conf >= CONF_THRESH) else COL_YELLOW
        shadow_text(frame, self.pred, (px, py+46), 1.45, pc, thick=3)

        if has_hand:
            cv2.putText(frame, f"Confidence: {self.conf*100:.1f}%",
                        (px, py+74), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, COL_WHITE, 1, cv2.LINE_AA)

        by = py + 92
        for i, cls in enumerate(CLASSES):
            p_val = float(self.probs[i])
            top   = (i == int(np.argmax(self.probs))) and has_hand
            prob_bar(frame, px, by, pw, p_val, COL_GREEN if top else COL_BLUE)
            cv2.putText(frame, f"{cls:<7}  {p_val*100:5.1f}%",
                        (px+5, by+11), cv2.FONT_HERSHEY_SIMPLEX,
                        0.43, COL_GREEN if top else COL_WHITE, 1, cv2.LINE_AA)
            by += 19

        bt = "HAND DETECTED" if has_hand else "SHOW YOUR HAND"
        bc = COL_GREEN        if has_hand else COL_RED
        (bw2, _), _ = cv2.getTextSize(bt, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
        fill_rect(frame, (w-bw2-26, h-38), (w-4, h-4), COL_BLACK, 0.7)
        shadow_text(frame, bt, (w-bw2-18, h-12), 0.65, bc)

    def run(self):
        rospy.spin()
        cv2.destroyAllWindows()


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        node = MSLDetectorNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
