import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, max_hands=1, detection_conf=0.7, track_conf=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=max_hands, min_detection_confidence=detection_conf, min_tracking_confidence=track_conf)
        self.mp_draw = mp.solutions.drawing_utils
    
    def find_hands(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return result
    
    def get_landmarks(self, result, frame_shape):
        h, w, _ = frame_shape
        landmarks = []
        if result.multi_hand_landmarks:
            for hand_lms in result.multi_hand_landmarks:
                for lm in hand_lms.landmark:
                    landmarks.append((int(lm.x * w), int(lm.y * h)))
        return landmarks