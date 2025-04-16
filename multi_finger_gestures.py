import cv2
import numpy as np
import pyautogui
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)  # Open webcam

def detect_zoom(hand_landmarks):
    index_tip = hand_landmarks.landmark[8]
    thumb_tip = hand_landmarks.landmark[4]

    distance = np.linalg.norm(np.array([index_tip.x, index_tip.y]) - np.array([thumb_tip.x, thumb_tip.y]))
    
    if distance > 0.07:  
        pyautogui.hotkey('ctrl', '+')  # Zoom In
    elif distance < 0.03:  
        pyautogui.hotkey('ctrl', '-')  # Zoom Out

def detect_window_switch(hand_landmarks):
    pinky_tip = hand_landmarks.landmark[20]
    if pinky_tip.y < 0.2:  # Pinky raised high
        pyautogui.hotkey('alt', 'tab')  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            detect_zoom(hand_landmarks)
            detect_window_switch(hand_landmarks)

    cv2.imshow("Virtual Mouse", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
