import cv2
import pyautogui
import numpy as np
import mediapipe as mp
import screeninfo  # For getting screen size

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize OpenCV webcam capture
cap = cv2.VideoCapture(0)

# Get screen dimensions
screen = screeninfo.get_monitors()[0]
SCREEN_WIDTH, SCREEN_HEIGHT = screen.width, screen.height

def move_cursor(hand_landmarks, frame_width, frame_height):
    index_finger_tip = hand_landmarks.landmark[8]  # Index finger tip
    x = int(index_finger_tip.x * frame_width)
    y = int(index_finger_tip.y * frame_height)
    
    # Convert to screen coordinates
    screen_x = np.interp(x, (0, frame_width), (0, SCREEN_WIDTH))
    screen_y = np.interp(y, (0, frame_height), (0, SCREEN_HEIGHT))

    # Move cursor
    pyautogui.moveTo(screen_x, screen_y, duration=0.1)  # Smooth movement

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if the camera is not returning frames

    frame = cv2.flip(frame, 1)  # Flip horizontally for mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            move_cursor(hand_landmarks, frame.shape[1], frame.shape[0])

    cv2.imshow("Virtual Mouse", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
