import cv2
import mediapipe as mp
import pyautogui
import joblib
import numpy as np

# Load trained model
model = joblib.load("model/gesture_model.pkl")
print("Model loaded successfully!")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    result = hands.process(rgb_frame)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract landmark positions
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            # Convert to NumPy array and predict gesture
            landmarks = np.array(landmarks).reshape(1, -1)
            predicted_gesture = model.predict(landmarks)[0]
            print(f"Predicted Gesture: {predicted_gesture}")

            # Perform actions based on gesture
            if predicted_gesture == "move_cursor":
                x, y = int(landmarks[0][0] * 1920), int(landmarks[0][1] * 1080)
                pyautogui.moveTo(x, y)
            elif predicted_gesture == "click":
                pyautogui.click()
            elif predicted_gesture == "scroll":
                pyautogui.scroll(5)
            elif predicted_gesture == "zoom":
                pyautogui.hotkey("ctrl", "+")
            elif predicted_gesture == "window_switch":
                pyautogui.hotkey("alt", "tab")

            # Draw landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the output
    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
