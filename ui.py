import tkinter as tk
from tkinter import messagebox
import json
import joblib
import numpy as np
import cv2
import mediapipe as mp
from sklearn.neighbors import KNeighborsClassifier

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

gesture_data = {}

# Open webcam
cap = cv2.VideoCapture(0)

def detect_hands(frame):
    """Detect hands using MediaPipe"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return hands.process(rgb_frame)

def record_gesture():
    global gesture_data
    label = gesture_entry.get()
    if not label:
        messagebox.showerror("Error", "Enter a gesture name!")
        return

    messagebox.showinfo("Info", "Show your hand gesture for 5 seconds...")
    count = 0
    collected_data = []

    while count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        results = detect_hands(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                collected_data.append(landmarks)

        cv2.imshow("Live Gesture Preview", frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyWindow("Live Gesture Preview")
    
    if collected_data:
        gesture_data[label] = collected_data
        with open("gesture_data.json", "w") as f:
            json.dump(gesture_data, f)
        messagebox.showinfo("Success", f"Gesture '{label}' recorded!")
    else:
        messagebox.showerror("Error", "No gesture detected!")

def train_model():
    try:
        with open("gesture_data.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        messagebox.showerror("Error", "No recorded gestures found!")
        return

    X, y = [], []
    for label, samples in data.items():
        for landmarks in samples:
            X.append(np.array(landmarks).flatten())
            y.append(label)

    if not X:
        messagebox.showerror("Error", "No valid gesture data found!")
        return

    X = np.array(X)
    y = np.array(y)

    sensitivity = sensitivity_slider.get()
    knn = KNeighborsClassifier(n_neighbors=sensitivity)
    knn.fit(X, y)

    joblib.dump(knn, "gesture_model.pkl")
    messagebox.showinfo("Success", "Model trained!")

# GUI
root = tk.Tk()
root.title("Gesture-Based Virtual Mouse")

gesture_entry = tk.Entry(root, width=30)
gesture_entry.pack(pady=10)
gesture_entry.insert(0, "Enter gesture name")

record_button = tk.Button(root, text="Record Gesture", command=record_gesture)
record_button.pack(pady=10)

sensitivity_label = tk.Label(root, text="Gesture Sensitivity")
sensitivity_label.pack(pady=5)

sensitivity_slider = tk.Scale(root, from_=1, to=10, orient="horizontal")
sensitivity_slider.set(3)
sensitivity_slider.pack(pady=5)

train_button = tk.Button(root, text="Train Model", command=train_model)
train_button.pack(pady=10)

root.mainloop()

cap.release()
cv2.destroyAllWindows()
