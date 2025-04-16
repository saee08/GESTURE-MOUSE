import cv2
from hand_tracking import HandTracker
from cursor_control import CursorControl
from click_scroll import ClickScroll
from multi_finger_gestures import MultiFingerGestures
from gesture_training import GestureTraining

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    tracker = HandTracker()
    cursor = CursorControl()
    click_scroll = ClickScroll()
    gestures = MultiFingerGestures()
    training = GestureTraining()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Failed to capture frame.")
                break

            result = tracker.find_hands(frame)
            landmarks = tracker.get_landmarks(result, frame.shape)

            # Debugging: Print landmarks to check if they are detected
            if landmarks:
                print("✅ Landmarks detected:", landmarks)
            else:
                print("⚠️ No landmarks detected")

            if landmarks and len(landmarks) > 8:  # Ensure index 8 exists
                cursor.move_cursor(*landmarks[8], frame.shape)

                if click_scroll.detect_click(landmarks):
                    print("🖱️ Click detected")

                if click_scroll.detect_scroll(landmarks):
                    print("📜 Scroll detected")

                if gestures.detect_zoom(landmarks):
                    print("🔍 Zoom detected")

                if gestures.detect_window_switch(landmarks):
                    print("🔄 Window switch detected")

            cv2.imshow("Gesture Mouse", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🔴 Exiting program...")
                break

    except Exception as e:
        print(f"❌ Error occurred: {e}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
