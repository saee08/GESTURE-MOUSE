import pyautogui
import numpy as np
import time

class ClickScroll:
    def __init__(self):
        self.click_active = False
        self.scroll_active = False

    def detect_click(self, landmarks):
        """Detects click when thumb and index finger tips are close."""
        if len(landmarks) > 8:
            index_tip = np.array(landmarks[8])  # Index finger tip
            thumb_tip = np.array(landmarks[4])  # Thumb tip
            
            distance = np.linalg.norm(index_tip - thumb_tip)  # Euclidean distance
            
            if distance < 30 and not self.click_active:
                pyautogui.click()
                self.click_active = True
                time.sleep(0.3)  # Small delay to prevent rapid clicks
            elif distance > 40:
                self.click_active = False

    def detect_scroll(self, landmarks):
        """Detects scrolling using index and middle finger tips."""
        if len(landmarks) > 12:
            index_tip = landmarks[8]   # Index finger tip
            middle_tip = landmarks[12] # Middle finger tip

            if index_tip[1] < middle_tip[1] and not self.scroll_active:
                pyautogui.scroll(3)  # Scroll up
                self.scroll_active = True
                time.sleep(0.2)  # Prevents continuous rapid scrolling
            elif index_tip[1] > middle_tip[1]:
                self.scroll_active = False
