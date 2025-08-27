import cv2
import numpy as np

class MotionDetector:
    def __init__(self):
        self.prev_frame = None

    def is_moving(self, current_frame, bbox):
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        if self.prev_frame is None:
            self.prev_frame = gray
            return False

        (x1, y1, x2, y2) = map(int, bbox)
        roi_current = gray[y1:y2, x1:x2]
        roi_prev = self.prev_frame[y1:y2, x1:x2]

        if roi_current.shape != roi_prev.shape:
            self.prev_frame = gray
            return False

        diff = cv2.absdiff(roi_current, roi_prev)
        mean_diff = np.mean(diff)

        self.prev_frame = gray
        return mean_diff > 10