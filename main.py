import cv2
import os
from src.depth_estimation import MiDaSDepthEstimator
from src.object_detection import ObjectDetector
from src.motion_detection import MotionDetector

depth_estimator = MiDaSDepthEstimator()
detector = ObjectDetector()
motion_detector = MotionDetector()

cap = cv2.VideoCapture(0)
saved = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

 
    boxes = detector.detect(frame)

    
    depth_map = depth_estimator.estimate(frame)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        
        raw_depth = depth_map[cy, cx]
        norm_depth = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)
        distance_m = round((1.0 - norm_depth[cy, cx]) * 10, 2) 

      
        is_moving = motion_detector.is_moving(frame, (x1, y1, x2, y2))
        motion_status = "Moving" if is_moving else "Static"

       
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{motion_status}, Dist: {distance_m}m",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 0), 2)

    if not saved:
        os.makedirs("data/outputs", exist_ok=True)
        normalized_depth = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = normalized_depth.astype("uint8")
        cv2.imwrite("data/outputs/depth_map.png", depth_uint8)
        saved = True
        print("depth_map.png saved!")

    cv2.imshow("Real-Time Road Safety System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()