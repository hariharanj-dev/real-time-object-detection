import cv2
import numpy as np

predicted_depth = cv2.imread("data/outputs/depth_map.png", cv2.IMREAD_UNCHANGED)
if predicted_depth is None:
    print("Error: depth_map.png not found!")
    exit()

predicted_depth = predicted_depth.astype(np.float32) / 255.0
ground_truth = np.ones_like(predicted_depth)

mae = np.mean(np.abs(predicted_depth - ground_truth))
accuracy = (1 - mae) * 100
print(f"MAE: {mae:.4f} | Accuracy: {accuracy:.2f}%")