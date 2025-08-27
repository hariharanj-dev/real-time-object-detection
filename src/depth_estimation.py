import torch
import cv2
import numpy as np

class MiDaSDepthEstimator:
    def __init__(self):
        print("🔄 Loading MiDaS model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = "MiDaS"

        self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self.model.to(self.device)
        self.model.eval()

        self.transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = self.transforms.dpt_transform
        print("MiDaS model loaded.")

    def estimate(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        return depth_map