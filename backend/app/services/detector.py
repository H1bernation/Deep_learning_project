from ultralytics import YOLO
import cv2
import numpy as np
from typing import Dict, List

class FaceDetectionService:
    def __init__(self):
        self.model = YOLO('/app/models/detection.pt')
        self.class_names = [
            'forehead', 'left_eye', 'right_eye', 
            'nose', 'left_cheek', 'right_cheek', 'chin'
        ]
    
    def detect(self, image_bytes: bytes) -> Dict:
        """
        얼굴 부위 검출
        
        Returns:
            {
                'boxes': [{'class': 'forehead', 'bbox': [x1,y1,x2,y2], 'confidence': 0.95}, ...],
                'image_shape': (height, width)
            }
        """
        # bytes -> numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 검출 실행
        results = self.model(image)
        
        boxes = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                
                boxes.append({
                    'class': self.class_names[class_id],
                    'bbox': bbox,
                    'confidence': confidence
                })
        
        return {
            'boxes': boxes,
            'image_shape': image.shape[:2]  # (height, width)
        }