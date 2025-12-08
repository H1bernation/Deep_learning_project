import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import io
from typing import Dict

class SkinClassifier(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feature_dim = self.backbone.num_features
        
        self.wrinkle_head = nn.Linear(feature_dim, 7)
        self.pigmentation_head = nn.Linear(feature_dim, 6)
        self.pore_head = nn.Linear(feature_dim, 6)
        self.sagging_head = nn.Linear(feature_dim, 7)

    def forward(self, x):
        features = self.backbone(x)
        return {
            'wrinkle': self.wrinkle_head(features),
            'pigmentation': self.pigmentation_head(features),
            'pore': self.pore_head(features),
            'sagging': self.sagging_head(features)
        }

class EnsembleModel(nn.Module):
    def __init__(self, model1, model2, weights=[0.5, 0.5]):
        super().__init__()
        self.model1 = model1
        self.model2 = model2
        self.weights = weights

    def forward(self, x):
        outputs1 = self.model1(x)
        outputs2 = self.model2(x)
        ensemble_outputs = {}
        for symptom in ['wrinkle', 'pigmentation', 'pore', 'sagging']:
            prob1 = torch.softmax(outputs1[symptom], dim=1)
            prob2 = torch.softmax(outputs2[symptom], dim=1)
            ensemble_outputs[symptom] = self.weights[0] * prob1 + self.weights[1] * prob2
        return ensemble_outputs

class SkinClassificationService:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 구조 생성
        vit_model = SkinClassifier('vit_base_patch16_224', pretrained=False)
        effnet_model = SkinClassifier('tf_efficientnetv2_s', pretrained=False)
        self.model = EnsembleModel(vit_model, effnet_model, weights=[0.5, 0.5])
        
        # State dict 로드
        self.model.load_state_dict(
            torch.load('/app/models/classification.pth', map_location=self.device)
        )
        self.model.eval()
        
        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.symptom_names = ['wrinkle', 'pore', 'pigmentation', 'sagging']
    
    def classify_region(self, image_bytes: bytes, bbox: list) -> Dict:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cropped = image.crop((x1, y1, x2, y2))
        
        input_tensor = self.transform(cropped).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            result = {}
            for symptom in self.symptom_names:
                probs = outputs[symptom][0]
                pred = torch.argmax(probs).item()
                
                # 7단계를 4단계로 변환 (0-6 -> 0-3)
                if symptom == 'wrinkle' or symptom == 'sagging':
                    result[symptom] = min(pred // 2, 3)
                else:  # pore, pigmentation (0-5)
                    result[symptom] = min(pred // 2, 3)
        
        return result
    
    def classify_all_regions(self, image_bytes: bytes, boxes: list) -> Dict:
        all_results = []
        
        for box_info in boxes:
            bbox = box_info['bbox']
            result = self.classify_region(image_bytes, bbox)
            all_results.append(result)
        
        if not all_results:
            return {symptom: 0 for symptom in self.symptom_names}

        max_result = {}
        for symptom in self.symptom_names:
            values = [r[symptom] for r in all_results]
            max_result[symptom] = max(values)

        return max_result