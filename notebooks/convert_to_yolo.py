import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import shutil

CLASS_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6}

def convert_bbox_to_yolo(bbox, img_width, img_height):
    x, y, w, h = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    return x_center, y_center, width, height

def process_dataset(json_base, img_base, output_dir, split_name):
    print(f"\n{'='*50}")
    print(f"{split_name.upper()} 변환")
    print(f"{'='*50}")
    
    os.makedirs(f"{output_dir}/images/{split_name}", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/{split_name}", exist_ok=True)
    
    json_files = []
    for root, dirs, files in os.walk(json_base):
        for f in files:
            if f.endswith('.json'):
                json_files.append(Path(root) / f)
    
    print(f"{len(json_files)}개 JSON")
    
    label_dict = {}
    success = 0
    skipped = 0
    
    for json_file in tqdm(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            img_filename = data['info']['filename']
            relative = json_file.parent.relative_to(Path(json_base))
            img_path = Path(img_base) / relative / img_filename
            
            if not img_path.exists():
                skipped += 1
                continue
            
            img = Image.open(img_path)
            w, h = img.size
            
            facepart = data['images']['facepart']
            if facepart not in CLASS_MAP:
                skipped += 1
                continue
            
            bbox = data['images']['bbox']
            class_id = CLASS_MAP[facepart]
            xc, yc, bw, bh = convert_bbox_to_yolo(bbox, w, h)
            
            if img_filename not in label_dict:
                label_dict[img_filename] = {'labels': [], 'img_path': img_path}
            label_dict[img_filename]['labels'].append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            success += 1
        except:
            skipped += 1
    
    for img_filename, data in tqdm(label_dict.items()):
        txt_path = Path(output_dir) / "labels" / split_name / (Path(img_filename).stem + ".txt")
        with open(txt_path, 'w') as f:
            f.write('\n'.join(data['labels']))
        
        img_out = Path(output_dir) / "images" / split_name / img_filename
        shutil.copy2(data['img_path'], img_out)
    
    print(f"✅ {len(label_dict)}개 이미지")
    return len(label_dict)

if __name__ == "__main__":
    # 여기 경로만 수정하세요
    BASE = r"C:\Users\황동민\Downloads\data\Skin_data" 
    OUTPUT = r"C:\Users\황동민\Downloads\data\Skin_data\yolo_dataset" 
    
    train = process_dataset(
        f"{BASE}/Training/02.라벨링데이터/TL",
        f"{BASE}/Training/01.원천데이터/TS",
        OUTPUT,
        "train"
    )
    
    val = process_dataset(
        f"{BASE}/Validation/02.라벨링데이터/VL",
        f"{BASE}/Validation/01.원천데이터/VS",
        OUTPUT,
        "val"
    )
    
    print(f"\n완료: Train {train}개, Val {val}개")