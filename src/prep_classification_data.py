import os
import zipfile
import random
import shutil
import xml.etree.ElementTree as ET
from PIL import Image

ZIP_PATH = r"c:\Users\gupta\smart_road_safety_system\data\archive (7).zip"
RAW_DIR = r"c:\Users\gupta\smart_road_safety_system\data\raw_class"
OUTPUT_DIR = r"c:\Users\gupta\smart_road_safety_system\data\classification"

CLASS_MAP = {
    'With Helmet': 'With_Helmet',
    'Without Helmet': 'Without_Helmet'
}

def main():
    print("Extracting zip file...")
    if os.path.exists(RAW_DIR):
        shutil.rmtree(RAW_DIR)
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(RAW_DIR)
        
    annotations_dir = os.path.join(RAW_DIR, "annotations")
    images_dir = os.path.join(RAW_DIR, "images")
    
    # Setup output directories
    for split in ['train', 'val']:
        for cls in CLASS_MAP.values():
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)
            
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.seed(42)
    random.shuffle(images)
    
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    print(f"Total source images: {len(images)} | Train split: {len(train_images)} | Val split: {len(val_images)}")
    
    crop_counts = {'train': 0, 'val': 0}
    
    def process_split(split_images, split_name):
        for img_name in split_images:
            base_name = os.path.splitext(img_name)[0]
            xml_name = base_name + ".xml"
            
            src_img = os.path.join(images_dir, img_name)
            src_xml = os.path.join(annotations_dir, xml_name)
            
            if not os.path.exists(src_xml):
                continue
                
            try:
                img = Image.open(src_img)
                # Convert to RGB to ensure consistency
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                    
                tree = ET.parse(src_xml)
                root = tree.getroot()
                
                # Crop each object
                obj_idx = 0
                for obj in root.findall('object'):
                    cls_name = obj.find('name').text
                    if cls_name not in CLASS_MAP:
                        continue
                        
                    dir_name = CLASS_MAP[cls_name]
                    
                    xmlbox = obj.find('bndbox')
                    xmin = float(xmlbox.find('xmin').text)
                    xmax = float(xmlbox.find('xmax').text)
                    ymin = float(xmlbox.find('ymin').text)
                    ymax = float(xmlbox.find('ymax').text)
                    
                    # Ensure within bounds
                    width, height = img.size
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(width, xmax)
                    ymax = min(height, ymax)
                    
                    if xmax <= xmin or ymax <= ymin:
                        continue
                        
                    cropped_img = img.crop((xmin, ymin, xmax, ymax))
                    
                    save_name = f"{base_name}_{obj_idx}.png"
                    save_path = os.path.join(OUTPUT_DIR, split_name, dir_name, save_name)
                    
                    cropped_img.save(save_path)
                    crop_counts[split_name] += 1
                    obj_idx += 1
                    
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                
    print("Cropping and saving Train set...")
    process_split(train_images, "train")
    
    print("Cropping and saving Validation set...")
    process_split(val_images, "val")
    
    print("Cleaning up raw files...")
    shutil.rmtree(RAW_DIR)
    print(f"Dataset preparation complete! Total train crops: {crop_counts['train']} | Total val crops: {crop_counts['val']}")

if __name__ == "__main__":
    main()
