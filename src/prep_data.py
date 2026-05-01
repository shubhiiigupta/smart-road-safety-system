import os
import zipfile
import random
import shutil
import xml.etree.ElementTree as ET

ZIP_PATH = r"c:\Users\gupta\smart_road_safety_system\data\archive (7).zip"
RAW_DIR = r"c:\Users\gupta\smart_road_safety_system\data\raw"
OUTPUT_DIR = r"c:\Users\gupta\smart_road_safety_system\data"

CLASS_MAP = {
    'With Helmet': 0,
    'Without Helmet': 1
}

def convert_voc_to_yolo(xml_file, output_txt):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    yolo_labels = []
    for obj in root.findall('object'):
        cls_name = obj.find('name').text
        if cls_name not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[cls_name]
        
        xmlbox = obj.find('bndbox')
        xmin = float(xmlbox.find('xmin').text)
        xmax = float(xmlbox.find('xmax').text)
        ymin = float(xmlbox.find('ymin').text)
        ymax = float(xmlbox.find('ymax').text)
        
        # YOLO format: x_center y_center width height
        x_center = (xmin + xmax) / 2.0 / w
        y_center = (ymin + ymax) / 2.0 / h
        box_w = (xmax - xmin) / w
        box_h = (ymax - ymin) / h
        
        yolo_labels.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
        
    with open(output_txt, 'w') as f:
        f.write('\n'.join(yolo_labels))

def main():
    print("Extracting zip file...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(RAW_DIR)
        
    annotations_dir = os.path.join(RAW_DIR, "annotations")
    images_dir = os.path.join(RAW_DIR, "images")
    
    images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.seed(42)
    random.shuffle(images)
    
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    print(f"Total images: {len(images)} | Train: {len(train_images)} | Val: {len(val_images)}")
    
    def process_split(split_images, split_name):
        for img_name in split_images:
            base_name = os.path.splitext(img_name)[0]
            xml_name = base_name + ".xml"
            txt_name = base_name + ".txt"
            
            src_img = os.path.join(images_dir, img_name)
            src_xml = os.path.join(annotations_dir, xml_name)
            
            dest_img = os.path.join(OUTPUT_DIR, "images", split_name, img_name)
            dest_txt = os.path.join(OUTPUT_DIR, "labels", split_name, txt_name)
            
            # Copy image
            shutil.copy(src_img, dest_img)
            
            # Convert and save label if xml exists
            if os.path.exists(src_xml):
                convert_voc_to_yolo(src_xml, dest_txt)
            else:
                # If no xml, save empty txt
                open(dest_txt, 'w').close()
                
    print("Processing Train set...")
    process_split(train_images, "train")
    
    print("Processing Validation set...")
    process_split(val_images, "val")
    
    print("Cleaning up raw files...")
    shutil.rmtree(RAW_DIR)
    print("Dataset preparation complete!")

if __name__ == "__main__":
    main()
