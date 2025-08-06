import os
import cv2
import shutil
from PIL import Image
from sklearn.model_selection import train_test_split
import albumentations as A
import numpy as np

def process_pothole_dataset(raw_dir="raw_data", output_dir="processed"):
    """Process raw pothole images into YOLOv8-ready dataset"""
    
    # Get absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_dir = os.path.join(script_dir, raw_dir)
    output_dir = os.path.join(script_dir, output_dir)

    # Check if raw_data directory exists
    if not os.path.exists(raw_dir):
        print(f"⚠️ Error: Directory not found: {raw_dir}")
        print("Please create a 'raw_data' folder and add your pothole images/videos.")
        print(f"Creating empty 'raw_data' folder at: {raw_dir}")
        os.makedirs(raw_dir, exist_ok=True)
        return
    
    # Check if raw_data is empty
    if not os.listdir(raw_dir):
        print(f"⚠️ Error: The 'raw_data' folder is empty: {raw_dir}")
        print("Please add your pothole images/videos and run again.")
        return

    # Create directory structure
    dirs = [
        os.path.join(output_dir, "images/train"),
        os.path.join(output_dir, "images/val"),
        os.path.join(output_dir, "labels/train"),
        os.path.join(output_dir, "labels/val"),
        os.path.join(output_dir, "temp_frames"),
        os.path.join(output_dir, "augmented")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # Process images and videos
    image_files = []
    
    for file in os.listdir(raw_dir):
        file_path = os.path.join(raw_dir, file)
        
        # Handle images
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                with Image.open(file_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.thumbnail((640, 640))  # YOLOv8 optimal size
                    save_path = os.path.join(output_dir, "augmented", file)
                    img.save(save_path)
                    image_files.append(save_path)
            except Exception as e:
                print(f"Error processing {file}: {e}")
        
        # Handle videos
        elif file.lower().endswith(('.mp4', '.mov', '.avi')):
            try:
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    raise ValueError(f"Could not open video {file}")
                    
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % 10 == 0:  # Extract every 10th frame
                        frame_path = os.path.join(output_dir, "temp_frames", 
                                               f"{os.path.splitext(file)[0]}_{frame_count}.jpg")
                        cv2.imwrite(frame_path, frame)
                        image_files.append(frame_path)
                    
                    frame_count += 1
                
                cap.release()
            except Exception as e:
                print(f"Error processing video {file}: {e}")

    if not image_files:
        print("⚠️ Error: No valid images/videos found to process")
        return

    # Data augmentation
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
    ])
    
    augmented_images = []
    for img_path in image_files:
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Could not read image {img_path}")
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            for i in range(3):  # Create 3 augmented versions
                augmented = transform(image=image)['image']
                aug_path = os.path.join(output_dir, "augmented", 
                                     f"aug_{i}_{os.path.basename(img_path)}")
                cv2.imwrite(aug_path, cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR))
                augmented_images.append(aug_path)
        except Exception as e:
            print(f"Error augmenting {img_path}: {e}")

    # Prepare YOLOv8 dataset
    all_images = image_files + augmented_images
    if not all_images:
        print("⚠️ Error: No images available for dataset creation")
        return
        
    train_files, val_files = train_test_split(all_images, test_size=0.2)
    
    # Move files to YOLOv8 structure
    for file in train_files:
        try:
            shutil.copy(file, os.path.join(output_dir, "images/train", os.path.basename(file)))
            label_file = os.path.splitext(os.path.basename(file))[0] + ".txt"
            open(os.path.join(output_dir, "labels/train", label_file), 'a').close()
        except Exception as e:
            print(f"Error copying training file {file}: {e}")
    
    for file in val_files:
        try:
            shutil.copy(file, os.path.join(output_dir, "images/val", os.path.basename(file)))
            label_file = os.path.splitext(os.path.basename(file))[0] + ".txt"
            open(os.path.join(output_dir, "labels/val", label_file), 'a').close()
        except Exception as e:
            print(f"Error copying validation file {file}: {e}")
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"• Training images: {len(train_files)}")
    print(f"• Validation images: {len(val_files)}")
    print(f"• Output directory: {output_dir}")

if __name__ == "__main__":
    process_pothole_dataset()