import cv2
import numpy as np
import random

def add_dark_road_shadows(image_path, output_path, max_darkness=0.7):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error: Failed to load image")
        return False
    
    h, w = img.shape[:2]
    
    # Target zone (middle-bottom 1/3rd)
    zone_y_start = h - (h // 3)
    zone_x_center = w // 4
    
    # Create shadow mask (initialize as white)
    shadow_mask = np.ones_like(img, dtype=np.float32)
    
    # Add 20-30 tiny DARK patches
    for _ in range(random.randint(5,8)):
        # Random position in target zone
        x = random.randint(zone_x_center - w//4, zone_x_center + w//4)
        y = random.randint(zone_y_start, h - 10)
        
        # Tiny circle parameters
        radius = random.randint(10, 20)  # 5-15 pixel radius
        darkness = 1.0 - random.uniform(0.2, max_darkness)  # 0.8-0.3 darkness factor
        
        # Draw DARK circle (subtracts light)
        cv2.circle(shadow_mask, (x, y), radius, 
                  (darkness, darkness, darkness), -1, lineType=cv2.LINE_AA)
    
    # Apply Gaussian blur for soft edges
    shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
    
    # Multiply to darken image (not add)
    result = (img.astype(np.float32) * shadow_mask).clip(0, 255).astype(np.uint8)
    
    cv2.imwrite(output_path, result)
    print(f"✔ Saved {output_path} with {20-30} micro-dark-spots")
    return True

# Usage (adjust max_darkness 0.3-0.8)
input_path = r"C:\Users\anike\Downloads\Flux_Dev_realistic_photo_of_a_scenic_Indian_highway_viewed_fro_2.jpg"
output_path = r"C:\Users\anike\Downloads\highway_dark_shadows.jpg"
add_dark_road_shadows(input_path, output_path, max_darkness=0.6)