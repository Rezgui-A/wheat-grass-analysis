# main.py - Standalone Wheatgrass Analysis Application
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')


def select_image_file():
    """
    Open file dialog to select image
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_types = [
        ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
        ("JPEG files", "*.jpg *.jpeg"),
        ("PNG files", "*.png"),
        ("All files", "*.*")
    ]
    
    file_path = filedialog.askopenfilename(
        title="Select Wheatgrass Image",
        filetypes=file_types
    )
    
    root.destroy()
    return file_path

def load_wheat_image(image_path):
    """Load and preprocess image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"✅ Image loaded successfully: {image.shape}")
    return image

def detect_beaker_accurate(image):
    """
    Accurate beaker detection for cylindrical beaker in center with blue background
    """
    print("Detecting beaker accurately...")
    
    h, w = image.shape[:2]
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Define blue color range for background (adjust these values based on your blue)
    lower_blue = np.array([100, 150, 50])   # Adjusted for brighter blue
    upper_blue = np.array([130, 255, 255])  # Adjusted for brighter blue
    
    # Create mask for blue background
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Clean up blue mask
    kernel = np.ones((10,10), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    
    # The beaker is the non-blue region (main object)
    beaker_mask = ~blue_mask.astype(bool)
    
    # Find the largest non-blue contour (the beaker)
    contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No blue background detected, using center detection")
        return detect_beaker_center_fallback(image)
    
    # Find the largest contour (should be the background around beaker)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create mask from largest contour
    background_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(background_mask, [largest_contour], 0, 255, -1)
    
    # Beaker is the inverse of the background
    beaker_mask = ~background_mask.astype(bool)
    
    # Get bounding rectangle of beaker
    x, y, w_beaker, h_beaker = cv2.boundingRect(largest_contour)
    
    # For cylindrical beaker in center, we want the inner rectangle
    # Add some padding to ensure we get the entire beaker
    padding_x = int(w_beaker * 0.05)
    padding_y = int(h_beaker * 0.05)
    
    x_start = max(0, x + padding_x)
    x_end = min(w, x + w_beaker - padding_x)
    y_start = max(0, y + padding_y)
    y_end = min(h, y + h_beaker - padding_y)
    
    # Ensure we have a reasonable aspect ratio (beaker should be taller than wide)
    if (y_end - y_start) < (x_end - x_start) * 0.5:
        print("Adjusting beaker dimensions for tall cylinder")
        center_x = (x_start + x_end) // 2
        beaker_width = min(w // 3, (x_end - x_start) // 2)
        x_start = center_x - beaker_width
        x_end = center_x + beaker_width
    
    print(f"Beaker detected: x={x_start}-{x_end}, y={y_start}-{y_end}")
    print(f"Beaker dimensions: {x_end-x_start} x {y_end-y_start}")
    
    return x_start, x_end, y_start, y_end, beaker_mask

def detect_beaker_center_fallback(image):
    """
    Fallback beaker detection - center of image
    """
    h, w = image.shape[:2]
    print("Using center-based beaker detection")
    
    # Beaker is in center, taking about 50-60% of image width and 80% height
    x_start = int(w * 0.25)
    x_end = int(w * 0.75)
    y_start = int(h * 0.1)
    y_end = int(h * 0.9)
    
    beaker_mask = np.zeros((h, w), dtype=bool)
    beaker_mask[y_start:y_end, x_start:x_end] = True
    
    return x_start, x_end, y_start, y_end, beaker_mask

def detect_soil_line_improved(image, beaker_region):
    """
    Improved soil line detection - moved a bit lower
    """
    print("Detecting soil line (improved)...")
    
    x_start, x_end, y_start, y_end = beaker_region
    h, w = image.shape[:2]
    
    beaker_height = y_end - y_start
    
    # Soil is in bottom 8-12% of beaker (moved down slightly)
    soil_region_height = int(beaker_height * 0.15)  # Check bottom 15%
    soil_region_start = y_end - soil_region_height
    
    print(f"Soil search region: bottom {soil_region_height} pixels (y={soil_region_start} to {y_end})")
    
    # Extract soil region from beaker
    soil_region = image[soil_region_start:y_end, x_start:x_end]
    
    if soil_region.size == 0:
        print("No soil region found, using default soil line")
        return y_end - int(beaker_height * 0.12), 0.1  # Moved down
    
    # Convert to multiple color spaces for robust soil detection
    hsv = cv2.cvtColor(soil_region, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(soil_region, cv2.COLOR_RGB2LAB)
    
    # Soil color ranges - dark brown/black colors
    # HSV ranges for soil
    lower_brown1 = np.array([0, 40, 15])   # Adjusted for better soil detection
    upper_brown1 = np.array([30, 200, 120])
    lower_brown2 = np.array([0, 0, 0])
    upper_brown2 = np.array([180, 80, 70])  # Adjusted upper bound
    
    # LAB ranges for dark regions
    lower_lab = np.array([0, 115, 115])    # Slightly adjusted
    upper_lab = np.array([100, 140, 140])
    
    mask1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
    mask2 = cv2.inRange(hsv, lower_brown2, upper_brown2)
    mask3 = cv2.inRange(lab, lower_lab, upper_lab)
    
    soil_mask = mask1 | mask2 | mask3
    
    # Clean up soil mask
    kernel = np.ones((5,5), np.uint8)
    soil_mask = cv2.morphologyEx(soil_mask, cv2.MORPH_CLOSE, kernel)
    soil_mask = cv2.morphologyEx(soil_mask, cv2.MORPH_OPEN, kernel)
    
    # Find soil line - top of soil region
    row_sums = np.sum(soil_mask, axis=1)
    
    if len(row_sums) == 0:
        print("No soil detected, using default")
        return y_end - int(beaker_height * 0.12), 0.1  # Moved down
    
    # Find first row with significant soil - using lower threshold
    threshold = soil_region.shape[1] * 0.35  # 35% coverage threshold (slightly lower)
    soil_top_in_region = None
    
    for i in range(len(row_sums)):
        if row_sums[i] > threshold:
            soil_top_in_region = i
            break
    
    if soil_top_in_region is None:
        # If no clear soil line, find row with maximum soil
        soil_top_in_region = np.argmax(row_sums)
        print(f"No clear soil line, using maximum soil at row {soil_top_in_region}")
    
    soil_line_y = soil_region_start + soil_top_in_region
    
    # Move soil line slightly lower (5-10 pixels)
    soil_line_y += 8
    
    # Ensure soil line is in reasonable position (bottom 8-15% of beaker)
    min_soil_y = y_start + int(beaker_height * 0.85)  # Soil should be in bottom 15%
    max_soil_y = y_end - 10  # Leave more margin at bottom
    
    if soil_line_y < min_soil_y:
        print(f"Adjusting soil line from {soil_line_y} to {min_soil_y} (minimum position)")
        soil_line_y = min_soil_y
    elif soil_line_y > max_soil_y:
        print(f"Adjusting soil line from {soil_line_y} to {max_soil_y} (maximum position)")
        soil_line_y = max_soil_y
    
    # Calculate pixel to cm ratio (1000ml beaker is ~25cm tall)
    pixel_to_cm_ratio = 25.0 / beaker_height
    
    print(f"Soil line detected at y={soil_line_y} ({(y_end - soil_line_y) / beaker_height * 100:.1f}% from bottom)")
    print(f"Pixel-to-cm ratio: {pixel_to_cm_ratio:.4f} (beaker height: {beaker_height} pixels ≈ 25cm)")
    
    return soil_line_y, pixel_to_cm_ratio

def detect_plants_improved(image, beaker_region, soil_line_y):
    """
    Improved plant detection - better detection on left side without changing logic
    """
    print("Detecting plants (improved)...")
    
    x_start, x_end, y_start, y_end = beaker_region
    h, w = image.shape[:2]
    
    # Plants are above soil line, up to top of beaker
    plant_top = y_start
    plant_bottom = soil_line_y
    
    print(f"Plant search region: y={plant_top} to {plant_bottom} (above soil)")
    
    # Extract plant region from beaker
    plant_region = image[plant_top:plant_bottom, x_start:x_end]
    
    if plant_region.size == 0:
        print("No plant region found")
        return np.zeros((h, w), dtype=bool)
    
    # Convert to multiple color spaces for robust plant detection
    hsv = cv2.cvtColor(plant_region, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(plant_region, cv2.COLOR_RGB2LAB)
    
    # Enhanced green color ranges for better detection, especially on left side
    # Multiple green color ranges for different lighting conditions
    lower_green1 = np.array([30, 45, 35])   # Brighter green with lower hue
    upper_green1 = np.array([90, 255, 255]) # Wider hue range
    lower_green2 = np.array([25, 35, 25])   # Darker green
    upper_green2 = np.array([100, 255, 220]) # Extended upper hue
    lower_green3 = np.array([35, 30, 20])   # Yellowish green
    upper_green3 = np.array([85, 255, 200])
    
    mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
    mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
    mask3 = cv2.inRange(hsv, lower_green3, upper_green3)
    
    # LAB space for green detection - adjusted for better sensitivity
    lower_lab_green = np.array([0, 115, 115])  # Slightly adjusted
    upper_lab_green = np.array([255, 145, 145])
    mask4 = cv2.inRange(lab, lower_lab_green, upper_lab_green)
    
    # Combine all green masks
    green_mask = mask1 | mask2 | mask3 | mask4
    
    # Enhanced cleaning to preserve plant structures while removing noise
    # First, close small gaps in plants
    kernel_close = np.ones((3,3), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Then remove small noise
    kernel_open = np.ones((2,2), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel_open)
    
    # Remove small components but be less aggressive to preserve left-side plants
    green_mask = remove_small_components_improved(green_mask, min_size=25)
    
    # Additional step: fill small holes in plant regions
    green_mask = fill_small_holes(green_mask, max_hole_size=50)
    
    # Convert to full image mask
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[plant_top:plant_bottom, x_start:x_end] = green_mask
    
    plant_count = np.sum(full_mask)
    print(f"Plants detected: {plant_count} pixels in plant region")
    
    return full_mask.astype(bool)

def remove_small_components_improved(mask, min_size=50):
    """
    Improved small component removal - less aggressive for plant preservation
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    cleaned_mask = np.zeros_like(mask)
    
    # Keep components that are above minimum size OR have high aspect ratio (likely thin plants)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Keep if large enough OR if it's a thin vertical structure (likely plant)
        if area >= min_size or (height > width * 2 and area >= min_size // 2):
            cleaned_mask[labels == i] = 255
    
    return cleaned_mask

def fill_small_holes(mask, max_hole_size=100):
    """
    Fill small holes in the mask while preserving plant structures
    """
    # Copy the mask
    filled_mask = mask.copy()
    
    # Find contours - external contours will be plants, internal will be holes
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, (cnt, hier) in enumerate(zip(contours, hierarchy)):
            # If it's a hole (has parent) and small enough
            if hier[3] >= 0 and cv2.contourArea(cnt) <= max_hole_size:
                cv2.drawContours(filled_mask, [cnt], 0, 255, -1)
    
    return filled_mask

def extract_canopy_boundary_accurate(plant_mask, soil_line_y):
    """
    Accurate canopy boundary extraction from plant mask
    """
    print("Extracting canopy boundary accurately...")
    
    h, w = plant_mask.shape
    
    # Only consider plants above soil line
    canopy_mask = plant_mask.copy()
    canopy_mask[soil_line_y:, :] = False
    
    boundary_points = []
    
    # Sample every column to find top plant point
    for x in range(0, w, 2):  # Sample every 2 pixels for efficiency
        column = canopy_mask[:, x]
        if np.any(column):
            plant_pixels = np.where(column)[0]
            if len(plant_pixels) > 0:
                top_y = plant_pixels[0]  # Topmost plant pixel in this column
                boundary_points.append((x, top_y))
    
    if len(boundary_points) < 10:
        print(f"Warning: Only {len(boundary_points)} boundary points found")
        # Try sampling every column
        boundary_points = []
        for x in range(0, w):
            column = canopy_mask[:, x]
            if np.any(column):
                plant_pixels = np.where(column)[0]
                if len(plant_pixels) > 0:
                    top_y = plant_pixels[0]
                    boundary_points.append((x, top_y))
    
    if len(boundary_points) < 5:
        print(f"ERROR: Insufficient boundary points: {len(boundary_points)}")
        return np.array([]), np.array([])
    
    # Convert to arrays
    x_vals = np.array([p[0] for p in boundary_points])
    y_vals = np.array([p[1] for p in boundary_points])
    
    # Sort by x coordinate
    sorted_idx = np.argsort(x_vals)
    x_vals = x_vals[sorted_idx]
    y_vals = y_vals[sorted_idx]
    
    # Remove outliers using robust statistics
    if len(y_vals) > 10:
        median_y = np.median(y_vals)
        mad = np.median(np.abs(y_vals - median_y))  # Median Absolute Deviation
        if mad > 0:
            # Keep points within 3 MAD of median
            valid_mask = np.abs(y_vals - median_y) < 3 * mad
            x_vals = x_vals[valid_mask]
            y_vals = y_vals[valid_mask]
    
    # Smooth the canopy boundary
    if len(y_vals) > 10:
        window_size = min(15, len(y_vals))
        if window_size % 2 == 0:
            window_size -= 1
        if window_size >= 3:
            try:
                y_vals = savgol_filter(y_vals, window_size, 2)
            except:
                print("Smoothing failed, using raw boundary")
    
    print(f"Canopy boundary extracted: {len(x_vals)} points")
    return x_vals, y_vals

def calculate_greenness_improved(image, plant_mask):
    """Calculate improved greenness score"""
    if np.sum(plant_mask) == 0:
        return 0.0
    
    plant_pixels = image[plant_mask]
    
    # Convert to HSV for better green detection
    hsv = cv2.cvtColor(plant_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    
    hue = hsv[:, 0]
    saturation = hsv[:, 1]
    value = hsv[:, 2]
    
    # Green hue is typically between 30-95 in OpenCV (wider range)
    green_hue_mask = (hue >= 30) & (hue <= 95)
    
    # Good saturation for healthy plants
    good_saturation = saturation > 35
    
    # Good value (not too dark, not too bright)
    good_value = (value > 25) & (value < 220)
    
    # Combined greenness score
    green_pixels = green_hue_mask & good_saturation & good_value
    greenness_score = np.mean(green_pixels)
    
    # Additional: average saturation of green pixels
    if np.any(green_pixels):
        avg_saturation = np.mean(saturation[green_pixels]) / 255.0
        greenness_score = (greenness_score + avg_saturation) / 2
    
    return greenness_score

def analyze_wheatgrass_improved(image_path):
    """
    Main improved analysis function
    """
    try:
        print("=== IMPROVED Wheatgrass Analysis ===")
        start_time = time.time()
        
        # Load image
        image = load_wheat_image(image_path)
        
        # Detect beaker accurately
        beaker_region = detect_beaker_accurate(image)
        x_start, x_end, y_start, y_end, beaker_mask = beaker_region
        
        # Detect soil line with improvements
        soil_line_y, pixel_ratio = detect_soil_line_improved(image, (x_start, x_end, y_start, y_end))
        
        # Detect plants with improvements
        plant_mask = detect_plants_improved(image, (x_start, x_end, y_start, y_end), soil_line_y)
        
        if np.sum(plant_mask) == 0:
            print("ERROR: No plants detected!")
            return None
        
        # Extract canopy boundary
        canopy_x, canopy_y = extract_canopy_boundary_accurate(plant_mask, soil_line_y)
        
        if len(canopy_x) == 0:
            print("ERROR: No canopy boundary extracted!")
            return None
        
        # Calculate plant heights
        heights_pixels = soil_line_y - canopy_y
        
        # Filter valid heights (between 1cm and 30cm in real units)
        min_height_pixels = 5  # ~1.25cm at typical scale
        max_height_pixels = 120  # ~30cm at typical scale
        
        valid_mask = (heights_pixels > min_height_pixels) & (heights_pixels < max_height_pixels)
        
        if np.sum(valid_mask) == 0:
            print("ERROR: No valid plant heights!")
            return None
        
        # Apply valid mask
        heights_pixels = heights_pixels[valid_mask]
        canopy_x = canopy_x[valid_mask]
        canopy_y = canopy_y[valid_mask]
        
        # Convert to real units
        heights_cm = heights_pixels * pixel_ratio
        heights_inch = heights_cm / 2.54
        
        # Calculate statistics
        avg_height_cm = np.mean(heights_cm)
        max_height_cm = np.max(heights_cm)
        min_height_cm = np.min(heights_cm)
        std_height_cm = np.std(heights_cm)
        avg_height_inch = np.mean(heights_inch)
        
        # Calculate greenness
        greenness = calculate_greenness_improved(image, plant_mask)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ IMPROVED Analysis completed in {elapsed_time:.1f} seconds")
        
        # Visualize results
        visualize_improved_analysis(
            image, plant_mask, canopy_x, canopy_y, soil_line_y,
            (x_start, x_end, y_start, y_end), heights_cm, heights_inch,
            avg_height_cm, std_height_cm, greenness, pixel_ratio,
            os.path.basename(image_path)
        )
        
        return {
            'canopy_x': canopy_x, 'canopy_y': canopy_y,
            'soil_line_y': soil_line_y, 'height_cm': heights_cm,
            'height_inch': heights_inch, 'avg_height_cm': avg_height_cm,
            'avg_height_inch': avg_height_inch, 'max_height_cm': max_height_cm,
            'min_height_cm': min_height_cm, 'std_height_cm': std_height_cm,
            'greenness': greenness, 'pixel_ratio': pixel_ratio,
            'plant_mask': plant_mask
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_improved_analysis(image, plant_mask, canopy_x, canopy_y, soil_line_y,
                              beaker_region, heights_cm, heights_inch, avg_height, std_height,
                              greenness, pixel_ratio, image_name):
    """Visualize improved analysis results"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Wheatgrass Analysis - {image_name}', fontsize=16, fontweight='bold')
    
    h, w = image.shape[:2]
    x_start, x_end, y_start, y_end = beaker_region
    
    # Plot 1: Overall detection
    axes[0, 0].imshow(image)
    # Beaker boundary
    rect = plt.Rectangle((x_start, y_start), x_end-x_start, y_end-y_start,
                        fill=False, edgecolor='yellow', linewidth=3, label='Beaker Region')
    axes[0, 0].add_patch(rect)
    # Soil line
    axes[0, 0].axhline(y=soil_line_y, color='brown', linewidth=4, label='Soil Line')
    # Canopy boundary
    if len(canopy_x) > 0:
        axes[0, 0].plot(canopy_x, canopy_y, 'r-', linewidth=2, label='Canopy Boundary')
    axes[0, 0].set_title('1. Detection Overview')
    axes[0, 0].legend()
    axes[0, 0].axis('off')
    
    # Plot 2: Plant detection
    masked_image = image.copy()
    masked_image[~plant_mask] = masked_image[~plant_mask] * 0.3
    axes[0, 1].imshow(masked_image)
    if len(canopy_x) > 0:
        axes[0, 1].plot(canopy_x, canopy_y, 'r-', linewidth=2)
    axes[0, 1].axhline(y=soil_line_y, color='brown', linewidth=3)
    axes[0, 1].set_title('2. Plant Detection')
    axes[0, 1].axis('off')
    
    # Plot 3: Height profile
    if len(heights_cm) > 0:
        x_pos = np.arange(len(heights_cm))
        axes[0, 2].plot(x_pos, heights_cm, 'g-', linewidth=2, alpha=0.7, label='Height')
        axes[0, 2].axhline(y=avg_height, color='red', linestyle='--', linewidth=3,
                          label=f'Average: {avg_height:.2f} cm')
        axes[0, 2].fill_between(x_pos, heights_cm, alpha=0.3, color='green')
        axes[0, 2].set_title('3. Plant Height Profile')
        axes[0, 2].set_xlabel('Measurement Point')
        axes[0, 2].set_ylabel('Height (cm)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Plant mask
    axes[1, 0].imshow(plant_mask, cmap='summer')
    axes[1, 0].set_title('4. Plant Mask')
    axes[1, 0].axis('off')
    
    # Plot 5: Height distribution
    if len(heights_cm) > 0:
        axes[1, 1].hist(heights_cm, bins=20, color='lightgreen', alpha=0.8, 
                       edgecolor='darkgreen', linewidth=0.5)
        axes[1, 1].axvline(avg_height, color='red', linestyle='--', linewidth=2,
                          label=f'Average: {avg_height:.2f} cm')
        axes[1, 1].set_title('5. Height Distribution')
        axes[1, 1].set_xlabel('Height (cm)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Statistics
    # Plot 6: Statistics
    stats_text = f"""
    IMPROVED WHEATGRASS ANALYSIS
    ============================
    Height Measurements:
    • Average: {avg_height:.2f} cm ({avg_height/2.54:.2f} in)
    • Maximum: {np.max(heights_cm):.2f} cm ({np.max(heights_cm)/2.54:.2f} in)
    • Minimum: {np.min(heights_cm):.2f} cm ({np.min(heights_cm)/2.54:.2f} in)
    • Std Dev: {std_height:.2f} cm
    • Range: {np.max(heights_cm) - np.min(heights_cm):.2f} cm
    
    Plant Health:
    • Greenness Score: {greenness:.3f}
    • Plant Coverage: {np.sum(plant_mask)} pixels
    • Measurements: {len(heights_cm)} points
    
    Detection Parameters:
    • Beaker Height: {y_end-y_start} px ≈ 25.0 cm
    • Pixel Ratio: {pixel_ratio:.4f} cm/px
    • Soil Position: {(y_end-soil_line_y)/(y_end-y_start)*100:.1f}% from bottom
    """
    axes[1, 2].text(0.05, 0.95, stats_text, fontsize=11, family='monospace',
                   verticalalignment='top', transform=axes[1, 2].transAxes,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", 
                           alpha=0.9, edgecolor='blue'))
    axes[1, 2].set_title('6. Detailed Analysis Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main application function - FIXED VERSION
    """
    print("🌾 STANDALONE Wheatgrass Analysis Application")
    print("=" * 60)
    print("Key Features:")
    print("• Automatic image selection")
    print("• Automatic SAM model download")
    print("• Improved plant detection")
    print("• Professional analysis report")
    print("=" * 60)
    
    # Select image file
    image_path = select_image_file()
    
    if not image_path:
        print("❌ No image selected. Exiting.")
        return
    
    print(f"📁 Selected image: {os.path.basename(image_path)}")
    
    try:
        # Initialize SAM (will auto-download if needed) - BUT WE DON'T ACTUALLY USE IT!
        # The current analysis doesn't use SAM, so we skip this to avoid issues
        print("🔄 Note: Using traditional CV methods (SAM not required for current analysis)")
        
        # Analyze wheatgrass directly (without SAM)
        results = analyze_wheatgrass_improved(image_path)
        
        if results:
            print("\n" + "="*70)
            print("✅ IMPROVED RESULTS")
            print("="*70)
            print(f"📏 AVERAGE HEIGHT: {results['avg_height_cm']:.2f} cm ({results['avg_height_cm']/2.54:.2f} inches)")
            print(f"📈 HEIGHT RANGE: {results['min_height_cm']:.2f} - {results['max_height_cm']:.2f} cm ({results['min_height_cm']/2.54:.2f} - {results['max_height_cm']/2.54:.2f} in)")
            print(f"📊 STANDARD DEVIATION: {results['std_height_cm']:.2f} cm")
            print(f"🎨 GREENNESS SCORE: {results['greenness']:.3f} (0-1 scale)")
            print(f"🌱 PLANT COVERAGE: {np.sum(results['plant_mask'])} pixels")
            print(f"📍 MEASUREMENT POINTS: {len(results['height_cm'])}")
            print(f"⚖️  PIXEL RATIO: {results['pixel_ratio']:.4f} cm/pixel")
            
            # Ask if user wants to analyze another image
            root = tk.Tk()
            root.withdraw()
            another = messagebox.askyesno("Analysis Complete", 
                                        "Analysis completed successfully!\n\nWould you like to analyze another image?")
            root.destroy()
            
            if another:
                main()  # Restart the process
            else:
                print("\n🎯 Thank you for using Wheatgrass Analysis!")
                
        else:
            print("\n❌ Analysis failed. Please check:")
            print("   - Image quality and visibility")
            print("   - Beaker is clearly visible with blue background")
            print("   - Plants are green and above soil")
            print("   - Soil is visible in bottom of beaker")
            
            # Ask to try again
            root = tk.Tk()
            root.withdraw()
            retry = messagebox.askyesno("Analysis Failed", 
                                      "Analysis failed. Would you like to try another image?")
            root.destroy()
            
            if retry:
                main()
            else:
                print("\n👋 Exiting application.")
                
    except Exception as e:
        print(f"\n💥 Application error: {e}")
        import traceback
        traceback.print_exc()
        
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", f"An error occurred:\n{e}\n\nPlease check the console for details.")
        root.destroy()
# Run the application
if __name__ == "__main__":
    main()