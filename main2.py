# main.py - Enhanced Wheatgrass Analysis with Improved Detection
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
    root.withdraw()
    
    file_types = [
        ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
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

def manual_select_beaker_region(image):
    """
    Allow user to manually select beaker region
    """
    print("\n🎯 MANUAL BEAKER SELECTION")
    print("Select the ENTIRE beaker (from top rim to bottom)")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.set_title('Manual Beaker Selection\nSelect entire rectangular beaker\nPress ENTER when done', 
                 fontsize=14, fontweight='bold')
    
    rect = None
    start_point = None
    rect_coords = {'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0}
    
    def on_press(event):
        nonlocal start_point, rect
        if event.button == 1:
            start_point = (event.xdata, event.ydata)
            if rect:
                rect.remove()
            rect = plt.Rectangle(start_point, 0, 0, fill=False, 
                                edgecolor='red', linewidth=3, linestyle='--')
            ax.add_patch(rect)
            fig.canvas.draw()
    
    def on_motion(event):
        nonlocal start_point, rect
        if start_point is not None and rect is not None:
            x1, y1 = start_point
            x2, y2 = event.xdata, event.ydata
            rect.set_width(x2 - x1)
            rect.set_height(y2 - y1)
            rect.set_xy((x1, y1))
            fig.canvas.draw()
    
    def on_release(event):
        nonlocal start_point, rect_coords
        if start_point is not None and rect is not None:
            x1, y1 = start_point
            x2, y2 = event.xdata, event.ydata
            rect_coords['x1'] = min(x1, x2)
            rect_coords['y1'] = min(y1, y2)
            rect_coords['x2'] = max(x1, x2)
            rect_coords['y2'] = max(y1, y2)
    
    def on_key(event):
        if event.key == 'enter':
            plt.close(fig)
        elif event.key == 'escape':
            plt.close(fig)
            raise KeyboardInterrupt("User cancelled")
    
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.show()
    
    x1, y1, x2, y2 = rect_coords['x1'], rect_coords['y1'], rect_coords['x2'], rect_coords['y2']
    
    # Validate
    if x2 - x1 < 50 or y2 - y1 < 50:
        print("⚠️ Selection too small")
        return manual_select_beaker_region(image)
    
    h, w = image.shape[:2]
    x1, x2 = int(max(0, min(x1, w))), int(max(0, min(x2, w)))
    y1, y2 = int(max(0, min(y1, h))), int(max(0, min(y2, h)))
    
    beaker_mask = np.zeros((h, w), dtype=bool)
    beaker_mask[y1:y2, x1:x2] = True
    
    print(f"✅ Manual selection: {x2-x1}x{y2-y1} px")
    
    return x1, x2, y1, y2, beaker_mask

def detect_beaker_robust(image, use_manual=False):
    """
    Robust beaker detection
    """
    if use_manual:
        return manual_select_beaker_region(image)
    
    print("\n🔍 AUTOMATIC BEAKER DETECTION")
    h, w = image.shape[:2]
    
    # Strategy 1: Edge-based with Hough lines
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray)
    
    # Edge detection
    edges = cv2.Canny(gray_enhanced, 50, 150)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find vertical lines (beaker sides)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                           minLineLength=h//3, maxLineGap=20)
    
    vertical_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if line is mostly vertical
            if abs(x2 - x1) < 15 and abs(y2 - y1) > h//3:
                vertical_lines.append((min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)))
    
    # Strategy 2: Color-based background detection
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Common background colors
    bg_masks = []
    
    # Blue background
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    bg_masks.append(blue_mask)
    
    # Pink/peach background
    lower_pink = np.array([0, 15, 140])
    upper_pink = np.array([30, 90, 255])
    pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
    bg_masks.append(pink_mask)
    
    # Combine background masks
    bg_combined = np.zeros_like(blue_mask)
    for mask in bg_masks:
        bg_combined = cv2.bitwise_or(bg_combined, mask)
    
    # Clean background mask
    bg_combined = cv2.morphologyEx(bg_combined, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8))
    
    # Find contours in background (should be beaker outline)
    contours, _ = cv2.findContours(bg_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5000 or area > w*h*0.7:
            continue
        
        x, y, rect_w, rect_h = cv2.boundingRect(contour)
        
        # Check aspect ratio (beaker is tall rectangle)
        aspect_ratio = rect_h / rect_w
        if aspect_ratio < 0.7 or aspect_ratio > 3.5:
            continue
        
        if rect_h < h * 0.4:
            continue
        
        # Check rectangularity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            rectangularity = area / hull_area
            if rectangularity < 0.6:
                continue
        
        valid_contours.append((contour, x, y, rect_w, rect_h, area, rectangularity))
    
    # Use best detection method
    best_region = None
    best_score = 0
    
    # Method 1: Use vertical lines
    if vertical_lines:
        x_min = min([line[0] for line in vertical_lines])
        x_max = max([line[1] for line in vertical_lines])
        y_min = min([line[2] for line in vertical_lines])
        y_max = max([line[3] for line in vertical_lines])
        
        width = x_max - x_min
        height = y_max - y_min
        
        if width > 50 and height > 100:
            score = height / h
            if score > best_score:
                best_region = (x_min, x_max, y_min, y_max)
                best_score = score
    
    # Method 2: Use best contour
    if valid_contours:
        valid_contours.sort(key=lambda x: (x[6], x[5]), reverse=True)
        contour, x, y, rect_w, rect_h, area, rectangularity = valid_contours[0]
        
        score = rectangularity * (rect_h / h)
        if score > best_score:
            best_region = (x, x+rect_w, y, y+rect_h)
            best_score = score
    
    # If we found a region
    if best_region:
        x_start, x_end, y_start, y_end = best_region
        
        # Add small padding
        padding_x = int((x_end - x_start) * 0.03)
        padding_y = int((y_end - y_start) * 0.03)
        
        x_start = max(0, x_start - padding_x)
        x_end = min(w, x_end + padding_x)
        y_start = max(0, y_start - padding_y)
        y_end = min(h, y_end + padding_y)
        
        # Create mask
        beaker_mask = np.zeros((h, w), dtype=bool)
        beaker_mask[y_start:y_end, x_start:x_end] = True
        
        print(f"✅ Automatic detection: {x_end-x_start}x{y_end-y_start} px")
        
        return x_start, x_end, y_start, y_end, beaker_mask
    
    # Fallback
    print("⚠️ Automatic detection inconclusive")
    root = tk.Tk()
    root.withdraw()
    use_manual = messagebox.askyesno("Detection Failed",
                                    "Automatic beaker detection was not confident.\n\n"
                                    "Would you like to select the beaker manually?")
    root.destroy()
    
    if use_manual:
        return manual_select_beaker_region(image)
    else:
        return detect_beaker_center_fallback(image)

def detect_beaker_center_fallback(image):
    """
    Fallback beaker detection
    """
    h, w = image.shape[:2]
    print("⚠️ Using center-based fallback")
    
    width_ratio = 0.6
    height_ratio = 0.8
    
    x_start = int(w * (1 - width_ratio) / 2)
    x_end = int(w - x_start)
    y_start = int(h * (1 - height_ratio) / 2)
    y_end = int(h - y_start)
    
    beaker_mask = np.zeros((h, w), dtype=bool)
    beaker_mask[y_start:y_end, x_start:x_end] = True
    
    print(f"Fallback region: {x_end-x_start}x{y_end-y_start} px")
    
    return x_start, x_end, y_start, y_end, beaker_mask

def detect_soil_line_accurate(image, beaker_region):
    """
    IMPROVED soil line detection - always works
    """
    print("\n🌱 DETECTING SOIL LINE")
    
    x_start, x_end, y_start, y_end = beaker_region
    h, w = image.shape[:2]
    
    beaker_height = y_end - y_start
    
    # Soil is typically in bottom 20-35% of beaker
    soil_search_height = int(beaker_height * 0.35)
    soil_region_start = max(y_start, y_end - soil_search_height)
    
    print(f"Searching for soil in bottom {soil_search_height} px")
    
    # Extract soil region
    soil_region = image[soil_region_start:y_end, x_start:x_end]
    
    if soil_region.size == 0:
        print("⚠️ No soil region found, using default")
        default_soil_y = y_end - int(beaker_height * 0.15)
        pixel_ratio = 18.034 / beaker_height
        return default_soil_y, pixel_ratio
    
    # MULTI-METHOD SOIL DETECTION
    hsv = cv2.cvtColor(soil_region, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(soil_region, cv2.COLOR_RGB2LAB)
    
    # Method 1: Dark colors (soil is dark)
    # V (value) channel in HSV - soil has low value
    v_channel = hsv[:, :, 2]
    v_mask = v_channel < 100  # Dark pixels
    
    # Method 2: Brown colors in HSV
    lower_brown1 = np.array([0, 20, 10])   # Dark brown
    upper_brown1 = np.array([30, 200, 120])
    
    lower_brown2 = np.array([0, 0, 0])     # Very dark/black
    upper_brown2 = np.array([180, 100, 80])
    
    # Method 3: Dark areas in LAB space
    l_channel = lab[:, :, 0]
    lab_mask = l_channel < 100  # Dark in LAB
    
    # Create combined mask
    hsv_mask1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
    hsv_mask2 = cv2.inRange(hsv, lower_brown2, upper_brown2)
    
    # Combine all methods
    soil_mask = np.zeros_like(v_mask, dtype=np.uint8)
    soil_mask[v_mask] = 255
    soil_mask = cv2.bitwise_or(soil_mask, hsv_mask1)
    soil_mask = cv2.bitwise_or(soil_mask, hsv_mask2)
    soil_mask[lab_mask] = 255
    
    # Clean up mask
    kernel = np.ones((7, 7), np.uint8)
    soil_mask = cv2.morphologyEx(soil_mask, cv2.MORPH_CLOSE, kernel)
    soil_mask = cv2.morphologyEx(soil_mask, cv2.MORPH_OPEN, kernel)
    
    # Fill holes
    soil_mask = cv2.morphologyEx(soil_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    
    # Calculate soil coverage per row
    row_sums = np.sum(soil_mask, axis=1)
    
    if len(row_sums) == 0:
        print("⚠️ No soil detected in mask")
        soil_line_y = y_end - int(beaker_height * 0.15)
    else:
        # Find soil line using RELIABLE method
        # Look for consistent soil coverage from bottom up
        soil_top_in_region = None
        
        # Method A: Look for where soil coverage becomes consistent
        avg_soil_coverage = np.mean(row_sums[-10:]) if len(row_sums) >= 10 else np.mean(row_sums)
        threshold = avg_soil_coverage * 0.7
        
        # Scan from bottom up
        for i in range(len(row_sums)-1, -1, -1):
            if row_sums[i] >= threshold:
                # Check if next few rows also have soil
                check_ahead = min(5, len(row_sums) - i - 1)
                if check_ahead > 0:
                    ahead_coverage = np.mean(row_sums[i:i+check_ahead])
                    if ahead_coverage >= threshold * 0.8:
                        soil_top_in_region = i
                        break
            elif i < len(row_sums) - 10:
                # If we've been scanning and found nothing, use alternative
                break
        
        # Method B: If no consistent soil found, use gradient method
        if soil_top_in_region is None:
            # Find where soil density increases (going upward)
            smoothed = np.convolve(row_sums, np.ones(5)/5, mode='same')
            gradient = np.gradient(smoothed)
            
            # Look for significant positive gradient (transition to more soil)
            for i in range(len(gradient)-1, 0, -1):
                if gradient[i] > np.std(gradient) * 0.5 and row_sums[i] > soil_region.shape[1] * 0.2:
                    soil_top_in_region = i
                    break
        
        # Method C: Last resort - use maximum
        if soil_top_in_region is None:
            soil_top_in_region = np.argmax(row_sums)
            print(f"⚠️ Using maximum soil row: {soil_top_in_region}")
        
        soil_line_y = soil_region_start + soil_top_in_region
    
    # VALIDATE soil line position
    # Soil must be in bottom portion of beaker
    min_soil_y = y_start + int(beaker_height * 0.65)  # Bottom 35%
    max_soil_y = y_end - 10  # Leave margin
    
    soil_line_y = max(min_soil_y, min(soil_line_y, max_soil_y))
    
    # Calculate pixel ratio
    pixel_to_cm_ratio = 18.034 / beaker_height
    
    print(f"✅ Soil line: y={soil_line_y}")
    print(f"   From bottom: {y_end-soil_line_y} px")
    print(f"   Beaker: {beaker_height} px = 18.034 cm")
    print(f"   Ratio: {pixel_to_cm_ratio:.4f} cm/px")
    
    return soil_line_y, pixel_to_cm_ratio

def detect_plants_enhanced(image, beaker_region, soil_line_y):
    """
    ENHANCED plant detection for wheatgrass (green to brown)
    """
    print("\n🌿 ENHANCED PLANT DETECTION")
    
    x_start, x_end, y_start, y_end = beaker_region
    h, w = image.shape[:2]
    
    # IMPORTANT: Allow plants to grow above beaker
    # Start search from slightly above beaker to capture tall plants
    plant_top = max(0, y_start - 50)  # Allow 50px above beaker for tall plants
    plant_bottom = soil_line_y
    
    plant_height = plant_bottom - plant_top
    plant_width = x_end - x_start
    
    print(f"Plant search region: {plant_width}x{plant_height} px")
    print(f"From y={plant_top} to y={plant_bottom}")
    
    # Extract plant region (including area above beaker)
    plant_region = image[plant_top:plant_bottom, x_start:x_end]
    
    if plant_region.size == 0:
        print("⚠️ No plant region found")
        return np.zeros((h, w), dtype=bool)
    
    # Convert to multiple color spaces
    hsv = cv2.cvtColor(plant_region, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(plant_region, cv2.COLOR_RGB2LAB)
    
    # EXTENSIVE WHEATGRASS COLOR DETECTION
    
    # Healthy green wheatgrass
    lower_green1 = np.array([35, 40, 40])
    upper_green1 = np.array([85, 255, 220])
    
    # Dark green mature
    lower_green2 = np.array([25, 30, 20])
    upper_green2 = np.array([95, 255, 180])
    
    # Yellowish (maturing)
    lower_yellow = np.array([20, 40, 50])
    upper_yellow = np.array([35, 200, 200])
    
    # Brown (drying/mature tops)
    lower_brown1 = np.array([5, 30, 30])
    upper_brown1 = np.array([20, 150, 150])
    
    # Dark brown
    lower_brown2 = np.array([10, 20, 20])
    upper_brown2 = np.array([25, 100, 100])
    
    # Extended green range
    lower_green3 = np.array([15, 20, 20])
    upper_green3 = np.array([100, 220, 200])
    
    # LAB space for green
    lower_lab_green = np.array([0, 120, 120])
    upper_lab_green = np.array([255, 150, 150])
    
    # Create masks for all color ranges
    masks = []
    masks.append(cv2.inRange(hsv, lower_green1, upper_green1))
    masks.append(cv2.inRange(hsv, lower_green2, upper_green2))
    masks.append(cv2.inRange(hsv, lower_yellow, upper_yellow))
    masks.append(cv2.inRange(hsv, lower_brown1, upper_brown1))
    masks.append(cv2.inRange(hsv, lower_brown2, upper_brown2))
    masks.append(cv2.inRange(hsv, lower_green3, upper_green3))
    masks.append(cv2.inRange(lab, lower_lab_green, upper_lab_green))
    
    # Combine all plant masks
    plant_mask = masks[0]
    for mask in masks[1:]:
        plant_mask = cv2.bitwise_or(plant_mask, mask)
    
    # REMOVE BACKGROUND COLORS (critical for accuracy)
    
    # Pink/peach background
    lower_pink = np.array([0, 15, 140])
    upper_pink = np.array([30, 80, 255])
    pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
    plant_mask = cv2.bitwise_and(plant_mask, cv2.bitwise_not(pink_mask))
    
    # Blue background
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    plant_mask = cv2.bitwise_and(plant_mask, cv2.bitwise_not(blue_mask))
    
    # Light background (white/light colors)
    lower_light = np.array([0, 0, 200])
    upper_light = np.array([180, 30, 255])
    light_mask = cv2.inRange(hsv, lower_light, upper_light)
    plant_mask = cv2.bitwise_and(plant_mask, cv2.bitwise_not(light_mask))
    
    # ENHANCED POST-PROCESSING
    
    # 1. Initial cleaning
    kernel_open = np.ones((2, 2), np.uint8)
    kernel_close = np.ones((3, 3), np.uint8)
    
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel_open)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # 2. Remove very small components
    plant_mask = remove_small_components_enhanced(plant_mask, min_size=20)
    
    # 3. Enhance plant structures (wheatgrass grows vertically)
    vertical_kernel = np.ones((7, 1), np.uint8)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, vertical_kernel)
    
    # 4. Fill small holes within plants
    plant_mask = fill_small_holes(plant_mask, max_hole_size=50)
    
    # 5. Focus on vertical structures (wheatgrass characteristics)
    plant_mask = keep_vertical_structures(plant_mask, min_aspect_ratio=1.2, min_height=15)
    
    # Create full image mask
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[plant_top:plant_bottom, x_start:x_end] = plant_mask
    
    # Also check for plants above the beaker (tall wheatgrass)
    if plant_top < y_start:
        above_beaker_region = image[plant_top:y_start, x_start:x_end]
        if above_beaker_region.size > 0:
            above_hsv = cv2.cvtColor(above_beaker_region, cv2.COLOR_RGB2HSV)
            
            # Check for plant colors above beaker
            above_mask1 = cv2.inRange(above_hsv, lower_green1, upper_green1)
            above_mask2 = cv2.inRange(above_hsv, lower_green2, upper_green2)
            above_mask3 = cv2.inRange(above_hsv, lower_brown1, upper_brown1)
            above_mask = above_mask1 | above_mask2 | above_mask3
            
            # Clean up above-beaker mask
            above_mask = cv2.morphologyEx(above_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
            above_mask = remove_small_components_enhanced(above_mask, min_size=10)
            
            # Add to full mask
            full_mask[plant_top:y_start, x_start:x_end] = above_mask
    
    plant_count = np.sum(full_mask > 0)
    print(f"✅ Plants detected: {plant_count} pixels")
    
    return full_mask.astype(bool)

def remove_small_components_enhanced(mask, min_size=20):
    """
    Remove small components with aspect ratio consideration
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    cleaned_mask = np.zeros_like(mask)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        if width > 0:
            aspect_ratio = height / width
        else:
            aspect_ratio = 0
        
        # Keep if:
        # 1. Large enough area, OR
        # 2. Tall and thin (likely wheatgrass)
        if (area >= min_size) or (aspect_ratio > 1.5 and area >= min_size//2):
            cleaned_mask[labels == i] = 255
    
    return cleaned_mask

def keep_vertical_structures(mask, min_aspect_ratio=1.2, min_height=15):
    """
    Keep only components that are vertical
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    vertical_mask = np.zeros_like(mask)
    
    for i in range(1, num_labels):
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        if width > 0:
            aspect_ratio = height / width
        else:
            aspect_ratio = 0
        
        # Keep vertical structures
        if aspect_ratio >= min_aspect_ratio and height >= min_height:
            vertical_mask[labels == i] = 255
        # Also keep if area is large (dense plant area)
        elif stats[i, cv2.CC_STAT_AREA] > 200:
            vertical_mask[labels == i] = 255
    
    return vertical_mask

def fill_small_holes(mask, max_hole_size=50):
    """
    Fill small holes in binary mask
    """
    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    filled_mask = mask.copy()
    
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, (cnt, hier) in enumerate(zip(contours, hierarchy)):
            if hier[3] >= 0:  # Hole contour
                area = cv2.contourArea(cnt)
                if area <= max_hole_size:
                    cv2.drawContours(filled_mask, [cnt], 0, 255, -1)
    
    return filled_mask

def extract_canopy_boundary_enhanced(plant_mask, soil_line_y, beaker_region):
    """
    ENHANCED canopy boundary extraction - stays within beaker width
    """
    print("\n📐 ENHANCED CANOPY BOUNDARY EXTRACTION")
    
    h, w = plant_mask.shape
    x_start, x_end, y_start, y_end = beaker_region
    
    # Create working mask (plants above soil only)
    working_mask = plant_mask.copy()
    working_mask[soil_line_y:, :] = False
    
    # CRITICAL: Constrain search to beaker width
    # Create a mask that's True only within beaker width
    width_mask = np.zeros_like(working_mask, dtype=bool)
    width_mask[:, x_start:x_end] = True
    working_mask = np.logical_and(working_mask, width_mask)
    
    # Strategy 1: Dense sampling within beaker width
    boundary_points = []
    
    # Sample densely within beaker
    sampling_step = max(1, (x_end - x_start) // 150)  # More dense sampling
    if sampling_step < 1:
        sampling_step = 1
    
    for x in range(x_start, x_end, sampling_step):
        column = working_mask[:, x]
        if np.any(column):
            plant_pixels = np.where(column)[0]
            if len(plant_pixels) > 0:
                # Find the top of plants in this column
                # Look for clusters to avoid outliers
                if len(plant_pixels) > 5:
                    # Find gaps between plant clusters
                    gaps = np.diff(plant_pixels)
                    large_gaps = np.where(gaps > 10)[0]
                    
                    if len(large_gaps) > 0:
                        # Multiple plant groups, take the highest one
                        top_group_end = large_gaps[0]
                        if top_group_end < len(plant_pixels):
                            top_y = plant_pixels[0]  # Top of first group
                        else:
                            top_y = plant_pixels[0]
                    else:
                        # Continuous plant, take the top
                        top_y = plant_pixels[0]
                else:
                    top_y = plant_pixels[0]
                
                boundary_points.append((x, top_y))
    
    # Strategy 2: If too few points, use horizontal projection
    if len(boundary_points) < 10:
        print(f"⚠️ Only {len(boundary_points)} points, using horizontal projection")
        
        # Calculate horizontal projection (sum of plant pixels per row)
        horizontal_proj = np.sum(working_mask, axis=1)
        
        # Find rows with significant plant presence
        if np.max(horizontal_proj) > 0:
            threshold = np.max(horizontal_proj) * 0.1
            plant_rows = np.where(horizontal_proj > threshold)[0]
            
            if len(plant_rows) > 0:
                top_row = np.min(plant_rows)
                
                # Create boundary at this row within beaker
                for x in range(x_start, x_end, sampling_step*2):
                    if working_mask[top_row, x]:
                        boundary_points.append((x, top_row))
    
    if len(boundary_points) < 5:
        print(f"⚠️ Only {len(boundary_points)} boundary points found")
        return np.array([]), np.array([])
    
    # Convert to arrays
    x_vals = np.array([p[0] for p in boundary_points])
    y_vals = np.array([p[1] for p in boundary_points])
    
    # Sort by x
    sorted_idx = np.argsort(x_vals)
    x_vals = x_vals[sorted_idx]
    y_vals = y_vals[sorted_idx]
    
    # Remove outliers
    if len(y_vals) > 10:
        # Use IQR method to remove outliers
        q1 = np.percentile(y_vals, 25)
        q3 = np.percentile(y_vals, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        inlier_mask = (y_vals >= lower_bound) & (y_vals <= upper_bound)
        x_vals = x_vals[inlier_mask]
        y_vals = y_vals[inlier_mask]
    
    # Smooth the boundary
    if len(y_vals) > 10:
        window_size = min(15, len(y_vals))
        if window_size % 2 == 0:
            window_size -= 1
        if window_size >= 5:
            try:
                y_vals = savgol_filter(y_vals, window_size, 3)
            except:
                pass
    
    # CRITICAL: Ensure boundary stays within beaker width
    # Create a continuous boundary across beaker width
    if len(x_vals) > 10:
        # Create dense x coordinates within beaker
        x_dense = np.arange(x_start, x_end, 5)
        
        if len(x_vals) > 1:
            # Interpolate y values
            y_interp = np.interp(x_dense, x_vals, y_vals)
            
            # Apply additional smoothing
            if len(y_interp) > 10:
                y_interp = np.convolve(y_interp, np.ones(5)/5, mode='same')
            
            x_vals = x_dense
            y_vals = y_interp
    
    # Ensure y values are reasonable
    min_valid_y = max(0, np.min(y_vals) - 30)  # Allow some variation
    max_valid_y = min(h, soil_line_y + 20)  # Should be above soil
    
    valid_mask = (y_vals >= min_valid_y) & (y_vals <= max_valid_y)
    x_vals = x_vals[valid_mask]
    y_vals = y_vals[valid_mask]
    
    print(f"✅ Canopy boundary: {len(x_vals)} points within beaker")
    
    return x_vals, y_vals

def calculate_plant_health_enhanced(image, plant_mask):
    """
    Enhanced plant health calculation
    """
    if np.sum(plant_mask) == 0:
        return 0.0, 0.0, 0.0
    
    plant_pixels = image[plant_mask]
    
    # Convert to HSV
    hsv = cv2.cvtColor(plant_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    hue = hsv[:, 0]
    saturation = hsv[:, 1]
    value = hsv[:, 2]
    
    # Enhanced health categories
    healthy_mask = (hue >= 35) & (hue <= 85) & (saturation >= 40) & (value >= 40)
    mature_mask = ((hue >= 25) & (hue < 35)) | ((hue > 85) & (hue <= 95)) & (saturation >= 30)
    drying_mask = ((hue >= 10) & (hue < 25)) | ((hue > 95) & (hue <= 110)) & (saturation >= 20)
    
    # Calculate percentages
    total_pixels = len(hue)
    healthy_pct = np.sum(healthy_mask) / total_pixels if total_pixels > 0 else 0
    mature_pct = np.sum(mature_mask) / total_pixels if total_pixels > 0 else 0
    drying_pct = np.sum(drying_mask) / total_pixels if total_pixels > 0 else 0
    
    # Enhanced scoring
    health_score = (healthy_pct * 0.7 + mature_pct * 0.3 + drying_pct * 0.1)
    health_score = max(0, min(1, health_score))
    
    greenness_score = np.mean((hue >= 30) & (hue <= 90)) if np.any(hue) else 0
    colorfulness_score = np.mean(saturation) / 255.0 if np.any(saturation) else 0
    
    return health_score, greenness_score, colorfulness_score

def analyze_wheatgrass_enhanced(image_path):
    """
    Main enhanced analysis function
    """
    print("=" * 70)
    print("🌾 ENHANCED WHEATGRASS ANALYSIS")
    print("=" * 70)
    
    try:
        start_time = time.time()
        
        # Load image
        image = load_wheat_image(image_path)
        
        # Ask for detection method
        root = tk.Tk()
        root.withdraw()
        detection_choice = messagebox.askyesno("Detection Method",
                                             "Select beaker detection method:\n\n"
                                             "YES: Automatic detection (recommended)\n"
                                             "NO: Manual selection")
        root.destroy()
        
        # Detect beaker
        beaker_region = detect_beaker_robust(image, use_manual=not detection_choice)
        x_start, x_end, y_start, y_end, beaker_mask = beaker_region
        
        # Detect soil line (IMPROVED - always works)
        soil_line_y, pixel_ratio = detect_soil_line_accurate(image, (x_start, x_end, y_start, y_end))
        
        # Detect plants (ENHANCED - detects green to brown plants)
        plant_mask = detect_plants_enhanced(image, (x_start, x_end, y_start, y_end), soil_line_y)
        
        if np.sum(plant_mask) == 0:
            print("❌ No plants detected!")
            return None
        
        # Extract canopy boundary (ENHANCED - stays within beaker width)
        canopy_x, canopy_y = extract_canopy_boundary_enhanced(plant_mask, soil_line_y, (x_start, x_end, y_start, y_end))
        
        if len(canopy_x) == 0:
            print("❌ No canopy boundary extracted!")
            return None
        
        # Calculate heights
        heights_pixels = soil_line_y - canopy_y
        heights_cm = heights_pixels * pixel_ratio
        heights_inch = heights_cm / 2.54
        
        # Filter valid heights (0.5 to 50 cm)
        valid_mask = (heights_cm > 0.5) & (heights_cm < 50)
        heights_cm = heights_cm[valid_mask]
        heights_inch = heights_inch[valid_mask]
        canopy_x = canopy_x[valid_mask]
        canopy_y = canopy_y[valid_mask]
        
        if len(heights_cm) == 0:
            print("❌ No valid heights calculated!")
            return None
        
        # Statistics
        avg_height_cm = np.mean(heights_cm)
        max_height_cm = np.max(heights_cm)
        min_height_cm = np.min(heights_cm)
        std_height_cm = np.std(heights_cm)
        median_height_cm = np.median(heights_cm)
        avg_height_inch = avg_height_cm / 2.54
        
        # Health scores
        health_score, greenness_score, colorfulness_score = calculate_plant_health_enhanced(image, plant_mask)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ Analysis completed in {elapsed_time:.1f} seconds")
        
        # Prepare results
        results = {
            'canopy_x': canopy_x, 'canopy_y': canopy_y,
            'soil_line_y': soil_line_y, 'height_cm': heights_cm,
            'height_inch': heights_inch, 'avg_height_cm': avg_height_cm,
            'avg_height_inch': avg_height_inch, 'max_height_cm': max_height_cm,
            'min_height_cm': min_height_cm, 'std_height_cm': std_height_cm,
            'median_height_cm': median_height_cm,
            'health_score': health_score, 'greenness_score': greenness_score,
            'colorfulness_score': colorfulness_score,
            'pixel_ratio': pixel_ratio, 'plant_mask': plant_mask,
            'beaker_region': (x_start, x_end, y_start, y_end)
        }
        
        # Display results
        print(f"\n📊 ENHANCED RESULTS:")
        print(f"  Height: {avg_height_cm:.2f} cm ({avg_height_inch:.2f} in)")
        print(f"  Range: {min_height_cm:.2f} to {max_height_cm:.2f} cm")
        print(f"  Std Dev: {std_height_cm:.2f} cm")
        print(f"  Health Score: {health_score:.3f}/1.0")
        print(f"  Greenness: {greenness_score:.3f}/1.0")
        print(f"  Plants Detected: {np.sum(plant_mask):,} pixels")
        print(f"  Canopy Points: {len(canopy_x)}")
        
        # Create visualization
        create_enhanced_visualization(image, plant_mask, canopy_x, canopy_y, soil_line_y,
                                     (x_start, x_end, y_start, y_end), heights_cm, 
                                     results, os.path.basename(image_path))
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_enhanced_visualization(image, plant_mask, canopy_x, canopy_y, soil_line_y,
                                 beaker_region, heights_cm, results, image_name):
    """
    Create enhanced visualization
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    fig.suptitle(f'Enhanced Wheatgrass Analysis - {image_name}', 
                 fontsize=18, fontweight='bold')
    
    h, w = image.shape[:2]
    x_start, x_end, y_start, y_end = beaker_region
    
    # Plot 1: Enhanced detection overview
    axes[0, 0].imshow(image)
    # Beaker rectangle
    rect = plt.Rectangle((x_start, y_start), x_end-x_start, y_end-y_start,
                        fill=False, edgecolor='yellow', linewidth=3, label='Beaker')
    axes[0, 0].add_patch(rect)
    # Soil line
    axes[0, 0].axhline(y=soil_line_y, color='brown', linewidth=3, label='Soil Line')
    # Canopy line
    if len(canopy_x) > 0:
        axes[0, 0].plot(canopy_x, canopy_y, 'r-', linewidth=2, alpha=0.8, label='Canopy')
    axes[0, 0].set_title('1. Enhanced Detection Overview', fontsize=14)
    axes[0, 0].legend(loc='upper right', fontsize=9)
    axes[0, 0].axis('off')
    
    # Plot 2: Plant detection with health colors
    # Create color-coded plant mask
    health_image = np.zeros_like(image)
    plant_indices = np.where(plant_mask)
    
    if len(plant_indices[0]) > 0:
        # Get plant pixels
        plant_pixels = image[plant_mask]
        
        # Convert to HSV for health assessment
        hsv_pixels = cv2.cvtColor(plant_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        hue = hsv_pixels[:, 0]
        
        # Color code by health
        for i in range(len(plant_indices[0])):
            y, x = plant_indices[0][i], plant_indices[1][i]
            h = hue[i] if i < len(hue) else 0
            
            if 35 <= h <= 85:  # Healthy green
                health_image[y, x] = [0, 200, 0]
            elif 25 <= h < 35 or 85 < h <= 95:  # Mature yellow-green
                health_image[y, x] = [200, 200, 0]
            elif 10 <= h < 25 or 95 < h <= 110:  # Drying brown
                health_image[y, x] = [139, 69, 19]
            else:  # Other
                health_image[y, x] = [100, 100, 100]
    
    # Blend with original image
    blended = cv2.addWeighted(image, 0.3, health_image, 0.7, 0)
    axes[0, 1].imshow(blended)
    axes[0, 1].axhline(y=soil_line_y, color='brown', linewidth=2)
    if len(canopy_x) > 0:
        axes[0, 1].plot(canopy_x, canopy_y, 'r-', linewidth=1.5, alpha=0.7)
    axes[0, 1].set_title('2. Plant Health Color Coding', fontsize=14)
    axes[0, 1].axis('off')
    
    # Plot 3: Height profile with confidence
    if len(heights_cm) > 0:
        x_pos = np.arange(len(heights_cm))
        axes[0, 2].plot(x_pos, heights_cm, 'g-', linewidth=2, alpha=0.8)
        axes[0, 2].fill_between(x_pos, 
                               heights_cm - results['std_height_cm']/2,
                               heights_cm + results['std_height_cm']/2,
                               alpha=0.2, color='green')
        axes[0, 2].axhline(y=results['avg_height_cm'], color='red', linestyle='--',
                          linewidth=2, label=f'Avg: {results["avg_height_cm"]:.2f} cm')
        axes[0, 2].axhline(y=results['median_height_cm'], color='blue', linestyle=':',
                          linewidth=2, label=f'Med: {results["median_height_cm"]:.2f} cm')
        axes[0, 2].set_title('3. Height Profile with Confidence', fontsize=14)
        axes[0, 2].set_xlabel('Position along canopy')
        axes[0, 2].set_ylabel('Height (cm)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Plant mask with individual plants
    axes[1, 0].imshow(plant_mask, cmap='Greens')
    axes[1, 0].axhline(y=soil_line_y, color='brown', linewidth=2)
    # Show beaker boundaries
    axes[1, 0].axvline(x=x_start, color='yellow', linewidth=1, linestyle='--', alpha=0.5)
    axes[1, 0].axvline(x=x_end, color='yellow', linewidth=1, linestyle='--', alpha=0.5)
    if len(canopy_x) > 0:
        axes[1, 0].plot(canopy_x, canopy_y, 'r-', linewidth=1.5, alpha=0.7)
    axes[1, 0].set_title('4. Plant Detection Mask (within beaker)', fontsize=14)
    axes[1, 0].axis('off')
    
    # Plot 5: Height distribution
    if len(heights_cm) > 0:
        n_bins = min(20, len(heights_cm) // 5)
        axes[1, 1].hist(heights_cm, bins=n_bins, color='lightgreen', alpha=0.8,
                       edgecolor='darkgreen', linewidth=0.5)
        axes[1, 1].axvline(results['avg_height_cm'], color='red', linestyle='--',
                          linewidth=2, label='Average')
        axes[1, 1].axvline(results['median_height_cm'], color='blue', linestyle=':',
                          linewidth=2, label='Median')
        axes[1, 1].set_title('5. Height Distribution', fontsize=14)
        axes[1, 1].set_xlabel('Height (cm)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Enhanced summary
    stats_text = f"""
    ENHANCED WHEATGRASS ANALYSIS
    ============================
    
    Beaker Dimensions:
    • Width: {x_end-x_start} px
    • Height: {y_end-y_start} px
    • Pixel Ratio: {results['pixel_ratio']:.4f} cm/px
    • Actual Height: 7.1 in (18.034 cm)
    
    Height Statistics:
    • Average: {results['avg_height_cm']:.2f} cm
    • Maximum: {results['max_height_cm']:.2f} cm
    • Minimum: {results['min_height_cm']:.2f} cm
    • Median: {results['median_height_cm']:.2f} cm
    • Std Dev: {results['std_height_cm']:.2f} cm
    • Range: {max(heights_cm)-min(heights_cm):.2f} cm
    
    Plant Health:
    • Overall Health: {results['health_score']:.3f}/1.0
    • Greenness: {results['greenness_score']:.3f}/1.0
    • Colorfulness: {results['colorfulness_score']:.3f}/1.0
    
    Detection Metrics:
    • Plant Pixels: {np.sum(plant_mask):,}
    • Canopy Points: {len(canopy_x)} (within beaker)
    • Soil Line: y={soil_line_y} px
    • Beaker Coverage: {100*(y_end-y_start)/h:.1f}% of image
    • Plants above beaker: {"Yes" if y_start > 0 else "No"}
    """
    
    axes[1, 2].text(0.05, 0.95, stats_text, fontsize=9, family='monospace',
                   verticalalignment='top', transform=axes[1, 2].transAxes,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", 
                           alpha=0.95, edgecolor='orange'))
    axes[1, 2].set_title('6. Enhanced Summary', fontsize=14)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main application function
    """
    print("\n" + "="*70)
    print("🌾 ENHANCED WHEATGRASS ANALYSIS SYSTEM")
    print("="*70)
    print("Key Improvements:")
    print("• Reliable soil line detection (always works)")
    print("• Enhanced plant detection (green to brown)")
    print("• Canopy line stays within beaker width")
    print("• Detects plants growing above beaker")
    print("• Improved accuracy for all plant colors")
    print("="*70)
    
    # Select image
    image_path = select_image_file()
    
    if not image_path:
        print("No image selected.")
        return
    
    print(f"\n📁 Selected: {os.path.basename(image_path)}")
    
    # Run enhanced analysis
    results = analyze_wheatgrass_enhanced(image_path)
    
    if results:
        # Ask for another analysis
        root = tk.Tk()
        root.withdraw()
        another = messagebox.askyesno("Analysis Complete",
                                    "Enhanced analysis completed!\n\n"
                                    "Analyze another image?")
        root.destroy()
        
        if another:
            main()
        else:
            print("\n✅ Thank you for using Enhanced Wheatgrass Analysis!")
    else:
        print("\n⚠️ Analysis failed. Please try a different image.")
        
        root = tk.Tk()
        root.withdraw()
        retry = messagebox.askyesno("Analysis Failed",
                                  "Analysis failed. Try another image?")
        root.destroy()
        
        if retry:
            main()

# Run the application
if __name__ == "__main__":
    main()