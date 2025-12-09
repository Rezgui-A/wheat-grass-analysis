# main.py - Enhanced Wheatgrass Analysis with Accurate Soil Detection
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
    print("Select the ENTIRE beaker (from top rim to bottom including soil)")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    ax.set_title('Manual Beaker Selection\nSelect entire rectangular beaker including soil at bottom\nPress ENTER when done', 
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
    
    # INCREASE BEAKER HEIGHT BY 10% to account for plants growing above
    original_height = y2 - y1
    height_increase = int(original_height * 0.10)
    y1 = max(0, y1 - height_increase)
    
    beaker_mask = np.zeros((h, w), dtype=bool)
    beaker_mask[y1:y2, x1:x2] = True
    
    print(f"✅ Manual selection: {x2-x1}x{y2-y1} px (increased height by 10%)")
    
    return x1, x2, y1, y2, beaker_mask

def manual_select_soil_line(image, beaker_region):
    """
    Allow user to manually select soil line
    """
    print("\n🎯 MANUAL SOIL LINE SELECTION")
    print("Click on the soil line (where soil ends and plants begin)")
    
    x_start, x_end, y_start, y_end = beaker_region
    h, w = image.shape[:2]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image)
    
    # Draw beaker rectangle
    rect = plt.Rectangle((x_start, y_start), x_end-x_start, y_end-y_start,
                        fill=False, edgecolor='yellow', linewidth=2, label='Beaker')
    ax.add_patch(rect)
    
    ax.set_title('Manual Soil Line Selection\nClick on the soil line (where soil ends and plants begin)\nPress ENTER when done', 
                 fontsize=14, fontweight='bold')
    
    soil_line_y = None
    click_points = []
    
    def on_click(event):
        nonlocal soil_line_y, click_points
        if event.button == 1:
            if event.xdata is not None and event.ydata is not None:
                click_y = int(event.ydata)
                # Ensure click is within beaker region
                if x_start <= event.xdata <= x_end and y_start <= click_y <= y_end:
                    soil_line_y = click_y
                    click_points.append((int(event.xdata), click_y))
                    
                    # Draw the point
                    ax.plot(event.xdata, click_y, 'ro', markersize=8)
                    # Draw horizontal line across beaker
                    ax.axhline(y=click_y, color='red', linewidth=2, alpha=0.7, linestyle='--')
                    
                    fig.canvas.draw()
                    print(f"  Click at y={click_y}")
                else:
                    print("⚠️ Click outside beaker region")
    
    def on_key(event):
        if event.key == 'enter':
            plt.close(fig)
        elif event.key == 'escape':
            plt.close(fig)
            raise KeyboardInterrupt("User cancelled")
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.show()
    
    if soil_line_y is None:
        print("⚠️ No soil line selected, using default")
        beaker_height = y_end - y_start
        soil_line_y = y_end - int(beaker_height * 0.20)  # Default to 20% from bottom
    elif len(click_points) > 1:
        # Use average y if multiple points clicked
        soil_line_y = int(np.mean([p[1] for p in click_points]))
    
    # Validate soil line position
    soil_line_y = max(y_start, min(soil_line_y, y_end))
    
    print(f"✅ Manual soil line selected: y={soil_line_y}")
    print(f"   Distance from beaker bottom: {y_end - soil_line_y} px")
    
    # Calculate pixel ratio (actual beaker height: 18.034 cm)
    beaker_height = y_end - y_start
    pixel_to_cm_ratio = 18.034 / beaker_height
    
    return soil_line_y, pixel_to_cm_ratio

def detect_beaker_robust(image, use_manual=False):
    """
    Robust beaker detection focusing on finding complete beaker
    """
    if use_manual:
        return manual_select_beaker_region(image)
    
    print("\n🔍 AUTOMATIC BEAKER DETECTION")
    h, w = image.shape[:2]
    
    # Strategy: Find dark soil at bottom first, then work upward
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_enhanced = clahe.apply(gray)
    
    # Look for dark soil at bottom
    # Soil occupies bottom 20-40% of beaker
    bottom_region_height = int(h * 0.4)
    bottom_region = image[h-bottom_region_height:, :]
    
    # Convert to HSV for soil detection
    hsv_bottom = cv2.cvtColor(bottom_region, cv2.COLOR_RGB2HSV)
    
    # Dark brown soil colors (as per your values)
    lower_soil1 = np.array([0, 40, 20])     # Dark brown
    upper_soil1 = np.array([30, 150, 100])
    
    lower_soil2 = np.array([0, 50, 10])     # Very dark brown
    upper_soil2 = np.array([180, 150, 70])
    
    soil_mask1 = cv2.inRange(hsv_bottom, lower_soil1, upper_soil1)
    soil_mask2 = cv2.inRange(hsv_bottom, lower_soil2, upper_soil2)
    soil_mask = cv2.bitwise_or(soil_mask1, soil_mask2)
    
    # Clean soil mask
    kernel = np.ones((15, 15), np.uint8)
    soil_mask = cv2.morphologyEx(soil_mask, cv2.MORPH_CLOSE, kernel)
    soil_mask = cv2.morphologyEx(soil_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    
    # Find soil region contours
    contours, _ = cv2.findContours(soil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest soil region
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        soil_contour = contours[0]
        
        # Get bounding box of soil
        x_soil, y_soil, w_soil, h_soil = cv2.boundingRect(soil_contour)
        
        # Adjust y coordinate to full image
        y_soil += h - bottom_region_height
        
        # Soil is typically at the bottom of beaker
        # Beaker bottom is just above soil
        beaker_bottom = y_soil + h_soil
        
        # Estimate beaker width from soil width (soil fills beaker width)
        beaker_width = w_soil
        
        # Estimate beaker position (centered horizontally)
        beaker_left = x_soil
        beaker_right = x_soil + w_soil
        
        # Estimate beaker top (typical aspect ratio for beakers is 2:1 height:width)
        estimated_height = beaker_width * 2
        beaker_top = max(0, beaker_bottom - estimated_height)
        
        # Refine using edge detection in the estimated region
        beaker_region_est = image[beaker_top:beaker_bottom, beaker_left:beaker_right]
        
        if beaker_region_est.size > 0:
            gray_beaker = cv2.cvtColor(beaker_region_est, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray_beaker, 50, 150)
            
            # Find vertical edges (beaker sides)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=30, 
                                   minLineLength=estimated_height//3, maxLineGap=10)
            
            if lines is not None:
                vertical_lines = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(x2 - x1) < 10 and abs(y2 - y1) > estimated_height//4:
                        vertical_lines.append((x1, y1, x2, y2))
                
                if vertical_lines:
                    # Use vertical lines to refine beaker edges
                    x_vals = []
                    for line in vertical_lines:
                        x_vals.extend([line[0], line[2]])
                    
                    if x_vals:
                        beaker_left_refined = beaker_left + min(x_vals)
                        beaker_right_refined = beaker_left + max(x_vals)
                        
                        if beaker_right_refined - beaker_left_refined > 50:
                            beaker_left = beaker_left_refined
                            beaker_right = beaker_right_refined
        
        # Add padding
        padding_x = int(beaker_width * 0.05)
        padding_y = int(estimated_height * 0.05)
        
        x_start = max(0, beaker_left - padding_x)
        x_end = min(w, beaker_right + padding_x)
        y_start = max(0, beaker_top - padding_y)
        y_end = min(h, beaker_bottom + padding_y)
        
        # INCREASE BEAKER HEIGHT BY 10% to account for plants growing above
        original_beaker_height = y_end - y_start
        height_increase = int(original_beaker_height * 0.10)
        y_start = max(0, y_start - height_increase)
        
        beaker_mask = np.zeros((h, w), dtype=bool)
        beaker_mask[y_start:y_end, x_start:x_end] = True
        
        print(f"✅ Beaker detected via soil: {x_end-x_start}x{y_end-y_start} px (increased height by 10%)")
        print(f"   Soil at bottom: y={y_soil}-{y_soil+h_soil}")
        
        return x_start, x_end, y_start, y_end, beaker_mask
    
    # Fallback to edge-based detection
    print("⚠️ Soil-based detection failed, using edge detection")
    
    edges = cv2.Canny(gray_enhanced, 30, 100)
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                           minLineLength=h//3, maxLineGap=20)
    
    if lines is not None:
        vertical_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 15 and abs(y2 - y1) > h//3:
                vertical_lines.append((x1, y1, x2, y2))
        
        if vertical_lines:
            # Find leftmost and rightmost lines
            left_x = min([min(line[0], line[2]) for line in vertical_lines])
            right_x = max([max(line[0], line[2]) for line in vertical_lines])
            
            # Find top and bottom from lines
            top_y = min([min(line[1], line[3]) for line in vertical_lines])
            bottom_y = max([max(line[1], line[3]) for line in vertical_lines])
            
            x_start, x_end = left_x, right_x
            y_start, y_end = top_y, bottom_y
            
            # Ensure reasonable aspect ratio
            if (x_end - x_start) > 50 and (y_end - y_start) > 100:
                # INCREASE BEAKER HEIGHT BY 10% to account for plants growing above
                original_beaker_height = y_end - y_start
                height_increase = int(original_beaker_height * 0.10)
                y_start = max(0, y_start - height_increase)
                
                beaker_mask = np.zeros((h, w), dtype=bool)
                beaker_mask[y_start:y_end, x_start:x_end] = True
                
                print(f"✅ Edge-based detection: {x_end-x_start}x{y_end-y_start} px (increased height by 10%)")
                return x_start, x_end, y_start, y_end, beaker_mask
    
    # Final fallback: ask for manual selection
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
    
    # Beaker is typically 60% of image width and 80% height
    width_ratio = 0.6
    height_ratio = 0.8
    
    x_start = int(w * (1 - width_ratio) / 2)
    x_end = int(w - x_start)
    y_start = int(h * (1 - height_ratio) / 2)
    y_end = int(h - y_start)
    
    # INCREASE BEAKER HEIGHT BY 10% to account for plants growing above
    original_beaker_height = y_end - y_start
    height_increase = int(original_beaker_height * 0.10)
    y_start = max(0, y_start - height_increase)
    
    beaker_mask = np.zeros((h, w), dtype=bool)
    beaker_mask[y_start:y_end, x_start:x_end] = True
    
    print(f"Fallback region: {x_end-x_start}x{y_end-y_start} px (increased height by 10%)")
    
    return x_start, x_end, y_start, y_end, beaker_mask

def detect_soil_line_accurate(image, beaker_region):
    """
    ACCURATE soil line detection - finds the DARKEST AREA in bottom 20% of beaker
    """
    print("\n🌱 ACCURATE SOIL LINE DETECTION")
    print("Finding darkest area in bottom 20% of beaker (soil)")
    
    x_start, x_end, y_start, y_end = beaker_region
    h, w = image.shape[:2]
    
    beaker_height = y_end - y_start
    beaker_width = x_end - x_start
    
    # Soil is ALWAYS at bottom of beaker - focus on bottom 20%
    soil_search_height = int(beaker_height * 0.20)  # BOTTOM 20% ONLY
    soil_region_start = max(y_start, y_end - soil_search_height)
    
    print(f"Searching for darkest soil in BOTTOM 20%: {soil_region_start}-{y_end} (height: {y_end-soil_region_start} px)")
    
    # Extract soil region (full beaker width, bottom 20% only)
    soil_region = image[soil_region_start:y_end, x_start:x_end]
    
    if soil_region.size == 0:
        print("⚠️ No soil region found, using default")
        default_soil_y = y_end - int(beaker_height * 0.10)  # 10% from bottom
        pixel_ratio = 18.034 / beaker_height
        return default_soil_y, pixel_ratio
    
    # CONVERT TO GRAYSCALE for DARKNESS detection
    gray_soil = cv2.cvtColor(soil_region, cv2.COLOR_RGB2GRAY)
    
    # Also convert to HSV for color-based detection
    hsv = cv2.cvtColor(soil_region, cv2.COLOR_RGB2HSV)
    
    # Method 1: Find DARKEST rows (soil is darkest in bottom)
    # Calculate average darkness per row
    row_darkness = np.mean(gray_soil, axis=1)
    
    # Invert so darker = higher value
    row_darkness = 255 - row_darkness
    
    # Smooth the darkness values
    row_darkness_smoothed = np.convolve(row_darkness, np.ones(5)/5, mode='same')
    
    # Method 2: Dark brown soil colors in HSV
    lower_brown1 = np.array([0, 40, 20])    # Dark brown
    upper_brown1 = np.array([30, 150, 100])
    
    lower_brown2 = np.array([0, 50, 10])    # Very dark brown
    upper_brown2 = np.array([180, 150, 70])
    
    # Create soil mask based on color
    hsv_mask1 = cv2.inRange(hsv, lower_brown1, upper_brown1)
    hsv_mask2 = cv2.inRange(hsv, lower_brown2, upper_brown2)
    soil_color_mask = cv2.bitwise_or(hsv_mask1, hsv_mask2)
    
    # Calculate soil color coverage per row
    row_color_coverage = np.sum(soil_color_mask, axis=1) / 255.0
    
    # COMBINE METHODS: Find rows that are both DARK and BROWN
    # Normalize both metrics
    darkness_norm = row_darkness_smoothed / np.max(row_darkness_smoothed) if np.max(row_darkness_smoothed) > 0 else row_darkness_smoothed
    color_norm = row_color_coverage / np.max(row_color_coverage) if np.max(row_color_coverage) > 0 else row_color_coverage
    
    # Combined score: prioritize darkness (soil is darkest at bottom)
    combined_score = darkness_norm * 0.7 + color_norm * 0.3
    
    # Find the row with maximum combined score (darkest+brownest)
    if len(combined_score) > 0:
        # Look in bottom 70% of soil region (soil is at very bottom)
        search_start = int(len(combined_score) * 0.3)  # Start from 30% from top of soil region
        search_end = len(combined_score)
        
        if search_end > search_start:
            soil_peak_in_region = np.argmax(combined_score[search_start:search_end]) + search_start
        else:
            soil_peak_in_region = np.argmax(combined_score)
        
        # Convert to full image coordinates
        soil_line_y = soil_region_start + soil_peak_in_region
    else:
        print("⚠️ Could not find soil peak, using default")
        soil_line_y = y_end - int(beaker_height * 0.10)
    
    # VALIDATE soil line position
    # Soil line should be in bottom 20% of beaker
    min_allowed_soil = y_start + int(beaker_height * 0.80)  # Bottom 20%
    max_allowed_soil = y_end - 5  # Leave small margin
    
    soil_line_y = max(min_allowed_soil, min(soil_line_y, max_allowed_soil))
    
    # Calculate pixel ratio (actual beaker height: 18.034 cm)
    pixel_to_cm_ratio = 18.034 / beaker_height
    
    print(f"✅ Soil line (darkest area) detected at: y={soil_line_y}")
    print(f"   Distance from beaker bottom: {y_end - soil_line_y} px")
    print(f"   From beaker top: {soil_line_y - y_start} px")
    print(f"   Beaker: {beaker_height} px = 18.034 cm")
    print(f"   Pixel ratio: {pixel_to_cm_ratio:.4f} cm/px")
    
    return soil_line_y, pixel_to_cm_ratio

def detect_plants_enhanced(image, beaker_region, soil_line_y):
    """
    ENHANCED plant detection - specifically excludes soil
    """
    print("\n🌿 ENHANCED PLANT DETECTION (excluding soil)")
    
    x_start, x_end, y_start, y_end = beaker_region
    h, w = image.shape[:2]
    
    # Plants are ABOVE soil line, within beaker width
    plant_top = max(0, y_start - 50)  # Allow 50px above beaker (plants can grow above)
    plant_bottom = soil_line_y  # Stop at soil line
    
    plant_height = plant_bottom - plant_top
    plant_width = x_end - x_start
    
    print(f"Plant search region: {plant_width}x{plant_height} px")
    print(f"From y={plant_top} (above beaker) to y={plant_bottom} (soil line)")
    print(f"Excluding soil region completely")
    
    # Extract plant region (ABOVE soil only)
    plant_region = image[plant_top:plant_bottom, x_start:x_end]
    
    if plant_region.size == 0:
        print("⚠️ No plant region found")
        return np.zeros((h, w), dtype=bool)
    
    # Convert to multiple color spaces
    hsv = cv2.cvtColor(plant_region, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(plant_region, cv2.COLOR_RGB2LAB)
    
    # WHEATGRASS COLOR DETECTION (GREEN to YELLOW to BROWN - but NOT soil brown)
    
    # Healthy green wheatgrass
    lower_green1 = np.array([35, 40, 40])
    upper_green1 = np.array([85, 255, 220])
    
    # Dark green
    lower_green2 = np.array([25, 30, 20])
    upper_green2 = np.array([95, 255, 180])
    
    # Yellowish (maturing)
    lower_yellow = np.array([20, 40, 50])
    upper_yellow = np.array([35, 200, 200])
    
    # Plant brown (NOT soil brown - lighter, drier)
    lower_plant_brown = np.array([5, 30, 50])   # Lighter brown
    upper_plant_brown = np.array([20, 150, 180])
    
    # CRITICAL: EXCLUDE SOIL COLORS
    lower_soil_exclude = np.array([0, 40, 10])    # Dark soil brown
    upper_soil_exclude = np.array([30, 150, 80])  # Exclude dark brown
    
    # Create masks
    masks = []
    masks.append(cv2.inRange(hsv, lower_green1, upper_green1))
    masks.append(cv2.inRange(hsv, lower_green2, upper_green2))
    masks.append(cv2.inRange(hsv, lower_yellow, upper_yellow))
    masks.append(cv2.inRange(hsv, lower_plant_brown, upper_plant_brown))
    
    # Combine plant masks
    plant_mask = masks[0]
    for mask in masks[1:]:
        plant_mask = cv2.bitwise_or(plant_mask, mask)
    
    # EXCLUDE SOIL COLORS
    soil_mask_exclude = cv2.inRange(hsv, lower_soil_exclude, upper_soil_exclude)
    plant_mask = cv2.bitwise_and(plant_mask, cv2.bitwise_not(soil_mask_exclude))
    
    # Also exclude very dark pixels (likely soil/shadow)
    v_channel = hsv[:, :, 2]
    dark_pixels = v_channel < 40
    plant_mask[dark_pixels] = 0
    
    # EXCLUDE BACKGROUND COLORS
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
    
    # POST-PROCESSING
    # Initial cleaning
    kernel_open = np.ones((2, 2), np.uint8)
    kernel_close = np.ones((3, 3), np.uint8)
    
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel_open)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # Remove small components
    plant_mask = remove_small_components_enhanced(plant_mask, min_size=20)
    
    # Enhance vertical structures (wheatgrass grows vertically)
    vertical_kernel = np.ones((7, 1), np.uint8)
    plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, vertical_kernel)
    
    # Create full image mask
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[plant_top:plant_bottom, x_start:x_end] = plant_mask
    
    # Also check for plants above the beaker
    if plant_top < y_start:
        above_beaker_region = image[plant_top:y_start, x_start:x_end]
        if above_beaker_region.size > 0:
            above_hsv = cv2.cvtColor(above_beaker_region, cv2.COLOR_RGB2HSV)
            
            above_mask1 = cv2.inRange(above_hsv, lower_green1, upper_green1)
            above_mask2 = cv2.inRange(above_hsv, lower_green2, upper_green2)
            above_mask = above_mask1 | above_mask2
            
            above_mask = cv2.morphologyEx(above_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
            above_mask = remove_small_components_enhanced(above_mask, min_size=10)
            
            full_mask[plant_top:y_start, x_start:x_end] = above_mask
    
    plant_count = np.sum(full_mask > 0)
    print(f"✅ Plants detected: {plant_count} pixels (excluding soil)")
    
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
        
        # Keep if large enough or tall and thin (wheatgrass)
        if (area >= min_size) or (aspect_ratio > 1.5 and area >= min_size//2):
            cleaned_mask[labels == i] = 255
    
    return cleaned_mask

def extract_canopy_boundary_enhanced(plant_mask, soil_line_y, beaker_region):
    """
    Canopy boundary extraction - plants above soil only
    """
    print("\n📐 CANOPY BOUNDARY EXTRACTION")
    
    h, w = plant_mask.shape
    x_start, x_end, y_start, y_end = beaker_region
    
    # Create working mask (plants above soil only)
    working_mask = plant_mask.copy()
    working_mask[soil_line_y:, :] = False  # Remove everything at/below soil
    
    # Constrain to beaker width
    width_mask = np.zeros_like(working_mask, dtype=bool)
    width_mask[:, x_start:x_end] = True
    working_mask = np.logical_and(working_mask, width_mask)
    
    # Extract boundary points
    boundary_points = []
    sampling_step = max(1, (x_end - x_start) // 100)
    
    for x in range(x_start, x_end, sampling_step):
        column = working_mask[:, x]
        if np.any(column):
            plant_pixels = np.where(column)[0]
            if len(plant_pixels) > 0:
                top_y = plant_pixels[0]  # Topmost plant pixel
                boundary_points.append((x, top_y))
    
    if len(boundary_points) < 5:
        print(f"⚠️ Only {len(boundary_points)} boundary points found")
        
        # Try horizontal projection method
        horizontal_proj = np.sum(working_mask, axis=1)
        if np.max(horizontal_proj) > 0:
            threshold = np.max(horizontal_proj) * 0.1
            plant_rows = np.where(horizontal_proj > threshold)[0]
            
            if len(plant_rows) > 0:
                top_row = np.min(plant_rows)
                for x in range(x_start, x_end, sampling_step*3):
                    if working_mask[top_row, x]:
                        boundary_points.append((x, top_row))
    
    if len(boundary_points) < 3:
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
    
    print(f"✅ Canopy boundary: {len(x_vals)} points")
    
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
    
    # Health categories
    healthy_mask = (hue >= 35) & (hue <= 85) & (saturation >= 40) & (value >= 40)
    mature_mask = ((hue >= 25) & (hue < 35)) | ((hue > 85) & (hue <= 95)) & (saturation >= 30)
    drying_mask = ((hue >= 10) & (hue < 25)) & (saturation >= 20) & (value >= 50)
    
    total_pixels = len(hue)
    healthy_pct = np.sum(healthy_mask) / total_pixels if total_pixels > 0 else 0
    mature_pct = np.sum(mature_mask) / total_pixels if total_pixels > 0 else 0
    drying_pct = np.sum(drying_mask) / total_pixels if total_pixels > 0 else 0
    
    # Health score
    health_score = (healthy_pct * 0.7 + mature_pct * 0.3 + drying_pct * 0.1)
    health_score = max(0, min(1, health_score))
    
    greenness_score = np.mean((hue >= 30) & (hue <= 90)) if np.any(hue) else 0
    colorfulness_score = np.mean(saturation) / 255.0 if np.any(saturation) else 0
    
    return health_score, greenness_score, colorfulness_score

def analyze_wheatgrass_enhanced(image_path):
    """
    Main enhanced analysis function with accurate soil detection
    """
    print("=" * 70)
    print("🌾 ENHANCED WHEATGRASS ANALYSIS WITH ACCURATE SOIL DETECTION")
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
        
        # Detect beaker (with 10% increased height)
        beaker_region = detect_beaker_robust(image, use_manual=not detection_choice)
        x_start, x_end, y_start, y_end, beaker_mask = beaker_region
        
        # Ask user if they want manual soil line selection
        root = tk.Tk()
        root.withdraw()
        soil_choice = messagebox.askyesno("Soil Line Detection",
                                        "Select soil line detection method:\n\n"
                                        "YES: Manual selection (click on soil line)\n"
                                        "NO: Automatic detection (darkest area in bottom 20%)")
        root.destroy()
        
        if soil_choice:
            # Manual soil line selection
            soil_line_y, pixel_ratio = manual_select_soil_line(image, (x_start, x_end, y_start, y_end))
            print(f"📝 Using MANUALLY selected soil line: y={soil_line_y}")
        else:
            # Automatic soil line detection
            soil_line_y, pixel_ratio = detect_soil_line_accurate(image, (x_start, x_end, y_start, y_end))
            print(f"🤖 Using AUTOMATICALLY detected soil line: y={soil_line_y}")
        
        # Detect plants (EXCLUDES soil)
        plant_mask = detect_plants_enhanced(image, (x_start, x_end, y_start, y_end), soil_line_y)
        
        if np.sum(plant_mask) == 0:
            print("❌ No plants detected!")
            return None
        
        # Extract canopy boundary
        canopy_x, canopy_y = extract_canopy_boundary_enhanced(plant_mask, soil_line_y, (x_start, x_end, y_start, y_end))
        
        if len(canopy_x) == 0:
            print("❌ No canopy boundary extracted!")
            return None
        
        # Calculate heights
        heights_pixels = soil_line_y - canopy_y
        heights_cm = heights_pixels * pixel_ratio
        heights_inch = heights_cm / 2.54
        
        # Filter valid heights
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
            'beaker_region': (x_start, x_end, y_start, y_end),
            'soil_selection_method': 'manual' if soil_choice else 'automatic'
        }
        
        # Display results
        print(f"\n📊 ACCURATE RESULTS:")
        print(f"  Average Height: {avg_height_cm:.2f} cm ({avg_height_inch:.2f} in)")
        print(f"  Height Range: {min_height_cm:.2f} to {max_height_cm:.2f} cm")
        print(f"  Standard Deviation: {std_height_cm:.2f} cm")
        print(f"  Plant Health Score: {health_score:.3f}/1.0")
        print(f"  Plant Pixels Detected: {np.sum(plant_mask):,}")
        print(f"  Soil Line Position: y={soil_line_y} ({'manually' if soil_choice else 'automatically'} selected)")
        print(f"  Beaker Height: +10% to account for plants growing above")
        
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
    Create enhanced visualization with soil emphasis
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    
    # Determine soil selection method for title
    soil_method = results.get('soil_selection_method', 'automatic')
    method_text = 'Manual Soil Line' if soil_method == 'manual' else 'Automatic Soil Detection'
    
    fig.suptitle(f'Enhanced Wheatgrass Analysis - {image_name}\n({method_text} + Extended Beaker)', 
                 fontsize=18, fontweight='bold')
    
    h, w = image.shape[:2]
    x_start, x_end, y_start, y_end = beaker_region
    
    # Plot 1: Overview with soil emphasis
    axes[0, 0].imshow(image)
    # Beaker rectangle (with extended height)
    rect = plt.Rectangle((x_start, y_start), x_end-x_start, y_end-y_start,
                        fill=False, edgecolor='yellow', linewidth=3, label='Beaker (+10% height)')
    axes[0, 0].add_patch(rect)
    # Original beaker top (if extended)
    if y_start > 0:
        axes[0, 0].axhline(y=y_start + (y_end-y_start)*0.1, color='orange', linewidth=1, 
                          linestyle='--', alpha=0.5, label='Original beaker top')
    # Soil line (emphasized)
    soil_label = 'Soil Line (manually selected)' if soil_method == 'manual' else 'Soil Line (darkest area)'
    axes[0, 0].axhline(y=soil_line_y, color='#8B4513', linewidth=4, label=soil_label, alpha=0.8)
    # Highlight soil region (bottom 20%)
    soil_bottom_20_start = y_start + int((y_end - y_start) * 0.80)
    axes[0, 0].axhspan(soil_line_y, y_end, alpha=0.15, color='brown', label='Soil region')
    # Canopy line
    if len(canopy_x) > 0:
        axes[0, 0].plot(canopy_x, canopy_y, 'r-', linewidth=2, alpha=0.8, label='Canopy')
    axes[0, 0].set_title(f'1. Detection Overview\n({method_text})', fontsize=14)
    axes[0, 0].legend(loc='upper right', fontsize=8)
    axes[0, 0].axis('off')
    
    # Plot 2: Plant detection with health colors
    health_image = np.zeros_like(image)
    plant_indices = np.where(plant_mask)
    
    if len(plant_indices[0]) > 0:
        plant_pixels = image[plant_mask]
        hsv_pixels = cv2.cvtColor(plant_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        hue = hsv_pixels[:, 0]
        
        for i in range(len(plant_indices[0])):
            y, x = plant_indices[0][i], plant_indices[1][i]
            h = hue[i] if i < len(hue) else 0
            
            if 35 <= h <= 85:  # Healthy green
                health_image[y, x] = [0, 200, 0]
            elif 25 <= h < 35 or 85 < h <= 95:  # Mature
                health_image[y, x] = [200, 200, 0]
            elif 10 <= h < 25:  # Drying
                health_image[y, x] = [160, 100, 50]
    
    blended = cv2.addWeighted(image, 0.3, health_image, 0.7, 0)
    axes[0, 1].imshow(blended)
    axes[0, 1].axhline(y=soil_line_y, color='brown', linewidth=3, linestyle='--')
    if len(canopy_x) > 0:
        axes[0, 1].plot(canopy_x, canopy_y, 'r-', linewidth=1.5, alpha=0.7)
    axes[0, 1].set_title('2. Plant Health\n(Green=Healthy, Yellow=Mature, Brown=Drying)', fontsize=14)
    axes[0, 1].axis('off')
    
    # Plot 3: Height profile
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
        axes[0, 2].set_title('3. Height Profile\n(From soil line to canopy)', fontsize=14)
        axes[0, 2].set_xlabel('Position along canopy')
        axes[0, 2].set_ylabel('Height (cm)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Plant mask (plants only, no soil)
    axes[1, 0].imshow(plant_mask, cmap='Greens')
    axes[1, 0].axhline(y=soil_line_y, color='brown', linewidth=3, linestyle='--', label='Soil Line')
    # Beaker boundaries
    axes[1, 0].axvline(x=x_start, color='yellow', linewidth=1, linestyle='--', alpha=0.5)
    axes[1, 0].axvline(x=x_end, color='yellow', linewidth=1, linestyle='--', alpha=0.5)
    if len(canopy_x) > 0:
        axes[1, 0].plot(canopy_x, canopy_y, 'r-', linewidth=1.5, alpha=0.7)
    axes[1, 0].set_title('4. Plant Detection Mask\n(Excluding soil region)', fontsize=14)
    axes[1, 0].legend(loc='upper right', fontsize=9)
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
        axes[1, 1].set_title('5. Height Distribution\n(Measurements from soil line)', fontsize=14)
        axes[1, 1].set_xlabel('Height (cm)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Enhanced summary
    soil_percent = (y_end - soil_line_y) / (y_end - y_start) * 100
    soil_method_text = "Manually selected by user" if soil_method == 'manual' else "Automatically detected (darkest bottom 20%)"
    
    stats_text = f"""
    ENHANCED WHEATGRASS ANALYSIS
    ============================
    
    Key Features:
    • {soil_method_text}
    • Beaker height extended by 10% for plants growing above
    • Plants measured from soil line upward
    
    Beaker Detection:
    • Width: {x_end-x_start} px
    • Height: {y_end-y_start} px (+10% extended)
    • Original height: {int((y_end-y_start)/1.1)} px
    • Pixel Ratio: {results['pixel_ratio']:.4f} cm/px
    • Soil position: {soil_percent:.1f}% from bottom
    
    Height Statistics (from soil line):
    • Average: {results['avg_height_cm']:.2f} cm
    • Maximum: {results['max_height_cm']:.2f} cm
    • Minimum: {results['min_height_cm']:.2f} cm
    • Median: {results['median_height_cm']:.2f} cm
    • Std Dev: {results['std_height_cm']:.2f} cm
    • Range: {max(heights_cm)-min(heights_cm):.2f} cm
    
    Plant Health:
    • Overall Health: {results['health_score']:.3f}/1.0
    • Greenness: {results['greenness_score']:.3f}/1.0
    • Plant Pixels: {np.sum(plant_mask):,}
    
    Detection Quality:
    • Canopy Points: {len(canopy_x)}
    • Soil Line: y={soil_line_y} px
    • Soil Method: {soil_method_text}
    • Beaker Coverage: {100*(y_end-y_start)/h:.1f}% of image
    • Extended for: plants growing above beaker
    """
    
    axes[1, 2].text(0.05, 0.95, stats_text, fontsize=9, family='monospace',
                   verticalalignment='top', transform=axes[1, 2].transAxes,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", 
                           alpha=0.95, edgecolor='orange'))
    axes[1, 2].set_title('6. Analysis Summary', fontsize=14)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main application function
    """
    print("\n" + "="*70)
    print("🌾 ENHANCED WHEATGRASS ANALYSIS WITH ACCURATE SOIL DETECTION")
    print("="*70)
    print("Key Improvements:")
    print("• OPTIONAL manual soil line selection (click on image)")
    print("• Automatic soil detection as darkest area in bottom 20%")
    print("• Beaker height INCREASED BY 10% to account for plants growing above")
    print("• Plants measured ONLY from soil line upward")
    print("• Soil excluded from all plant calculations")
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
                                    f"• Soil line: {'manually' if results.get('soil_selection_method') == 'manual' else 'automatically'} selected\n"
                                    "• Beaker extended by 10% for plants\n\n"
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