"""
🌾 WHEATGRASS ANALYZER WEB APPLICATION
Converts desktop app to web interface - ALL FUNCTIONALITY PRESERVED
"""

# Vercel detection
IS_VERCEL = os.environ.get('VERCEL') == '1'

if IS_VERCEL:
    # Use /tmp for uploads on Vercel
    UPLOAD_FOLDER = '/tmp/uploads'
    RESULTS_FOLDER = '/tmp/results'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
else:
    # Local development
    UPLOAD_FOLDER = 'static/uploads'
    RESULTS_FOLDER = 'wheatgrass_analysis_results'

# Then in your Flask app config section, replace:
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['RESULTS_FOLDER'] = 'wheatgrass_analysis_results'
# With:

import os
import sys
import io
import base64
import json
import traceback
from datetime import datetime
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for web
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import warnings
warnings.filterwarnings('ignore')

# Flask imports
from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename

# Import SAM (optional)
try:
    from segment_anything import sam_model_registry, SamPredictor
    import torch
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️ SAM not available, using enhanced traditional methods")

# ===== FLASK APP CONFIGURATION =====
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['SECRET_KEY'] = 'wheatgrass-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# ===== YOUR EXACT CODE (ADAPTED FOR WEB) =====

# Remove PyInstaller DLL fix (not needed for web)
# Keep all your imports and functions exactly as they are

# Configuration
SAM_MODEL_PATH = "sam_vit_b_01ec64.pth" if SAM_AVAILABLE else None
SAM_MODEL_TYPE = "vit_b"
SAM_PREDICTOR = None

def get_sam_model_path():
    """Get SAM model path - Vercel compatible"""
    # Default local path
    local_path = "sam_vit_b_01ec64.pth"
    
    # Check if we're on Vercel
    if os.environ.get('VERCEL') == '1':
        # On Vercel, try to find model in /tmp
        tmp_path = "/tmp/sam_models/sam_vit_b_01ec64.pth"
        
        # If model doesn't exist in /tmp, download it
        if not os.path.exists(tmp_path):
            print("⚠️ Vercel: SAM model not found, trying to download...")
            
            # Try to import downloader
            try:
                from sam_download import get_sam_model_path as download_sam
                downloaded_path = download_sam("vit_b")
                if downloaded_path:
                    return downloaded_path
            except ImportError:
                print("⚠️ SAM downloader not available")
        
        # Check if model exists in /tmp
        if os.path.exists(tmp_path):
            print(f"✅ Vercel: Using SAM model from /tmp: {tmp_path}")
            return tmp_path
        
        print("⚠️ Vercel: SAM model not available, using traditional methods")
        return None
    
    # Local development - use local file
    if os.path.exists(local_path):
        print(f"✅ Local: Using SAM model: {local_path}")
        return local_path
    
    print("⚠️ Local: SAM model not found")
    return None

# Update your initialize_sam() function - ONLY CHANGE THESE LINES:
def initialize_sam():
    """Initialize SAM model if available"""
    global SAM_PREDICTOR
    
    if not SAM_AVAILABLE:
        return False
    
    try:
        # Use the Vercel-compatible function
        SAM_MODEL_PATH = get_sam_model_path()
        
        if not SAM_MODEL_PATH or not os.path.exists(SAM_MODEL_PATH):
            print(f"⚠️ SAM model not found at {SAM_MODEL_PATH}")
            return False
        
        print(f"🔍 Loading SAM model from: {SAM_MODEL_PATH}")
        
        # On Vercel, always use CPU
        device = "cpu" if os.environ.get('VERCEL') == '1' else ("cuda" if torch.cuda.is_available() else "cpu")
        
        sam = sam_model_registry["vit_b"](checkpoint=SAM_MODEL_PATH)
        sam.to(device=device)
        SAM_PREDICTOR = SamPredictor(sam)
        print(f"✅ SAM loaded on {device}")
        return True
        
    except Exception as e:
        print(f"⚠️ SAM initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# Modified: Load image from bytes instead of file path
def load_wheat_image(image_data):
    """Load image from bytes (web version)"""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not load image from data")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# ===== COPY ALL YOUR FUNCTIONS HERE (EXACTLY AS IS) =====

def detect_beaker_ultimate(image):
    """
    ULTIMATE beaker detection - focuses on center vertical structure
    """
    h, w = image.shape[:2]
    
    # Method 1: Try SAM first (most accurate)
    if SAM_PREDICTOR is not None:
        try:
            SAM_PREDICTOR.set_image(image)
            
            # Create intelligent point prompts focused on beaker center
            center_x, center_y = w // 2, h // 2
            points = []
            
            # Beaker center points (vertical alignment)
            points.append([center_x, center_y])
            points.append([center_x, center_y - 100])  # Top of beaker
            points.append([center_x, center_y + 100])  # Bottom of beaker
            points.append([center_x - 50, center_y])   # Left side
            points.append([center_x + 50, center_y])   # Right side
            
            points = np.array(points)
            labels = np.ones(len(points))
            
            masks, scores, _ = SAM_PREDICTOR.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True
            )
            
            best_idx = np.argmax(scores)
            sam_mask = masks[best_idx].astype(np.uint8)
            
            # Refine mask
            kernel = np.ones((5, 5), np.uint8)
            sam_mask = cv2.morphologyEx(sam_mask, cv2.MORPH_CLOSE, kernel)
            sam_mask = cv2.morphologyEx(sam_mask, cv2.MORPH_OPEN, kernel)
            
            # Find bounding box
            rows, cols = np.where(sam_mask)
            if len(rows) > 100:  # Valid detection
                x_start, x_end = np.min(cols), np.max(cols)
                y_start, y_end = np.min(rows), np.max(rows)
                
                # Ensure beaker is taller than wide
                if (y_end - y_start) > (x_end - x_start) * 0.8:
                    beaker_mask = sam_mask.astype(bool)
                    
                    # ADD EXTRA HEIGHT for plants growing above
                    extra_height = int((y_end - y_start) * 0.15)
                    y_start = max(0, y_start - extra_height)
                    
                    return x_start, x_end, y_start, y_end, beaker_mask
        except Exception as e:
            print(f"⚠️ SAM detection failed: {e}")
    
    # Method 2: Vertical edge detection (beakers have strong vertical edges)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Enhanced edge detection
    edges = cv2.Canny(gray_blur, 30, 100)
    
    # Find vertical lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                           minLineLength=h//3, maxLineGap=20)
    
    vertical_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Check if line is mostly vertical
            if abs(x2 - x1) < 20 and abs(y2 - y1) > h//4:
                vertical_lines.append((min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)))
    
    if len(vertical_lines) >= 2:
        # Find leftmost and rightmost vertical lines
        left_lines = [l for l in vertical_lines if l[0] < w//2]
        right_lines = [l for l in vertical_lines if l[1] > w//2]
        
        if left_lines and right_lines:
            left_x = np.mean([l[0] for l in left_lines])
            right_x = np.mean([l[1] for l in right_lines])
            top_y = np.min([l[2] for l in vertical_lines])
            bottom_y = np.max([l[3] for l in vertical_lines])
            
            x_start = int(left_x)
            x_end = int(right_x)
            y_start = int(top_y)
            y_end = int(bottom_y)
            
            # Add small padding
            padding = int((x_end - x_start) * 0.02)
            x_start = max(0, x_start - padding)
            x_end = min(w, x_end + padding)
            
            # ADD EXTRA HEIGHT for plants growing above
            extra_height = int((y_end - y_start) * 0.15)
            y_start = max(0, y_start - extra_height)
            
            beaker_mask = np.zeros((h, w), dtype=bool)
            beaker_mask[y_start:y_end, x_start:x_end] = True
            
            return x_start, x_end, y_start, y_end, beaker_mask
    
    # Method 3: Color-based background subtraction
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Common background colors (pink/blue)
    lower_pink = np.array([140, 20, 140])
    upper_pink = np.array([180, 100, 255])
    pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
    
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    bg_mask = pink_mask | blue_mask
    
    # Clean and fill background
    kernel_large = np.ones((21, 21), np.uint8)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel_large)
    
    # Find non-background regions (beaker + plants)
    non_bg = cv2.bitwise_not(bg_mask)
    contours, _ = cv2.findContours(non_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, rect_w, rect_h = cv2.boundingRect(largest_contour)
        
        # Validate aspect ratio
        if rect_h > rect_w * 1.2:  # Should be taller than wide
            x_start = x
            x_end = x + rect_w
            y_start = y
            y_end = y + rect_h
            
            # ADD EXTRA HEIGHT for plants growing above
            extra_height = int(rect_h * 0.15)
            y_start = max(0, y_start - extra_height)
            
            beaker_mask = np.zeros((h, w), dtype=bool)
            beaker_mask[y_start:y_end, x_start:x_end] = True
            
            return x_start, x_end, y_start, y_end, beaker_mask
    
    # Method 4: Center-based fallback (intelligent)
    center_ratio = 0.7
    height_ratio = 0.8
    
    x_start = int(w * (1 - center_ratio) / 2)
    x_end = int(w - x_start)
    y_start = int(h * (1 - height_ratio) / 2)
    y_end = int(h - y_start)
    
    # ADD EXTRA HEIGHT for plants growing above
    extra_height = int((y_end - y_start) * 0.15)
    y_start = max(0, y_start - extra_height)
    
    beaker_mask = np.zeros((h, w), dtype=bool)
    beaker_mask[y_start:y_end, x_start:x_end] = True
    
    return x_start, x_end, y_start, y_end, beaker_mask

def detect_soil_line_ultimate(image, beaker_region):
    """
    ULTIMATE soil line detection - finds DARKEST BROWN line inside beaker
    """
    x_start, x_end, y_start, y_end = beaker_region
    h, w = image.shape[:2]
    
    beaker_height = y_end - y_start
    beaker_width = x_end - x_start
    
    # Extract beaker region ONLY
    beaker_image = image[y_start:y_end, x_start:x_end]
    
    if beaker_image.size == 0:
        default_soil_y = y_end - int(beaker_height * 0.15)
        pixel_ratio = 18.034 * 1.175 / beaker_height
        return default_soil_y, pixel_ratio
    
    # Convert to multiple color spaces
    hsv = cv2.cvtColor(beaker_image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(beaker_image, cv2.COLOR_RGB2LAB)
    
    # SOIL DETECTION STRATEGY
    lower_dark_brown1 = np.array([0, 30, 0])
    upper_dark_brown1 = np.array([30, 120, 60])
    
    lower_dark_brown2 = np.array([5, 40, 5])
    upper_dark_brown2 = np.array([25, 100, 50])
    
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 40, 30])
    
    # Create soil masks
    soil_mask1 = cv2.inRange(hsv, lower_dark_brown1, upper_dark_brown1)
    soil_mask2 = cv2.inRange(hsv, lower_dark_brown2, upper_dark_brown2)
    soil_mask3 = cv2.inRange(hsv, lower_black, upper_black)
    
    soil_mask_combined = cv2.bitwise_or(soil_mask1, soil_mask2)
    soil_mask_combined = cv2.bitwise_or(soil_mask_combined, soil_mask3)
    
    # LAB space
    L_channel = lab[:, :, 0]
    lab_dark_mask = (L_channel < 50).astype(np.uint8) * 255
    
    # Value channel
    V_channel = hsv[:, :, 2]
    v_dark_mask = (V_channel < 60).astype(np.uint8) * 255
    
    # FINAL SOIL MASK
    soil_mask = np.zeros_like(soil_mask_combined, dtype=np.float32)
    soil_mask += soil_mask_combined.astype(np.float32) * 2.0
    soil_mask += lab_dark_mask.astype(np.float32) * 1.5
    soil_mask += v_dark_mask.astype(np.float32) * 1.2
    
    soil_mask_binary = (soil_mask > 150).astype(np.uint8) * 255
    
    # CLEAN THE MASK
    kernel_clean = np.ones((3, 3), np.uint8)
    soil_mask_binary = cv2.morphologyEx(soil_mask_binary, cv2.MORPH_OPEN, kernel_clean)
    soil_mask_binary = cv2.morphologyEx(soil_mask_binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    soil_mask_binary = cv2.morphologyEx(soil_mask_binary, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    
    # FOCUS ON BOTTOM 10-20% OF BEAKER ONLY
    beaker_h = beaker_image.shape[0]
    search_top = int(beaker_h * 0.80)
    search_bottom = int(beaker_h * 0.90)
    
    search_mask = np.zeros_like(soil_mask_binary)
    search_mask[search_top:search_bottom, :] = 255
    soil_mask_binary = cv2.bitwise_and(soil_mask_binary, search_mask)
    
    if np.sum(soil_mask_binary) < 100:
        search_top = int(beaker_h * 0.75)
        search_bottom = int(beaker_h * 0.95)
        search_mask = np.zeros_like(soil_mask_binary)
        search_mask[search_top:search_bottom, :] = 255
        soil_mask_binary = cv2.bitwise_and(soil_mask_binary, search_mask)
    
    # FIND SOIL LINE using horizontal projection
    row_density = np.sum(soil_mask_binary, axis=1) / beaker_width
    
    if np.max(row_density) < 0.05:
        soil_line_in_beaker = int(beaker_h * 0.85)
    else:
        if len(row_density) > 10:
            row_density_smooth = np.convolve(row_density, np.ones(5)/5, mode='same')
        else:
            row_density_smooth = row_density
        
        threshold = np.max(row_density_smooth) * 0.30
        soil_line_in_beaker = None
        
        search_indices = list(range(search_top, min(search_bottom, len(row_density_smooth))))
        
        for i in search_indices:
            if row_density_smooth[i] >= threshold:
                check_ahead = min(3, len(row_density_smooth) - i - 1)
                if check_ahead > 0:
                    ahead_avg = np.mean(row_density_smooth[i:i+check_ahead])
                    if ahead_avg >= threshold * 0.8:
                        soil_line_in_beaker = i
                        break
        
        if soil_line_in_beaker is None:
            if len(search_indices) > 0:
                search_density = row_density_smooth[search_indices[0]:search_indices[-1]]
                if len(search_density) > 0:
                    max_in_region = np.argmax(search_density)
                    soil_line_in_beaker = search_indices[0] + max_in_region
                else:
                    soil_line_in_beaker = search_indices[0]
            else:
                soil_line_in_beaker = int(beaker_h * 0.85)
    
    soil_line_y = y_start + soil_line_in_beaker
    
    # VALIDATION
    min_allowed_y = y_start + int(beaker_height * 0.75)
    max_allowed_y = y_end - int(beaker_height * 0.05)
    soil_line_y = max(min_allowed_y, min(soil_line_y, max_allowed_y))
    
    pixel_to_cm_ratio = 18.034 * 1.175 / beaker_height
    
    return soil_line_y, pixel_to_cm_ratio

def detect_plants_ultimate(image, beaker_region, soil_line_y):
    """
    ULTIMATE plant detection - detects ALL plant colors properly
    """
    x_start, x_end, y_start, y_end = beaker_region
    h, w = image.shape[:2]
    
    plant_top = max(0, y_start - 100)
    plant_bottom = soil_line_y
    
    plant_region = image[plant_top:plant_bottom, x_start:x_end]
    
    if plant_region.size == 0:
        return np.zeros((h, w), dtype=bool)
    
    hsv = cv2.cvtColor(plant_region, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(plant_region, cv2.COLOR_RGB2LAB)
    
    # EXPANDED WHEATGRASS COLOR RANGES
    lower_dark_green = np.array([35, 40, 30])
    upper_dark_green = np.array([85, 255, 180])
    
    lower_light_green = np.array([40, 30, 50])
    upper_light_green = np.array([90, 200, 220])
    
    lower_yellow_green = np.array([25, 40, 40])
    upper_yellow_green = np.array([40, 200, 200])
    
    lower_light_brown = np.array([10, 30, 40])
    upper_light_brown = np.array([25, 150, 160])
    
    lower_yellow = np.array([15, 40, 50])
    upper_yellow = np.array([30, 180, 200])
    
    lower_all_green = np.array([30, 20, 30])
    upper_all_green = np.array([100, 220, 200])
    
    lower_lab_green = np.array([0, 120, 120])
    upper_lab_green = np.array([255, 140, 140])
    
    # Create masks for ALL PLANT COLORS
    plant_masks = []
    plant_masks.append(cv2.inRange(hsv, lower_dark_green, upper_dark_green))
    plant_masks.append(cv2.inRange(hsv, lower_light_green, upper_light_green))
    plant_masks.append(cv2.inRange(hsv, lower_yellow_green, upper_yellow_green))
    plant_masks.append(cv2.inRange(hsv, lower_light_brown, upper_light_brown))
    plant_masks.append(cv2.inRange(hsv, lower_yellow, upper_yellow))
    plant_masks.append(cv2.inRange(hsv, lower_all_green, upper_all_green))
    plant_masks.append(cv2.inRange(lab, lower_lab_green, upper_lab_green))
    
    plant_mask_combined = plant_masks[0]
    for mask in plant_masks[1:]:
        plant_mask_combined = cv2.bitwise_or(plant_mask_combined, mask)
    
    # EXCLUDE PINK BACKGROUND
    lower_pink1 = np.array([140, 20, 140])
    upper_pink1 = np.array([180, 100, 255])
    lower_pink2 = np.array([160, 15, 150])
    upper_pink2 = np.array([180, 80, 255])
    lower_pink3 = np.array([0, 10, 140])
    upper_pink3 = np.array([10, 90, 255])
    
    pink_mask1 = cv2.inRange(hsv, lower_pink1, upper_pink1)
    pink_mask2 = cv2.inRange(hsv, lower_pink2, upper_pink2)
    pink_mask3 = cv2.inRange(hsv, lower_pink3, upper_pink3)
    
    pink_mask_combined = cv2.bitwise_or(pink_mask1, pink_mask2)
    pink_mask_combined = cv2.bitwise_or(pink_mask_combined, pink_mask3)
    
    lower_light = np.array([0, 0, 200])
    upper_light = np.array([180, 30, 255])
    light_mask = cv2.inRange(hsv, lower_light, upper_light)
    
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 40])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
    
    background_mask = cv2.bitwise_or(pink_mask_combined, light_mask)
    background_mask = cv2.bitwise_or(background_mask, dark_mask)
    
    plant_mask_combined = cv2.bitwise_and(plant_mask_combined, cv2.bitwise_not(background_mask))
    
    # POST-PROCESSING
    kernel_open = np.ones((2, 2), np.uint8)
    plant_mask_combined = cv2.morphologyEx(plant_mask_combined, cv2.MORPH_OPEN, kernel_open)
    
    kernel_close = np.ones((3, 3), np.uint8)
    plant_mask_combined = cv2.morphologyEx(plant_mask_combined, cv2.MORPH_CLOSE, kernel_close)
    
    vertical_kernel = np.ones((13, 1), np.uint8)
    plant_mask_combined = cv2.morphologyEx(plant_mask_combined, cv2.MORPH_CLOSE, vertical_kernel)
    
    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(plant_mask_combined, connectivity=8)
    cleaned_mask = np.zeros_like(plant_mask_combined)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        if area >= 20:
            if width > 0:
                aspect_ratio = height / width
            else:
                aspect_ratio = 0
            
            if (aspect_ratio > 1.5 and height > 15) or area > 80 or (area > 40 and height > 10):
                cleaned_mask[labels == i] = 255
    
    plant_mask_combined = cleaned_mask
    
    # Fill small holes
    contours, hierarchy = cv2.findContours(plant_mask_combined, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = plant_mask_combined.copy()
    
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, (cnt, hier) in enumerate(zip(contours, hierarchy)):
            if hier[3] >= 0:
                area = cv2.contourArea(cnt)
                if area <= 40:
                    cv2.drawContours(filled_mask, [cnt], 0, 255, -1)
    
    plant_mask_combined = filled_mask
    
    vertical_dilate = np.ones((5, 1), np.uint8)
    plant_mask_combined = cv2.dilate(plant_mask_combined, vertical_dilate, iterations=1)
    
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[plant_top:plant_bottom, x_start:x_end] = plant_mask_combined
    
    # Check for plants ABOVE the beaker
    if plant_top < y_start:
        above_beaker = image[plant_top:y_start, x_start:x_end]
        if above_beaker.size > 0:
            above_hsv = cv2.cvtColor(above_beaker, cv2.COLOR_RGB2HSV)
            above_lab = cv2.cvtColor(above_beaker, cv2.COLOR_RGB2LAB)
            
            above_mask1 = cv2.inRange(above_hsv, lower_dark_green, upper_dark_green)
            above_mask2 = cv2.inRange(above_hsv, lower_light_green, upper_light_green)
            above_mask3 = cv2.inRange(above_hsv, lower_yellow_green, upper_yellow_green)
            above_mask4 = cv2.inRange(above_lab, lower_lab_green, upper_lab_green)
            
            above_mask = cv2.bitwise_or(above_mask1, above_mask2)
            above_mask = cv2.bitwise_or(above_mask, above_mask3)
            above_mask = cv2.bitwise_or(above_mask, above_mask4)
            
            above_pink_mask = cv2.inRange(above_hsv, lower_pink1, upper_pink1)
            above_mask = cv2.bitwise_and(above_mask, cv2.bitwise_not(above_pink_mask))
            
            above_mask = cv2.morphologyEx(above_mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
            above_mask = cv2.morphologyEx(above_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
            
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(above_mask, connectivity=8)
            cleaned_above = np.zeros_like(above_mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= 15:
                    cleaned_above[labels == i] = 255
            
            full_mask[plant_top:y_start, x_start:x_end] = cleaned_above
    
    return full_mask.astype(bool)

def extract_canopy_boundary_ultimate(plant_mask, soil_line_y, beaker_region):
    """
    Canopy boundary extraction - plants above soil only
    """
    h, w = plant_mask.shape
    x_start, x_end, y_start, y_end = beaker_region
    
    working_mask = plant_mask.copy()
    working_mask[soil_line_y:, :] = False
    
    width_mask = np.zeros_like(working_mask, dtype=bool)
    width_mask[:, x_start:x_end] = True
    working_mask = np.logical_and(working_mask, width_mask)
    
    boundary_points = []
    sampling_step = max(1, (x_end - x_start) // 100)
    
    for x in range(x_start, x_end, sampling_step):
        column = working_mask[:, x]
        if np.any(column):
            plant_pixels = np.where(column)[0]
            if len(plant_pixels) > 0:
                top_y = plant_pixels[0]
                boundary_points.append((x, top_y))
    
    if len(boundary_points) < 5:
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
    
    x_vals = np.array([p[0] for p in boundary_points])
    y_vals = np.array([p[1] for p in boundary_points])
    
    sorted_idx = np.argsort(x_vals)
    x_vals = x_vals[sorted_idx]
    y_vals = y_vals[sorted_idx]
    
    if len(y_vals) > 10:
        q1 = np.percentile(y_vals, 25)
        q3 = np.percentile(y_vals, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        inlier_mask = (y_vals >= lower_bound) & (y_vals <= upper_bound)
        x_vals = x_vals[inlier_mask]
        y_vals = y_vals[inlier_mask]
    
    if len(y_vals) > 10:
        window_size = min(15, len(y_vals))
        if window_size % 2 == 0:
            window_size -= 1
        if window_size >= 5:
            try:
                y_vals = savgol_filter(y_vals, window_size, 3)
            except:
                pass
    
    return x_vals, y_vals

def calculate_plant_health_ultimate(image, plant_mask):
    """
    Ultimate plant health calculation
    """
    if np.sum(plant_mask) == 0:
        return 0.0, 0.0, 0.0
    
    plant_pixels = image[plant_mask]
    
    hsv = cv2.cvtColor(plant_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    hue = hsv[:, 0]
    saturation = hsv[:, 1]
    value = hsv[:, 2]
    
    healthy_green = (hue >= 35) & (hue <= 85) & (saturation >= 40) & (value >= 40)
    mature_green = ((hue >= 25) & (hue < 35)) | ((hue > 85) & (hue <= 95)) & (saturation >= 30)
    drying = ((hue >= 15) & (hue < 25)) & (saturation >= 20) & (value >= 50)
    fresh = (hue >= 85) & (hue <= 100) & (saturation >= 30)
    
    total_pixels = len(hue)
    
    healthy_pct = np.sum(healthy_green) / total_pixels
    mature_pct = np.sum(mature_green) / total_pixels
    drying_pct = np.sum(drying) / total_pixels
    fresh_pct = np.sum(fresh) / total_pixels
    
    health_score = (healthy_pct * 0.9 + mature_pct * 0.7 + fresh_pct * 0.8 + drying_pct * 0.4)
    health_score = max(0, min(1, health_score))
    
    greenness_score = np.mean((hue >= 25) & (hue <= 95)) if np.any(hue) else 0
    colorfulness_score = np.mean(saturation) / 255.0 if np.any(saturation) else 0
    
    return health_score, greenness_score, colorfulness_score

# ===== WEB-SPECIFIC FUNCTIONS =====

def analyze_wheatgrass_web(image_data, filename):
    """
    Web version of analysis - returns results as dict with base64 images
    """
    results = {
        'success': False,
        'error': None,
        'analysis': {},
        'images': {},
        'filename': filename,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        start_time = datetime.now()
        
        # Load image from bytes
        image = load_wheat_image(image_data)
        results['analysis']['image_shape'] = image.shape
        
        # 1. ULTIMATE BEAKER DETECTION
        beaker_region = detect_beaker_ultimate(image)
        x_start, x_end, y_start, y_end, beaker_mask = beaker_region
        
        # 2. ULTIMATE SOIL LINE DETECTION
        soil_line_y, pixel_ratio = detect_soil_line_ultimate(image, (x_start, x_end, y_start, y_end))
        
        # 3. ULTIMATE PLANT DETECTION
        plant_mask = detect_plants_ultimate(image, (x_start, x_end, y_start, y_end), soil_line_y)
        
        if np.sum(plant_mask) == 0:
            results['error'] = "No plants detected!"
            return results
        
        # 4. ULTIMATE CANOPY BOUNDARY
        canopy_x, canopy_y = extract_canopy_boundary_ultimate(plant_mask, soil_line_y, (x_start, x_end, y_start, y_end))
        
        if len(canopy_x) == 0:
            results['error'] = "No canopy boundary extracted!"
            return results
        
        # 5. HEIGHT CALCULATION
        heights_px = soil_line_y - canopy_y
        heights_cm = heights_px * pixel_ratio
        heights_inch = heights_cm / 2.54
        
        valid_mask = (heights_cm > 0.5) & (heights_cm < 50)
        heights_cm = heights_cm[valid_mask]
        heights_inch = heights_inch[valid_mask]
        canopy_x = canopy_x[valid_mask]
        canopy_y = canopy_y[valid_mask]
        
        if len(heights_cm) == 0:
            results['error'] = "No valid heights calculated!"
            return results
        
        # 6. STATISTICS
        avg_height_cm = np.mean(heights_cm)
        avg_height_inch = avg_height_cm / 2.54
        max_height_cm = np.max(heights_cm)
        max_height_inch = max_height_cm / 2.54
        min_height_cm = np.min(heights_cm)
        min_height_inch = min_height_cm / 2.54
        std_height_cm = np.std(heights_cm)
        std_height_inch = std_height_cm / 2.54
        median_height_cm = np.median(heights_cm)
        median_height_inch = median_height_cm / 2.54
        
        soil_height_px = y_end - soil_line_y
        soil_height_cm = soil_height_px * pixel_ratio
        soil_height_inch = soil_height_cm / 2.54
        
        # 7. PLANT HEALTH
        health_score, greenness_score, colorfulness_score = calculate_plant_health_ultimate(image, plant_mask)
        
        elapsed_time = (datetime.now() - start_time).total_seconds()
        
        # Store results
        results['analysis'].update({
            'avg_height_cm': round(float(avg_height_cm), 3),
            'avg_height_inch': round(float(avg_height_inch), 3),
            'max_height_cm': round(float(max_height_cm), 3),
            'max_height_inch': round(float(max_height_inch), 3),
            'min_height_cm': round(float(min_height_cm), 3),
            'min_height_inch': round(float(min_height_inch), 3),
            'median_height_cm': round(float(median_height_cm), 3),
            'median_height_inch': round(float(median_height_inch), 3),
            'std_height_cm': round(float(std_height_cm), 3),
            'std_height_inch': round(float(std_height_inch), 3),
            'health_score': round(float(health_score), 3),
            'greenness_score': round(float(greenness_score), 3),
            'colorfulness_score': round(float(colorfulness_score), 3),
            'plant_pixels': int(np.sum(plant_mask)),
            'soil_height_cm': round(float(soil_height_cm), 3),
            'soil_height_inch': round(float(soil_height_inch), 3),
            'pixel_ratio': float(pixel_ratio),
            'processing_time': round(elapsed_time, 2),
            'beaker_region': {
                'x_start': int(x_start), 'x_end': int(x_end),
                'y_start': int(y_start), 'y_end': int(y_end)
            },
            'soil_line_y': int(soil_line_y),
            'canopy_points': int(len(canopy_x)),
            'sam_available': SAM_AVAILABLE,
            'sam_loaded': SAM_PREDICTOR is not None
        })
        
        # Generate visualization images as base64
        results['images'] = generate_visualization_images(
            image, plant_mask, canopy_x, canopy_y, soil_line_y,
            (x_start, x_end, y_start, y_end), heights_cm, heights_inch,
            results['analysis'], filename
        )
        
        results['success'] = True
        
    except Exception as e:
        results['error'] = str(e)
        print(f"Analysis error: {e}")
        traceback.print_exc()
    
    return results

def generate_visualization_images(image, plant_mask, canopy_x, canopy_y, soil_line_y,
                                 beaker_region, heights_cm, heights_inch, results, filename):
    """
    Generate all visualization images as base64 strings
    """
    images = {}
    
    # 1. Detection Overview
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    x_start, x_end, y_start, y_end = beaker_region
    rect = plt.Rectangle((x_start, y_start), x_end-x_start, y_end-y_start,
                        fill=False, edgecolor='yellow', linewidth=2)
    ax.add_patch(rect)
    ax.axhline(y=soil_line_y, color='brown', linewidth=3, alpha=0.8)
    if len(canopy_x) > 0:
        ax.plot(canopy_x, canopy_y, 'r-', linewidth=2, alpha=0.8)
        ax.fill_between(canopy_x, canopy_y, soil_line_y, alpha=0.15, color='green')
    ax.set_title('Detection Overview')
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    images['overview'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    # 2. Plant Mask
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(plant_mask, cmap='Greens')
    ax.axhline(y=soil_line_y, color='brown', linewidth=2, linestyle='--')
    if len(canopy_x) > 0:
        ax.plot(canopy_x, canopy_y, 'r-', linewidth=2, alpha=0.8)
    ax.set_title('Plant Mask')
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    images['plant_mask'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    # 3. Height Profile
    if len(heights_inch) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(heights_inch))
        ax.plot(x_pos, heights_inch, 'g-', linewidth=2, alpha=0.8)
        ax.fill_between(x_pos, 
                       heights_inch - results['std_height_inch']/2,
                       heights_inch + results['std_height_inch']/2,
                       alpha=0.15, color='green')
        ax.axhline(y=results['avg_height_inch'], color='red', linestyle='--',
                  linewidth=2, label=f'Avg: {results["avg_height_inch"]:.3f} in')
        ax.axhline(y=results['median_height_inch'], color='blue', linestyle=':',
                  linewidth=2, label=f'Med: {results["median_height_inch"]:.3f} in')
        ax.set_title('Height Profile (inches)')
        ax.set_xlabel('Position')
        ax.set_ylabel('Height (in)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        images['height_profile'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
    
    # 4. Height Distribution
    if len(heights_inch) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        n_bins = min(15, len(heights_inch) // 5)
        ax.hist(heights_inch, bins=n_bins, color='lightgreen', alpha=0.8,
               edgecolor='darkgreen', linewidth=0.5, density=True)
        ax.axvline(results['avg_height_inch'], color='red', linestyle='--',
                  linewidth=2, label=f'Avg: {results["avg_height_inch"]:.3f} in')
        ax.axvline(results['median_height_inch'], color='blue', linestyle=':',
                  linewidth=2, label=f'Med: {results["median_height_inch"]:.3f} in')
        ax.set_title('Height Distribution (inches)')
        ax.set_xlabel('Height (in)')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        images['height_dist'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
    
    # 5. Plant Health Visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    health_image = np.zeros_like(image)
    plant_indices = np.where(plant_mask)
    
    if len(plant_indices[0]) > 0:
        plant_pixels = image[plant_mask]
        hsv_pixels = cv2.cvtColor(plant_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        hue = hsv_pixels[:, 0]
        value = hsv_pixels[:, 2]
        
        for i in range(min(len(plant_indices[0]), len(hue))):
            y, x = plant_indices[0][i], plant_indices[1][i]
            h_val = hue[i]
            v_val = value[i]
            
            if 35 <= h_val <= 85 and v_val > 40:
                health_image[y, x] = [0, 200, 0]
            elif 25 <= h_val < 35 and v_val > 40:
                health_image[y, x] = [200, 200, 0]
            elif 85 < h_val <= 100 and v_val > 40:
                health_image[y, x] = [150, 255, 150]
            elif 15 <= h_val < 25 and v_val > 50:
                health_image[y, x] = [205, 133, 63]
            else:
                health_image[y, x] = [100, 100, 100]
    
    blended = cv2.addWeighted(image, 0.25, health_image, 0.75, 0)
    ax.imshow(blended)
    ax.axhline(y=soil_line_y, color='brown', linewidth=2, linestyle='--')
    if len(canopy_x) > 0:
        ax.plot(canopy_x, canopy_y, 'r-', linewidth=2, alpha=0.8)
    ax.set_title('Plant Health (Green=Healthy, Yellow=Mature, Brown=Drying)')
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    images['plant_health'] = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return images

# ===== FLASK ROUTES =====

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for analysis"""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400
    
    # Read image data
    image_data = file.read()
    filename = secure_filename(file.filename)
    
    # Initialize SAM on first request
    if SAM_AVAILABLE and SAM_PREDICTOR is None:
        initialize_sam()
    
    # Run analysis
    results = analyze_wheatgrass_web(image_data, filename)
    
    return jsonify(results)

@app.route('/api/batch', methods=['POST'])
def api_batch():
    """Batch process multiple images"""
    if 'images[]' not in request.files:
        return jsonify({'success': False, 'error': 'No images uploaded'}), 400
    
    files = request.files.getlist('images[]')
    results = []
    
    # Initialize SAM once
    if SAM_AVAILABLE and SAM_PREDICTOR is None:
        initialize_sam()
    
    for file in files:
        if file.filename:
            image_data = file.read()
            filename = secure_filename(file.filename)
            
            result = analyze_wheatgrass_web(image_data, filename)
            results.append(result)
    
    return jsonify({'success': True, 'results': results, 'total': len(results)})

@app.route('/api/status', methods=['GET'])
def api_status():
    """Check server status"""
    return jsonify({
        'status': 'running',
        'sam_available': SAM_AVAILABLE,
        'sam_loaded': SAM_PREDICTOR is not None,
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'max_upload_size': '16MB',
        'version': '1.0.0'
    })

@app.route('/api/download/<filename>', methods=['GET'])
def api_download(filename):
    """Download original image"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
    if os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({'error': 'File not found'}), 404

# ===== ERROR HANDLERS =====

@app.errorhandler(413)
def too_large(e):
    return jsonify({'success': False, 'error': 'File too large (max 16MB)'}), 413

@app.errorhandler(500)
def server_error(e):
    return jsonify({'success': False, 'error': 'Server error', 'details': str(e)}), 500

# ===== MAIN =====
if __name__ == '__main__':
    # Initialize SAM on startup
    if SAM_AVAILABLE:
        initialize_sam()
    
    print("=" * 60)
    print("🌾 WHEATGRASS ANALYZER WEB APPLICATION")
    print("=" * 60)
    print(f"   SAM Available: {SAM_AVAILABLE}")
    print(f"   SAM Loaded: {SAM_PREDICTOR is not None}")
    print(f"   Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"   Results folder: {app.config['RESULTS_FOLDER']}")
    print("\n📡 Starting server...")
    print("   Open: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)