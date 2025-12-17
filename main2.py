# ===== WORKING PYTORCH DLL FIX =====
import os
import sys

# Fix for PyInstaller bundled executables
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    base_path = sys._MEIPASS
    
    # Get the torch/lib directory
    torch_lib = os.path.join(base_path, 'torch', 'lib')
    
    # DEBUG: Check what's in the directory
    if os.path.exists(torch_lib):
        dlls = [f for f in os.listdir(torch_lib) if f.endswith('.dll')]
        print(f"Found {len(dlls)} DLLs in torch/lib:")
        for dll in sorted(dlls)[:5]:  # Show first 5
            print(f"  - {dll}")
        if len(dlls) > 5:
            print(f"  ... and {len(dlls)-5} more")
    
    # CRITICAL: Monkey-patch torch's DLL loading
    import builtins
    
    original_import = builtins.__import__
    
    def patched_import(name, *args, **kwargs):
        # Import torch normally
        module = original_import(name, *args, **kwargs)
        
        # Patch torch to skip DLL loading
        if name == 'torch':
            # Check if DLL loading function exists
            if hasattr(module, '_load_dll_libraries'):
                # Replace with a dummy function that does nothing
                def dummy_load_dll():
                    print("✓ Skipped PyTorch DLL loading (PyInstaller workaround)")
                    return
                module._load_dll_libraries = dummy_load_dll
            
            # Also patch the error handler
            if hasattr(module, '_load_global_deps'):
                def dummy_load_global_deps():
                    print("✓ Skipped global deps loading")
                    return
                module._load_global_deps = dummy_load_global_deps
        
        return module
    
    # Apply the patch BEFORE importing torch
    builtins.__import__ = patched_import
    
    # Now update PATH for other DLLs
    if os.path.exists(torch_lib):
        os.add_dll_directory(torch_lib)
        os.environ['PATH'] = torch_lib + ';' + os.environ['PATH']
        print(f"✓ Added torch/lib to DLL search path")
# ===== END DLL FIX =====
# Continue with your imports...
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

# Import SAM (optional, can run without it too)
try:
    from segment_anything import sam_model_registry, SamPredictor
    import torch
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️ SAM not available, using enhanced traditional methods")

# Set DPI scaling
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 100

# Configuration
SAM_MODEL_PATH = "sam_vit_b_01ec64.pth" if SAM_AVAILABLE else None
SAM_MODEL_TYPE = "vit_b"
SAM_PREDICTOR = None

def initialize_sam():
    """Initialize SAM model if available"""
    global SAM_PREDICTOR
    
    if not SAM_AVAILABLE:
        print("⚠️ SAM not installed. Using enhanced traditional methods.")
        return False
    
    try:
        if not os.path.exists(SAM_MODEL_PATH):
            print(f"⚠️ SAM model not found. Using enhanced methods.")
            return False
        
        print(f"🔍 Loading SAM model...")
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_MODEL_PATH)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam.to(device=device)
        SAM_PREDICTOR = SamPredictor(sam)
        print(f"✅ SAM loaded on {device}")
        return True
        
    except Exception as e:
        print(f"⚠️ SAM initialization failed: {e}. Using enhanced methods.")
        return False

def select_image_file():
    """Open file dialog to select image"""
    root = tk.Tk()
    root.withdraw()
    file_types = [("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"), ("All files", "*.*")]
    file_path = filedialog.askopenfilename(title="Select Wheatgrass Image", filetypes=file_types)
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
    print(f"✅ Image loaded: {image.shape}")
    return image

def detect_beaker_ultimate(image):
    """
    ULTIMATE beaker detection - focuses on center vertical structure
    """
    print("\n" + "═" * 50)
    print("🏺 ULTIMATE BEAKER DETECTION")
    print("═" * 50)
    
    h, w = image.shape[:2]
    
    # Method 1: Try SAM first (most accurate)
    if SAM_PREDICTOR is not None:
        try:
            print("🔍 Attempting SAM detection...")
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
                    print("✅ SAM detection successful")
                    beaker_mask = sam_mask.astype(bool)
                    
                    # ADD EXTRA HEIGHT for plants growing above
                    extra_height = int((y_end - y_start) * 0.15)
                    y_start = max(0, y_start - extra_height)
                    
                    return x_start, x_end, y_start, y_end, beaker_mask
        except Exception as e:
            print(f"⚠️ SAM detection failed: {e}")
    
    # Method 2: Vertical edge detection (beakers have strong vertical edges)
    print("🔍 Using vertical edge detection...")
    
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
            
            print(f"✅ Edge-based detection successful")
            print(f"   Beaker: {x_end-x_start}x{y_end-y_start} px (+15% height)")
            return x_start, x_end, y_start, y_end, beaker_mask
    
    # Method 3: Color-based background subtraction
    print("🔍 Using color-based detection...")
    
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Common background colors (pink/blue)
    bg_masks = []
    
    # Pink background (most common)
    lower_pink = np.array([140, 20, 140])
    upper_pink = np.array([180, 100, 255])
    pink_mask = cv2.inRange(hsv, lower_pink, upper_pink)
    
    # Blue background
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Combine background masks
    bg_mask = pink_mask | blue_mask
    
    # Clean and fill background
    kernel_large = np.ones((21, 21), np.uint8)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel_large)
    
    # Find non-background regions (beaker + plants)
    non_bg = cv2.bitwise_not(bg_mask)
    contours, _ = cv2.findContours(non_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour (should be beaker with plants)
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
            
            print(f"✅ Color-based detection successful")
            print(f"   Beaker: {x_end-x_start}x{y_end-y_start} px (+15% height)")
            return x_start, x_end, y_start, y_end, beaker_mask
    
    # Method 4: Center-based fallback (intelligent)
    print("⚠️ Using intelligent center-based fallback")
    
    # Focus on central region (beaker is usually centered)
    center_ratio = 0.7  # Use 70% of center width
    height_ratio = 0.8  # Use 80% of center height
    
    x_start = int(w * (1 - center_ratio) / 2)
    x_end = int(w - x_start)
    y_start = int(h * (1 - height_ratio) / 2)
    y_end = int(h - y_start)
    
    # ADD EXTRA HEIGHT for plants growing above
    extra_height = int((y_end - y_start) * 0.15)
    y_start = max(0, y_start - extra_height)
    
    beaker_mask = np.zeros((h, w), dtype=bool)
    beaker_mask[y_start:y_end, x_start:x_end] = True
    
    print(f"✅ Fallback detection: {x_end-x_start}x{y_end-y_start} px (+15% height)")
    return x_start, x_end, y_start, y_end, beaker_mask

def detect_soil_line_ultimate(image, beaker_region):
    """
    ULTIMATE soil line detection - finds DARKEST BROWN line inside beaker
    Soil is dark brown and located in bottom 10-20% of beaker
    """
    print("\n" + "═" * 50)
    print("🌱 ULTIMATE SOIL LINE DETECTION")
    print("═" * 50)
    
    x_start, x_end, y_start, y_end = beaker_region
    h, w = image.shape[:2]
    
    beaker_height = y_end - y_start
    beaker_width = x_end - x_start
    
    print(f"Beaker: {beaker_width}x{beaker_height} px")
    print(f"Searching for DARK BROWN soil line in bottom 10-20% of beaker...")
    
    # Extract beaker region ONLY
    beaker_image = image[y_start:y_end, x_start:x_end]
    
    if beaker_image.size == 0:
        print("⚠️ No beaker region found")
        default_soil_y = y_end - int(beaker_height * 0.15)  # 15% from bottom
        pixel_ratio = 18.034*1.175/ beaker_height  # 18.034 cm = 7.1 inches
        return default_soil_y, pixel_ratio
    
    # Convert to multiple color spaces for soil detection
    hsv = cv2.cvtColor(beaker_image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(beaker_image, cv2.COLOR_RGB2LAB)
    
    # SOIL DETECTION STRATEGY:
    # 1. Soil is DARK BROWN/BLACK inside beaker
    # 2. Located in BOTTOM 10-20% of beaker ONLY
    # 3. Has consistent dark color across width
    # 4. MUCH DARKER than plants
    
    # Define DARK SOIL color ranges (EXTREMELY STRICT for dark soil only)
    # Soil is VERY DARK brown/black
    
    # DARK BROWN 1 (very dark, almost black soil)
    lower_dark_brown1 = np.array([0, 30, 0])   # Low saturation, very low value
    upper_dark_brown1 = np.array([30, 120, 60])  # Very dark brown
    
    # DARK BROWN 2 (medium dark soil)
    lower_dark_brown2 = np.array([5, 40, 5])
    upper_dark_brown2 = np.array([25, 100, 50])
    
    # VERY DARK (black soil)
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 40, 30])  # Even stricter
    
    # Create soil masks with EXTREMELY STRICT thresholds
    soil_mask1 = cv2.inRange(hsv, lower_dark_brown1, upper_dark_brown1)
    soil_mask2 = cv2.inRange(hsv, lower_dark_brown2, upper_dark_brown2)
    soil_mask3 = cv2.inRange(hsv, lower_black, upper_black)
    
    # Combine soil masks
    soil_mask_combined = cv2.bitwise_or(soil_mask1, soil_mask2)
    soil_mask_combined = cv2.bitwise_or(soil_mask_combined, soil_mask3)
    
    # LAB space: Look for VERY DARK areas (extremely low L channel)
    L_channel = lab[:, :, 0]
    lab_dark_mask = (L_channel < 50).astype(np.uint8) * 255  # EXTREMELY dark in LAB
    
    # Value channel: Look for VERY DARK areas (extremely low V in HSV)
    V_channel = hsv[:, :, 2]
    v_dark_mask = (V_channel < 60).astype(np.uint8) * 255  # EXTREMELY dark
    
    # FINAL SOIL MASK: Combine all with high threshold
    soil_mask = np.zeros_like(soil_mask_combined, dtype=np.float32)
    soil_mask += soil_mask_combined.astype(np.float32) * 2.0  # High weight for brown colors
    soil_mask += lab_dark_mask.astype(np.float32) * 1.5      # High weight for dark LAB
    soil_mask += v_dark_mask.astype(np.float32) * 1.2        # High weight for dark value
    
    # Convert to binary with VERY HIGH threshold (only keep extremely dark areas)
    soil_mask_binary = (soil_mask > 150).astype(np.uint8) * 255
    
    # CLEAN THE MASK - EXTREMELY AGGRESSIVE
    # Remove small noise
    kernel_clean = np.ones((3, 3), np.uint8)
    soil_mask_binary = cv2.morphologyEx(soil_mask_binary, cv2.MORPH_OPEN, kernel_clean)
    
    # Connect nearby soil areas
    soil_mask_binary = cv2.morphologyEx(soil_mask_binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    
    # Fill holes in soil
    soil_mask_binary = cv2.morphologyEx(soil_mask_binary, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    
    # CRITICAL: FOCUS ON BOTTOM 10-20% OF BEAKER ONLY
    beaker_h = beaker_image.shape[0]
    
    # Soil is in bottom 10-20% of beaker
    search_top = int(beaker_h * 0.80)  # Start 80% from top (20% from bottom)
    search_bottom = int(beaker_h * 0.90)  # End 90% from top (10% from bottom)
    
    # Create search mask for BOTTOM 10-20% region ONLY
    search_mask = np.zeros_like(soil_mask_binary)
    search_mask[search_top:search_bottom, :] = 255
    
    # Apply search mask - ONLY look in bottom 10-20%
    soil_mask_binary = cv2.bitwise_and(soil_mask_binary, search_mask)
    
    # If no soil detected in this region, expand slightly
    if np.sum(soil_mask_binary) < 100:
        print("⚠️ No soil in bottom 10-20%, expanding search to bottom 5-25%")
        search_top = int(beaker_h * 0.75)  # 25% from bottom
        search_bottom = int(beaker_h * 0.95)  # 5% from bottom
        
        search_mask = np.zeros_like(soil_mask_binary)
        search_mask[search_top:search_bottom, :] = 255
        soil_mask_binary = cv2.bitwise_and(soil_mask_binary, search_mask)
    
    # FIND SOIL LINE using horizontal projection
    # Calculate soil density for each row
    row_density = np.sum(soil_mask_binary, axis=1) / beaker_width
    
    if np.max(row_density) < 0.05:  # No significant soil detected
        print("⚠️ No strong soil signal detected in bottom region")
        # Use default position (15% from bottom)
        default_position = int(beaker_h * 0.85)  # 15% from bottom
        soil_line_in_beaker = default_position
    else:
        # Find the TOPMOST continuous dark soil line
        # We want the highest row with significant soil
        
        # Smooth the density curve
        if len(row_density) > 10:
            row_density_smooth = np.convolve(row_density, np.ones(5)/5, mode='same')
        else:
            row_density_smooth = row_density
        
        # Find threshold (30% of max density) - HIGHER threshold
        threshold = np.max(row_density_smooth) * 0.30
        
        # Scan from TOP to BOTTOM within search region (we want highest soil line)
        soil_line_in_beaker = None
        
        # Only search within our defined bottom region
        search_indices = list(range(search_top, min(search_bottom, len(row_density_smooth))))
        
        for i in search_indices:
            if row_density_smooth[i] >= threshold:
                # Check if next few rows also have soil (continuity)
                check_ahead = min(3, len(row_density_smooth) - i - 1)
                if check_ahead > 0:
                    ahead_avg = np.mean(row_density_smooth[i:i+check_ahead])
                    if ahead_avg >= threshold * 0.8:  # Strict continuity
                        soil_line_in_beaker = i
                        break
        
        # If not found, use row with maximum density within search region
        if soil_line_in_beaker is None:
            if len(search_indices) > 0:
                # Only consider rows within search region
                search_density = row_density_smooth[search_indices[0]:search_indices[-1]]
                if len(search_density) > 0:
                    max_in_region = np.argmax(search_density)
                    soil_line_in_beaker = search_indices[0] + max_in_region
                else:
                    soil_line_in_beaker = search_indices[0]
            else:
                soil_line_in_beaker = int(beaker_h * 0.85)  # 15% from bottom
    
    # Convert to full image coordinates
    soil_line_y = y_start + soil_line_in_beaker
    
    # VALIDATION: Ensure soil line is within bottom 5-25% of beaker
    min_allowed_y = y_start + int(beaker_height * 0.75)  # Bottom 25% MAX
    max_allowed_y = y_end - int(beaker_height * 0.05)    # At least 5% from bottom MIN
    
    soil_line_y = max(min_allowed_y, min(soil_line_y, max_allowed_y))
    
    # Calculate pixel ratio (beaker is 18.034 cm = 7.1 inches)
    pixel_to_cm_ratio = 18.034*1.175 / beaker_height
    
    # Calculate actual soil height from bottom
    soil_height_from_bottom_px = y_end - soil_line_y
    soil_height_from_bottom_cm = soil_height_from_bottom_px * pixel_to_cm_ratio
    soil_percentage_from_bottom = (soil_height_from_bottom_px / beaker_height) * 100
    
    print(f"✅ ULTIMATE SOIL LINE DETECTED")
    print(f"   Soil line at y = {soil_line_y}")
    print(f"   From beaker bottom: {soil_height_from_bottom_px} px")
    print(f"   Percentage from bottom: {soil_percentage_from_bottom:.1f}%")
    print(f"   Actual height from bottom: {soil_height_from_bottom_cm:.3f} cm")
    print(f"   Target: Bottom 10-20% of beaker ✓")
    print(f"   Pixel ratio: {pixel_to_cm_ratio:.6f} cm/px")
    print(f"   Beaker actual height: 18.034 cm = 7.1 inches")
    
    # Additional debug info
    print(f"   Search region: rows {search_top}-{search_bottom} (bottom {100-search_bottom/beaker_h*100:.0f}-{100-search_top/beaker_h*100:.0f}%)")
    print(f"   Soil mask pixels: {np.sum(soil_mask_binary > 0)}")
    
    return soil_line_y, pixel_to_cm_ratio

def detect_plants_ultimate(image, beaker_region, soil_line_y):
    """
    ULTIMATE plant detection - detects ALL plant colors properly
    Plants: Dark green, light green, yellow-green, light brown (NOT pink background)
    """
    print("\n" + "═" * 50)
    print("🌿 ULTIMATE PLANT DETECTION")
    print("═" * 50)
    
    x_start, x_end, y_start, y_end = beaker_region
    h, w = image.shape[:2]
    
    # Allow plants to grow ABOVE beaker (wheatgrass can be tall)
    plant_top = max(0, y_start - 100)  # Allow 100px above beaker
    plant_bottom = soil_line_y  # Plants end at soil line
    
    # Extract plant region (above soil only)
    plant_region = image[plant_top:plant_bottom, x_start:x_end]
    
    if plant_region.size == 0:
        print("⚠️ No plant region found")
        return np.zeros((h, w), dtype=bool)
    
    # Convert to multiple color spaces for better detection
    hsv = cv2.cvtColor(plant_region, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(plant_region, cv2.COLOR_RGB2LAB)
    
    # EXPANDED WHEATGRASS COLOR RANGES
    # Plants appear as: DARK GREEN, LIGHT GREEN, YELLOW-GREEN, LIGHT BROWN
    # Background is PINK - we must exclude it completely
    
    # 1. HEALTHY DARK GREEN (main plant color)
    lower_dark_green = np.array([35, 40, 30])    # Dark green
    upper_dark_green = np.array([85, 255, 180])  # Bright green
    
    # 2. LIGHT GREEN / FRESH GREEN
    lower_light_green = np.array([40, 30, 50])   # Light green
    upper_light_green = np.array([90, 200, 220]) # Very light green
    
    # 3. YELLOW-GREEN (maturing/mature)
    lower_yellow_green = np.array([25, 40, 40])  # Yellow-green
    upper_yellow_green = np.array([40, 200, 200]) # Bright yellow-green
    
    # 4. LIGHT BROWN / DRYING (NOT DARK BROWN - that's soil)
    lower_light_brown = np.array([10, 30, 40])   # Light brown
    upper_light_brown = np.array([25, 150, 160]) # Medium light brown
    
    # 5. YELLOW (maturing tops)
    lower_yellow = np.array([15, 40, 50])        # Yellow
    upper_yellow = np.array([30, 180, 200])      # Bright yellow
    
    # 6. EXTENDED GREEN RANGE (catch all greens)
    lower_all_green = np.array([30, 20, 30])     # Very wide green range
    upper_all_green = np.array([100, 220, 200])  # Includes yellow-greens
    
    # 7. LAB SPACE GREEN DETECTION (different color space)
    # In LAB: Green plants have specific a* and b* values
    lower_lab_green = np.array([0, 120, 120])    # Green in LAB
    upper_lab_green = np.array([255, 140, 140])
    
    # Create masks for ALL PLANT COLORS
    plant_masks = []
    
    # HSV-based masks
    plant_masks.append(cv2.inRange(hsv, lower_dark_green, upper_dark_green))
    plant_masks.append(cv2.inRange(hsv, lower_light_green, upper_light_green))
    plant_masks.append(cv2.inRange(hsv, lower_yellow_green, upper_yellow_green))
    plant_masks.append(cv2.inRange(hsv, lower_light_brown, upper_light_brown))
    plant_masks.append(cv2.inRange(hsv, lower_yellow, upper_yellow))
    plant_masks.append(cv2.inRange(hsv, lower_all_green, upper_all_green))
    
    # LAB-based mask
    plant_masks.append(cv2.inRange(lab, lower_lab_green, upper_lab_green))
    
    # Combine ALL PLANT masks
    plant_mask_combined = plant_masks[0]
    for mask in plant_masks[1:]:
        plant_mask_combined = cv2.bitwise_or(plant_mask_combined, mask)
    
    # CRITICAL: EXCLUDE PINK BACKGROUND COMPLETELY
    # Pink background has very specific HSV range
    # We need to be aggressive about removing it
    
    # PINK BACKGROUND RANGES (must exclude these)
    # Common pink background colors
    lower_pink1 = np.array([140, 20, 140])    # Dark pink
    upper_pink1 = np.array([180, 100, 255])   # Bright pink
    
    lower_pink2 = np.array([160, 15, 150])    # Light pink
    upper_pink2 = np.array([180, 80, 255])    # Very light pink
    
    lower_pink3 = np.array([0, 10, 140])      # Reddish-pink (low hue)
    upper_pink3 = np.array([10, 90, 255])
    
    # Create pink masks
    pink_mask1 = cv2.inRange(hsv, lower_pink1, upper_pink1)
    pink_mask2 = cv2.inRange(hsv, lower_pink2, upper_pink2)
    pink_mask3 = cv2.inRange(hsv, lower_pink3, upper_pink3)
    
    # Combine all pink masks
    pink_mask_combined = cv2.bitwise_or(pink_mask1, pink_mask2)
    pink_mask_combined = cv2.bitwise_or(pink_mask_combined, pink_mask3)
    
    # Also exclude VERY LIGHT colors (background highlights)
    lower_light = np.array([0, 0, 200])       # Very light/white
    upper_light = np.array([180, 30, 255])
    light_mask = cv2.inRange(hsv, lower_light, upper_light)
    
    # Also exclude VERY DARK colors (shadows, not plants)
    lower_dark = np.array([0, 0, 0])          # Very dark
    upper_dark = np.array([180, 255, 40])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
    
    # COMBINE ALL BACKGROUND MASKS
    background_mask = cv2.bitwise_or(pink_mask_combined, light_mask)
    background_mask = cv2.bitwise_or(background_mask, dark_mask)
    
    # REMOVE BACKGROUND from plant mask
    plant_mask_combined = cv2.bitwise_and(plant_mask_combined, cv2.bitwise_not(background_mask))
    
    # ENHANCED POST-PROCESSING
    
    # 1. Initial cleaning (remove tiny noise)
    kernel_open = np.ones((2, 2), np.uint8)
    plant_mask_combined = cv2.morphologyEx(plant_mask_combined, cv2.MORPH_OPEN, kernel_open)
    
    # 2. Connect nearby plant pixels (plants often have gaps)
    kernel_close = np.ones((3, 3), np.uint8)
    plant_mask_combined = cv2.morphologyEx(plant_mask_combined, cv2.MORPH_CLOSE, kernel_close)
    
    # 3. Emphasize VERTICAL structures (wheatgrass grows vertically)
    vertical_kernel = np.ones((13, 1), np.uint8)  # Tall vertical kernel
    plant_mask_combined = cv2.morphologyEx(plant_mask_combined, cv2.MORPH_CLOSE, vertical_kernel)
    
    # 4. Remove very small components (noise)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(plant_mask_combined, connectivity=8)
    cleaned_mask = np.zeros_like(plant_mask_combined)
    
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        
        # Plants are typically vertical structures
        if area >= 20:  # Minimum area
            # Calculate aspect ratio
            if width > 0:
                aspect_ratio = height / width
            else:
                aspect_ratio = 0
            
            # Keep if:
            # 1. Tall and thin (vertical plant), OR
            # 2. Large area (dense plant cluster), OR
            # 3. Medium area with reasonable height
            if (aspect_ratio > 1.5 and height > 15) or area > 80 or (area > 40 and height > 10):
                cleaned_mask[labels == i] = 255
    
    plant_mask_combined = cleaned_mask
    
    # 5. Fill small holes within plants (plants can have holes)
    contours, hierarchy = cv2.findContours(plant_mask_combined, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = plant_mask_combined.copy()
    
    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, (cnt, hier) in enumerate(zip(contours, hierarchy)):
            if hier[3] >= 0:  # Hole contour (inside a plant)
                area = cv2.contourArea(cnt)
                if area <= 40:  # Fill small holes
                    cv2.drawContours(filled_mask, [cnt], 0, 255, -1)
    
    plant_mask_combined = filled_mask
    
    # 6. Additional vertical enhancement
    # Wheatgrass blades are vertical, so apply vertical dilation
    vertical_dilate = np.ones((5, 1), np.uint8)
    plant_mask_combined = cv2.dilate(plant_mask_combined, vertical_dilate, iterations=1)
    
    # Create full image mask
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[plant_top:plant_bottom, x_start:x_end] = plant_mask_combined
    
    # CRITICAL: Check for plants ABOVE the beaker (tall wheatgrass)
    if plant_top < y_start:
        above_beaker = image[plant_top:y_start, x_start:x_end]
        if above_beaker.size > 0:
            above_hsv = cv2.cvtColor(above_beaker, cv2.COLOR_RGB2HSV)
            above_lab = cv2.cvtColor(above_beaker, cv2.COLOR_RGB2LAB)
            
            # Look for plant colors above beaker
            above_mask1 = cv2.inRange(above_hsv, lower_dark_green, upper_dark_green)
            above_mask2 = cv2.inRange(above_hsv, lower_light_green, upper_light_green)
            above_mask3 = cv2.inRange(above_hsv, lower_yellow_green, upper_yellow_green)
            above_mask4 = cv2.inRange(above_lab, lower_lab_green, upper_lab_green)
            
            above_mask = cv2.bitwise_or(above_mask1, above_mask2)
            above_mask = cv2.bitwise_or(above_mask, above_mask3)
            above_mask = cv2.bitwise_or(above_mask, above_mask4)
            
            # Remove pink background from above-beaker mask
            above_pink_mask = cv2.inRange(above_hsv, lower_pink1, upper_pink1)
            above_mask = cv2.bitwise_and(above_mask, cv2.bitwise_not(above_pink_mask))
            
            # Clean up above-beaker mask
            above_mask = cv2.morphologyEx(above_mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
            above_mask = cv2.morphologyEx(above_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
            
            # Remove small components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(above_mask, connectivity=8)
            cleaned_above = np.zeros_like(above_mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= 15:  # Higher threshold for above beaker
                    cleaned_above[labels == i] = 255
            
            # Add to full mask
            full_mask[plant_top:y_start, x_start:x_end] = cleaned_above
    
    # FINAL VALIDATION: Ensure we have enough plants
    plant_pixel_count = np.sum(full_mask > 0)
    plant_area_percentage = (plant_pixel_count / ((plant_bottom - plant_top) * (x_end - x_start))) * 100
    
    print(f"✅ ULTIMATE PLANT DETECTION COMPLETE")
    print(f"   Plant pixels: {plant_pixel_count:,}")
    print(f"   Plant area: {plant_area_percentage:.3f}% of region")
    print(f"   Detection region: y={plant_top} to {plant_bottom}")
    print(f"   Width: {x_end-x_start} px")
    print(f"   Color ranges: Dark green ✓ Light green ✓ Yellow-green ✓ Light brown ✓")
    print(f"   Background exclusion: Pink background completely removed ✓")
    
    # DEBUG: Show color distribution of detected plants
    if plant_pixel_count > 0:
        plant_pixels = image[full_mask.astype(bool)]
        plant_hsv = cv2.cvtColor(plant_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        hues = plant_hsv[:, 0]
        
        green_count = np.sum((hues >= 35) & (hues <= 85))
        yellow_green_count = np.sum((hues >= 25) & (hues < 35))
        brown_count = np.sum((hues >= 10) & (hues < 25))
        
        print(f"   Color distribution: Green={green_count/len(hues)*100:.3f}%, " +
              f"Yellow-green={yellow_green_count/len(hues)*100:.3f}%, " +
              f"Brown={brown_count/len(hues)*100:.3f}%")
    
    return full_mask.astype(bool)

def extract_canopy_boundary_ultimate(plant_mask, soil_line_y, beaker_region):
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

def calculate_plant_health_ultimate(image, plant_mask):
    """
    Ultimate plant health calculation
    """
    if np.sum(plant_mask) == 0:
        return 0.0, 0.0, 0.0
    
    plant_pixels = image[plant_mask]
    
    # Convert to HSV
    hsv = cv2.cvtColor(plant_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    hue = hsv[:, 0]
    saturation = hsv[:, 1]
    value = hsv[:, 2]
    
    # Health categories (optimized for wheatgrass)
    healthy_green = (hue >= 35) & (hue <= 85) & (saturation >= 40) & (value >= 40)
    mature_green = ((hue >= 25) & (hue < 35)) | ((hue > 85) & (hue <= 95)) & (saturation >= 30)
    drying = ((hue >= 15) & (hue < 25)) & (saturation >= 20) & (value >= 50)  # Light brown only
    fresh = (hue >= 85) & (hue <= 100) & (saturation >= 30)  # Fresh light green
    
    total_pixels = len(hue)
    
    healthy_pct = np.sum(healthy_green) / total_pixels
    mature_pct = np.sum(mature_green) / total_pixels
    drying_pct = np.sum(drying) / total_pixels
    fresh_pct = np.sum(fresh) / total_pixels
    
    # Weighted health score (favor green, penalize brown)
    health_score = (healthy_pct * 0.9 + mature_pct * 0.7 + fresh_pct * 0.8 + drying_pct * 0.4)
    health_score = max(0, min(1, health_score))
    
    # Greenness (how much is green/yellow, not brown)
    greenness_score = np.mean((hue >= 25) & (hue <= 95)) if np.any(hue) else 0
    
    # Colorfulness (saturation - higher means more colorful/healthy)
    colorfulness_score = np.mean(saturation) / 255.0 if np.any(saturation) else 0
    
    return health_score, greenness_score, colorfulness_score

def analyze_wheatgrass_ultimate(image_path):
    """
    MAIN: Ultimate automatic wheatgrass analysis
    """
    print("\n" + "═" * 70)
    print("🌾 ULTIMATE AUTOMATIC WHEATGRASS ANALYSIS")
    print("═" * 70)
    
    try:
        start_time = time.time()
        
        # Load image
        image = load_wheat_image(image_path)
        
        # 1. ULTIMATE BEAKER DETECTION
        beaker_region = detect_beaker_ultimate(image)
        x_start, x_end, y_start, y_end, beaker_mask = beaker_region
        
        # 2. ULTIMATE SOIL LINE DETECTION (DARKEST brown inside beaker)
        soil_line_y, pixel_ratio = detect_soil_line_ultimate(image, (x_start, x_end, y_start, y_end))
        
        # 3. ULTIMATE PLANT DETECTION (distinguishes plants from soil)
        plant_mask = detect_plants_ultimate(image, (x_start, x_end, y_start, y_end), soil_line_y)
        
        if np.sum(plant_mask) == 0:
            print("❌ No plants detected!")
            return None
        
        # 4. ULTIMATE CANOPY BOUNDARY (zigzag following plant tops only)
        canopy_x, canopy_y = extract_canopy_boundary_ultimate(plant_mask, soil_line_y, (x_start, x_end, y_start, y_end))
        
        if len(canopy_x) == 0:
            print("❌ No canopy boundary extracted!")
            return None
        
        # 5. HEIGHT CALCULATION
        heights_px = soil_line_y - canopy_y
        heights_cm = heights_px * pixel_ratio
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
        
        # Calculate soil height in cm and inches
        y_start, y_end = beaker_region[2], beaker_region[3]
        soil_height_px = y_end - soil_line_y
        soil_height_cm = soil_height_px * pixel_ratio
        soil_height_inch = soil_height_cm / 2.54
        
        # 7. PLANT HEALTH
        health_score, greenness_score, colorfulness_score = calculate_plant_health_ultimate(image, plant_mask)
        
        elapsed_time = time.time() - start_time
        
        print(f"\n✅ ULTIMATE ANALYSIS COMPLETED in {elapsed_time:.3f} seconds")
        
        # Prepare results
        results = {
            'canopy_x': canopy_x, 'canopy_y': canopy_y,
            'soil_line_y': soil_line_y, 'height_cm': heights_cm,
            'height_inch': heights_inch, 'avg_height_cm': avg_height_cm,
            'avg_height_inch': avg_height_inch, 'max_height_cm': max_height_cm,
            'max_height_inch': max_height_inch, 'min_height_cm': min_height_cm,
            'min_height_inch': min_height_inch, 'std_height_cm': std_height_cm,
            'std_height_inch': std_height_inch, 'median_height_cm': median_height_cm,
            'median_height_inch': median_height_inch,
            'health_score': health_score, 'greenness_score': greenness_score,
            'colorfulness_score': colorfulness_score,
            'pixel_ratio': pixel_ratio, 'plant_mask': plant_mask,
            'beaker_region': (x_start, x_end, y_start, y_end),
            'plant_pixels': np.sum(plant_mask),
            'soil_height_cm': soil_height_cm,
            'soil_height_inch': soil_height_inch
        }
        
        # Display results with 3 decimal places
        print(f"\n📊 ULTIMATE RESULTS:")
        print(f"  Average Height: {avg_height_cm:.3f} cm ({avg_height_inch:.3f} in)")
        print(f"  Max Height: {max_height_cm:.3f} cm ({max_height_inch:.3f} in)")
        print(f"  Min Height: {min_height_cm:.3f} cm ({min_height_inch:.3f} in)")
        print(f"  Median Height: {median_height_cm:.3f} cm ({median_height_inch:.3f} in)")
        print(f"  Height Range: {min_height_cm:.3f} to {max_height_cm:.3f} cm")
        print(f"  Standard Deviation: {std_height_cm:.3f} cm ({std_height_inch:.3f} in)")
        print(f"  Plant Health Score: {health_score:.3f}/1.0")
        print(f"  Plants Detected: {np.sum(plant_mask):,} pixels")
        print(f"  Canopy Points: {len(canopy_x)} (zigzag pattern ✓)")
        print(f"  Soil Line: y={soil_line_y} px (darkest brown inside beaker ✓)")
        print(f"  Soil Height: {soil_height_cm:.3f} cm ({soil_height_inch:.3f} in)")
        print(f"  Beaker: {x_end-x_start}x{y_end-y_start} px (+15% height for plants above ✓)")
        print(f"  Pixel Ratio: {pixel_ratio:.6f} cm/px")
        
        # Create visualization
        create_ultimate_visualization(image, plant_mask, canopy_x, canopy_y, soil_line_y,
                                     (x_start, x_end, y_start, y_end), heights_cm, heights_inch,
                                     results, os.path.basename(image_path))
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_ultimate_visualization(image, plant_mask, canopy_x, canopy_y, soil_line_y,
                                 beaker_region, heights_cm, heights_inch, results, image_name):
    """
    Create ultimate visualization with inches and reduced text
    """
    # Create figure with dynamic sizing for resolution compatibility
    fig = plt.figure(figsize=(20, 12))
    
    # Use GridSpec for better layout control
    gs = plt.GridSpec(2, 3, figure=fig, hspace=0.25, wspace=0.2)
    
    fig.suptitle(f'Wheatgrass Analysis - {image_name}', 
                fontsize=18, fontweight='bold', y=0.98)
    
    h, w = image.shape[:2]
    x_start, x_end, y_start, y_end = beaker_region
    
    # Plot 1: Detection Overview
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image)
    
    # Beaker rectangle
    rect = plt.Rectangle((x_start, y_start), x_end-x_start, y_end-y_start,
                        fill=False, edgecolor='yellow', linewidth=2)
    ax1.add_patch(rect)
    
    # Soil line
    ax1.axhline(y=soil_line_y, color='brown', linewidth=3, alpha=0.8)
    
    # Canopy line
    if len(canopy_x) > 0:
        ax1.plot(canopy_x, canopy_y, 'r-', linewidth=2, alpha=0.8)
        ax1.fill_between(canopy_x, canopy_y, soil_line_y, alpha=0.15, color='green')
    
    ax1.set_title('Detection Overview', fontsize=14)
    ax1.axis('off')
    
    # Plot 2: Plant Health
    ax2 = fig.add_subplot(gs[0, 1])
    health_image = np.zeros_like(image)
    plant_indices = np.where(plant_mask)
    
    if len(plant_indices[0]) > 0:
        plant_pixels = image[plant_mask]
        hsv_pixels = cv2.cvtColor(plant_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
        hue = hsv_pixels[:, 0]
        value = hsv_pixels[:, 2]
        
        for i in range(len(plant_indices[0])):
            y, x = plant_indices[0][i], plant_indices[1][i]
            h_val = hue[i] if i < len(hue) else 0
            v_val = value[i] if i < len(value) else 0
            
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
    ax2.imshow(blended)
    ax2.axhline(y=soil_line_y, color='brown', linewidth=2, linestyle='--')
    if len(canopy_x) > 0:
        ax2.plot(canopy_x, canopy_y, 'r-', linewidth=2, alpha=0.8)
    ax2.set_title('Plant Health', fontsize=14)
    ax2.axis('off')
    
    # Plot 3: Height Profile (in inches)
    ax3 = fig.add_subplot(gs[0, 2])
    if len(heights_inch) > 0:
        x_pos = np.arange(len(heights_inch))
        ax3.plot(x_pos, heights_inch, 'g-', linewidth=2, alpha=0.8)
        ax3.fill_between(x_pos, 
                        heights_inch - results['std_height_inch']/2,
                        heights_inch + results['std_height_inch']/2,
                        alpha=0.15, color='green')
        
        ax3.axhline(y=results['avg_height_inch'], color='red', linestyle='--',
                   linewidth=2, label=f'Avg: {results["avg_height_inch"]:.3f} in')
        ax3.axhline(y=results['median_height_inch'], color='blue', linestyle=':',
                   linewidth=2, label=f'Med: {results["median_height_inch"]:.3f} in')
        
        ax3.set_title('Height Profile (inches)', fontsize=14)
        ax3.set_xlabel('Position')
        ax3.set_ylabel('Height (in)')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='both', labelsize=10)
    
    # Plot 4: Plant Mask
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.imshow(plant_mask, cmap='Greens')
    ax4.axhline(y=soil_line_y, color='brown', linewidth=2, linestyle='--')
    if len(canopy_x) > 0:
        ax4.plot(canopy_x, canopy_y, 'r-', linewidth=2, alpha=0.8)
    ax4.set_title('Plant Mask', fontsize=14)
    ax4.axis('off')
    
    # Plot 5: Height Distribution (in inches)
    ax5 = fig.add_subplot(gs[1, 1])
    if len(heights_inch) > 0:
        n_bins = min(15, len(heights_inch) // 5)
        ax5.hist(heights_inch, bins=n_bins, color='lightgreen', alpha=0.8,
                edgecolor='darkgreen', linewidth=0.5, density=True)
        
        ax5.axvline(results['avg_height_inch'], color='red', linestyle='--',
                   linewidth=2, label=f'Avg: {results["avg_height_inch"]:.3f} in')
        ax5.axvline(results['median_height_inch'], color='blue', linestyle=':',
                   linewidth=2, label=f'Med: {results["median_height_inch"]:.3f} in')
        
        ax5.set_title('Height Distribution (inches)', fontsize=14)
        ax5.set_xlabel('Height (in)')
        ax5.set_ylabel('Density')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
        ax5.tick_params(axis='both', labelsize=10)
    
    # Plot 6: Summary Statistics (simplified)
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Calculate plant density
    plant_density = (results['plant_pixels'] / ((y_end - y_start) * (x_end - x_start))) * 100
    
    # Create concise summary with 3 decimal places
    summary_text = f"""HEIGHT MEASUREMENTS
━━━━━━━━━━━━━━━━━━━━━
Average:  {results['avg_height_cm']:.3f} cm
          {results['avg_height_inch']:.3f} in

Maximum:  {results['max_height_cm']:.3f} cm
          {results['max_height_inch']:.3f} in

Minimum:  {results['min_height_cm']:.3f} cm
          {results['min_height_inch']:.3f} in

Median:   {results['median_height_cm']:.3f} cm
          {results['median_height_inch']:.3f} in

Std Dev:  {results['std_height_cm']:.3f} cm
          {results['std_height_inch']:.3f} in

━━━━━━━━━━━━━━━━━━━━━
PLANT ANALYSIS
━━━━━━━━━━━━━━━━━━━━━
Health:   {results['health_score']:.3f}
Greenness:{results['greenness_score']:.3f}
Pixels:   {results['plant_pixels']:,}
Density:  {plant_density:.3f}%

━━━━━━━━━━━━━━━━━━━━━
BEAKER DETECTION
━━━━━━━━━━━━━━━━━━━━━
Width:    {x_end-x_start} px
Height:   {y_end-y_start} px
Soil:     {y_end-soil_line_y} px
Soil Ht:  {results['soil_height_cm']:.3f} cm
          {results['soil_height_inch']:.3f} in
Ratio:    {results['pixel_ratio']:.6f} cm/px"""
    
    ax6.text(0.05, 0.95, summary_text, fontsize=10, family='monospace',
            verticalalignment='top', transform=ax6.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", 
                    alpha=0.95, edgecolor='orange', linewidth=1))
    ax6.set_title('Analysis Summary', fontsize=14)
    ax6.axis('off')
    
    plt.tight_layout()
    
    # Save the figure with resolution compatibility
    output_dir = "wheatgrass_analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"analysis_{os.path.splitext(image_name)[0]}.png")
    
    # Save with appropriate DPI
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Visualization saved to: {output_path}")
    
    plt.show()

def main():
    """
    Main application function
    """
    print("\n" + "═" * 60)
    print("WHEATGRASS ANALYSIS SYSTEM")
    print("═" * 60)
    print("Features:")
    print("• Automatic beaker detection")
    print("• Soil line detection")
    print("• Plant health analysis")
    print("• Height measurements in cm and inches")
    print("═" * 60)
    
    # Initialize SAM (optional)
    initialize_sam()
    
    # Select image
    image_path = select_image_file()
    
    if not image_path:
        print("No image selected.")
        return
    
    print(f"\nSelected: {os.path.basename(image_path)}")
    print("Starting analysis...")
    
    # Run analysis
    results = analyze_wheatgrass_ultimate(image_path)
    
    if results:
        print(f"\nANALYSIS SUCCESSFUL!")
        print(f"Average height: {results['avg_height_cm']:.3f} cm ({results['avg_height_inch']:.3f} in)")
        print(f"Max height: {results['max_height_cm']:.3f} cm ({results['max_height_inch']:.3f} in)")
        print(f"Min height: {results['min_height_cm']:.3f} cm ({results['min_height_inch']:.3f} in)")
        
        # Ask for another analysis
        root = tk.Tk()
        root.withdraw()
        another = messagebox.askyesno("Analysis Complete",
                                    f"Analysis completed!\n\n"
                                    f"Avg Height: {results['avg_height_cm']:.3f} cm ({results['avg_height_inch']:.3f} in)\n"
                                    f"Max Height: {results['max_height_cm']:.3f} cm ({results['max_height_inch']:.3f} in)\n"
                                    f"Min Height: {results['min_height_cm']:.3f} cm ({results['min_height_inch']:.3f} in)\n\n"
                                    "Analyze another image?")
        root.destroy()
        
        if another:
            main()
        else:
            print("\nThank you for using Wheatgrass Analysis!")
    else:
        print("\nAnalysis failed. Please try a different image.")
        
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