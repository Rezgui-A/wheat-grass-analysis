#!/usr/bin/env python3
"""
Download SAM model automatically on Render
"""

import os
import sys
import subprocess
import urllib.request
import shutil

def download_sam_model():
    """Download SAM model to /tmp directory on Render"""
    
    # SAM model URL (Facebook's official release)
    sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    
    # Target location (Render's /tmp directory persists for the build)
    target_dir = "/tmp"
    model_path = os.path.join(target_dir, "sam_vit_b_01ec64.pth")
    
    print("🌐 Starting SAM model download...")
    print(f"   URL: {sam_url}")
    print(f"   Target: {model_path}")
    
    try:
        # Method 1: Use urllib (built-in)
        print("   Downloading using urllib...")
        
        def progress(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            if percent % 10 == 0:  # Update every 10%
                sys.stdout.write(f"\r   Progress: {percent}%")
                sys.stdout.flush()
        
        urllib.request.urlretrieve(sam_url, model_path, progress)
        print("\n   ✅ Download complete!")
        
        # Verify file size
        file_size = os.path.getsize(model_path)
        print(f"   File size: {file_size / (1024*1024):.2f} MB")
        
        if file_size < 300 * 1024 * 1024:  # Less than 300MB
            print("   ⚠️ Warning: File might be incomplete")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        
        # Fallback: Try wget if available
        try:
            print("   Trying wget as fallback...")
            subprocess.run([
                "wget", sam_url, "-O", model_path,
                "--show-progress", "--progress=bar:force"
            ], check=True)
            
            # Verify download
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path)
                print(f"   ✅ Wget download successful! Size: {file_size / (1024*1024):.2f} MB")
                return True
                
        except Exception as wget_error:
            print(f"   ❌ Wget also failed: {wget_error}")
            
        return False

if __name__ == "__main__":
    success = download_sam_model()
    sys.exit(0 if success else 1)