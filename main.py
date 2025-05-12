import cv2
import numpy as np
from rembg import remove
from PIL import Image

import argparse

# --- Set up argument parser ---
parser = argparse.ArgumentParser(description="Process an image.")
parser.add_argument("image_path", help="Path to the image file", type=str)

# --- Parse the arguments ---
args = parser.parse_args()
image_path = args.image_path

# --- Load original image ---
img = cv2.imread(image_path)
if img is not None:
    print("Image loaded successfully.")
else:
    print("Image not found.")

original = img.copy()

# --- Use rembg to get the mask (alpha channel) ---
input_pil = Image.open(image_path)
output_pil = remove(input_pil)
output_np = np.array(output_pil)

# Extract alpha channel as mask
if output_np.shape[2] == 4:
    mask = output_np[:, :, 3]
else:
    raise ValueError("Alpha channel not found.")

# Erode mask to remove fringe/pixelation at edges
kernel = np.ones((3, 3), np.uint8)
mask_eroded = cv2.erode(mask, kernel, iterations=1)
mask_closed = cv2.morphologyEx(mask_eroded, cv2.MORPH_CLOSE, kernel, iterations=1)

# Normalize and blur (feather) mask for smooth transitions
mask_normalized = mask_closed.astype(np.float32) / 255.0
mask_normalized = cv2.GaussianBlur(mask_normalized, (21, 21), 0)
mask_3ch = cv2.merge([mask_normalized]*3)

# --- Create background and foreground layers ---
foreground = original.astype(np.float32) * mask_3ch
background = original.astype(np.float32) * (1 - mask_3ch)

# --- Gamma correction on background ---
gamma = 2

lut = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype("uint8")
background_dark = cv2.LUT(background.astype(np.uint8), lut)

# --- Sharpen the foreground ---
sharpen_kernel = np.array([[0, -0.95, 0],
                           [-0.95, 4.95, -0.95],
                           [0, -0.95, 0]])
foreground_sharp = cv2.filter2D(foreground.astype(np.uint8), -1, sharpen_kernel)
foreground_smoothed = cv2.bilateralFilter(foreground_sharp, d=15, sigmaColor=75, sigmaSpace=75)

#Gamma correction on foreground
gamma = 0.9

lut = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype("uint8")
foreground_bright = cv2.LUT(foreground_sharp.astype(np.uint8), lut)


# Brighten foreground (exposure boost)
# exposure_factor = 1.2  # Try values between 1.1–1.5
# foreground_exposed = np.clip(foreground_sharp.astype(np.float32) * exposure_factor, 0, 255).astype(np.uint8)

# --- Reading Security Layer ---

overlay_img_bg = cv2.imread("./assets/SL1.jpg", cv2.IMREAD_GRAYSCALE)

# --- Create a binary mask of just the white areas ---
_, white_mask = cv2.threshold(overlay_img_bg, 240, 255, cv2.THRESH_BINARY)

# --- Create a mask for the gray areas ---
# Adjust this threshold for the range of gray values to exclude
gray_mask = cv2.inRange(overlay_img_bg, 100, 200)  # Mask for gray areas
gray_mask_3ch = cv2.merge([gray_mask] * 3)  # Convert to 3-channel

# --- Create a clean overlay (only the white areas) ---
# Set gray areas (those in the gray mask) to black (remove them completely)
clean_overlay = overlay_img_bg.copy()
clean_overlay[gray_mask == 255] = 0  # Set gray areas to 0 (black)

# --- Convert to 3-channel for blending ---
clean_overlay_3ch = cv2.merge([clean_overlay] * 3)

# Optional: feather the white mask
white_mask_float = white_mask.astype(np.float32) / 255.0
white_mask_float = cv2.GaussianBlur(white_mask_float, (15, 15), 0)

overlay_img_3ch = cv2.merge([overlay_img_bg] * 3)

# Expand to 3 channels and apply transparency (alpha blending)
white_mask_3ch = cv2.merge([white_mask_float] * 3)
white_overlay = np.full((*overlay_img_bg.shape, 3), 255).astype(np.float32)  # pure white image

# Create a vertical alpha gradient mask (top = 1.0, bottom = 0.0)
height = overlay_img_bg.shape[0]
fade_start = int(height * 0.01)  # start fading from 40% down (adjust this value)
fade_end = int(height * 0.5)

gradient = np.ones(height, dtype=np.float32)
gradient[fade_start:fade_end] = np.linspace(1.0, 0.0, fade_end - fade_start)
gradient[fade_end:] = 0.0
gradient_mask = np.repeat(gradient[:, np.newaxis], overlay_img_bg.shape[1], axis=1)  # apply across width


# Expand to 3 channels
gradient_mask_3ch = np.stack([gradient_mask]*3, axis=-1)

# Apply this gradient to your white mask alpha before blending
# Multiply white_mask_3ch (already 0–1) with the gradient
fading_mask = white_mask_3ch * gradient_mask_3ch

# Set opacity level (e.g., 0.2 for 20% visibility)
# Blend the pattern with white based on the mask and desired opacity
opacity = 0.5  # adjust for stronger or softer overlay


x_offset = 0  # horizontal position on background
y_offset = 0  # vertical position on background

# Blend white into darkened background
bg = background_dark.astype(np.uint8)
bg_h, bg_w = bg.shape[:2]
ptn_h, ptn_w = overlay_img_bg.shape[:2]


# Ensure the overlay doesn't go out of bounds
x_end = min(x_offset + ptn_w, bg_w)
y_end = min(y_offset + ptn_h, bg_h)

bg_region = bg[y_offset:y_end, x_offset:x_end]
fading_mask_region = fading_mask[y_offset:y_end, x_offset:x_end]

clean_overlay_3ch_region = clean_overlay_3ch[y_offset:y_end, x_offset:x_end]

pattern_blend = (
    clean_overlay_3ch_region * (fading_mask_region * opacity) + 
    bg_region * (1 - fading_mask_region * opacity)
)

pattern_blend = np.clip(pattern_blend, 0, 255).astype(np.uint8)

ptn_crop = pattern_blend[0: y_end - y_offset, 0: x_end - x_offset]

# Replace region in the background

bg[y_offset:y_end, x_offset:x_end] = pattern_blend.astype(np.uint8)


# Update background
background_with_overlay = bg

# Creating an image consisting of the background with its overlay and the foreground
combined = foreground_bright.astype(np.float32) + background_with_overlay.astype(np.float32) * (1 - mask_3ch)

# --- Reading Overlay for Foreground

overlay_img_fg = cv2.imread("./assets/SL2.jpg", cv2.IMREAD_GRAYSCALE)

# --- Create a binary mask of just the white areas ---
_, white_mask_fg = cv2.threshold(overlay_img_fg, 240, 255, cv2.THRESH_BINARY)

# --- Create a mask for the gray areas ---
# You can adjust this threshold for the range of gray values you want to exclude
gray_mask_fg = cv2.inRange(overlay_img_fg, 100, 200)  # Mask for gray areas
gray_mask_fg_3ch = cv2.merge([gray_mask_fg] * 3)  # Convert to 3-channel

# --- Create a clean overlay (only the white areas) ---
# Set gray areas (those in the gray mask) to black (remove them completely)
clean_overlay_fg = overlay_img_fg.copy()
clean_overlay_fg[gray_mask_fg == 255] = 0  # Set gray areas to 0 (black)

# --- Convert to 3-channel for blending ---
clean_overlay_fg_3ch = cv2.merge([clean_overlay_fg] * 3)

# Optional: feather the white mask
white_mask_float_fg = white_mask_fg.astype(np.float32) / 255.0
white_mask_float_fg = cv2.GaussianBlur(white_mask_float_fg, (15, 15), 0)

overlay_img_fg_3ch = cv2.merge([overlay_img_fg] * 3)

# Expand to 3 channels and apply transparency (alpha blending)
white_mask_fg_3ch = cv2.merge([white_mask_float_fg] * 3)
white_overlay_fg = np.full((*overlay_img_fg.shape, 3), 255).astype(np.float32)  # pure white image

# Set opacity level (e.g., 0.2 for 20% visibility)
# Blend the pattern with white based on the mask and desired opacity
opacity_fg = 0.3  
pattern_blend_fg = (
    clean_overlay_fg_3ch * (1 - white_mask_fg_3ch * opacity_fg) + 
    white_overlay_fg * (white_mask_fg_3ch * opacity_fg)
)
pattern_blend_fg = np.clip(pattern_blend_fg, 0, 255).astype(np.uint8)

fg_x_offset = 920  # horizontal position on background
fg_y_offset = 0  # vertical position on background

# Blend white into darkened background
combined_pic = combined.astype(np.uint8)
combined_h, combined_w = combined_pic.shape[:2]
fg_ptn_h, fg_ptn_w = overlay_img_fg.shape[:2]

# Ensure the overlay doesn't go out of bounds
fg_x_end = min(fg_x_offset + fg_ptn_w, combined_w)
fg_y_end = min(fg_y_offset + fg_ptn_h, combined_h)
fg_ptn_crop = pattern_blend_fg[0: fg_y_end - fg_y_offset, 0: fg_x_end - fg_x_offset]

# Replace region in the background
fg_region = combined_pic[fg_y_offset:fg_y_end, fg_x_offset:fg_x_end]
fg_blended_region = cv2.addWeighted(fg_region.astype(np.float32), 1.0, fg_ptn_crop.astype(np.float32), opacity_fg, 0)
combined_pic[fg_y_offset:fg_y_end, fg_x_offset:fg_x_end] = np.clip(fg_blended_region, 0, 255).astype(np.uint8)

# Update foreground
foreground_with_overlay = combined_pic

# --- Semi-final composition ---
semi_final_result = np.clip(combined, 0, 255).astype(np.uint8)
semi_final_region = semi_final_result[fg_y_offset:fg_y_end, fg_x_offset:fg_x_end]
semi_final_blended = cv2.addWeighted(semi_final_region.astype(np.float32), 1.0, fg_ptn_crop.astype(np.float32), opacity_fg, 0)
semi_final_result[fg_y_offset:fg_y_end, fg_x_offset:fg_x_end] = np.clip(semi_final_blended, 0, 255).astype(np.uint8)


# --- Brand Gradient

brand_color = np.array([45, 88, 250], dtype=np.uint8)

height, width = semi_final_result.shape[:2]
# height = int(height)
# width = int(width)

grad_x = np.linspace(1.0, 0.0, width, dtype=np.float32)
grad_y = np.linspace(1.0, 0.0, height, dtype=np.float32)

brand_fade_mask = np.outer(grad_y, grad_x)
brand_fade_mask_3ch = np.stack([brand_fade_mask]*3, axis = -1)

brand_layer = brand_color.astype(np.float32) * brand_fade_mask_3ch

final_result = (
    semi_final_result.astype(np.float32) * (1-brand_fade_mask_3ch) + brand_layer
)
final_result = np.clip(final_result,0,255).astype(np.uint8)




cv2.imwrite("output.png", final_result)