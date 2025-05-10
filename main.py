import cv2
import numpy as np
from rembg import remove
from PIL import Image

# --- Load original image ---
img = cv2.imread("ax.png")
if img is not None:
    print("Image loaded successfully.")
else:
    print("Image not found.")

original = img.copy()

# --- Use rembg to get the mask (alpha channel) ---
input_pil = Image.open("ax.jpg")
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

# Normalize and blur (feather) mask for smooth transitions
mask_normalized = mask_eroded.astype(np.float32) / 255.0
mask_normalized = cv2.GaussianBlur(mask_normalized, (11, 11), 0)
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

# #Gamma correction on foreground
# gamma = 0.9

# lut = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype("uint8")
# foreground_bright = cv2.LUT(foreground_sharp.astype(np.uint8), lut)


# Brighten foreground (exposure boost)
# exposure_factor = 1.2  # Try values between 1.1–1.5
# foreground_exposed = np.clip(foreground_sharp.astype(np.float32) * exposure_factor, 0, 255).astype(np.uint8)

# --- Combine the two ---
combined = cv2.addWeighted(foreground_sharp, 1.2, background_dark, 0.8, 0)

# --- Reading Security Layer ---

overlay_img = cv2.imread("SL.jpg", cv2.IMREAD_GRAYSCALE)

# --- Create a binary mask of just the white areas ---
_, white_mask = cv2.threshold(overlay_img, 240, 255, cv2.THRESH_BINARY)

# --- Create a mask for the gray areas ---
# You can adjust this threshold for the range of gray values you want to exclude
gray_mask = cv2.inRange(overlay_img, 100, 200)  # Mask for gray areas
gray_mask_3ch = cv2.merge([gray_mask] * 3)  # Convert to 3-channel

# --- Create a clean overlay (only the white areas) ---
# Set gray areas (those in the gray mask) to black (remove them completely)
clean_overlay = overlay_img.copy()
clean_overlay[gray_mask == 255] = 0  # Set gray areas to 0 (black)

# --- Convert to 3-channel for blending ---
clean_overlay_3ch = cv2.merge([clean_overlay] * 3)

# Optional: feather the white mask
white_mask_float = white_mask.astype(np.float32) / 255.0
white_mask_float = cv2.GaussianBlur(white_mask_float, (15, 15), 0)

overlay_img_3ch = cv2.merge([overlay_img] * 3)

# Expand to 3 channels and apply transparency (alpha blending)
white_mask_3ch = cv2.merge([white_mask_float] * 3)
white_overlay = np.full((*overlay_img.shape, 3), 255).astype(np.float32)  # pure white image

# Set opacity level (e.g., 0.2 for 20% visibility)
# Blend the pattern with white based on the mask and desired opacity
opacity = 0.3  # adjust for stronger or softer overlay
pattern_blend = (
    clean_overlay_3ch * (1 - white_mask_3ch * opacity) + 
    white_overlay * (white_mask_3ch * opacity)
)
pattern_blend = np.clip(pattern_blend, 0, 255).astype(np.uint8)

x_offset = 100  # horizontal position on background
y_offset = 150  # vertical position on background

# Blend white into darkened background
bg = background_dark.astype(np.uint8)
bg_h, bg_w = bg.shape[:2]
ptn_h, ptn_w = overlay_img.shape[:2]

# Ensure the overlay doesn't go out of bounds
x_end = min(x_offset + ptn_w, bg_w)
y_end = min(y_offset + ptn_h, bg_h)
ptn_crop = pattern_blend[0: y_end - y_offset, 0: x_end - x_offset]

# Replace region in the background
bg_region = bg[y_offset:y_end, x_offset:x_end]
blended_region = cv2.addWeighted(bg_region.astype(np.float32), 1.0, ptn_crop.astype(np.float32), 1.0, 0)
bg[y_offset:y_end, x_offset:x_end] = np.clip(blended_region, 0, 255).astype(np.uint8)

# Update background
background_with_overlay = bg

# --- Step 7: Final composition ---
final_result = foreground_sharp.astype(np.float32) * mask_3ch + background_with_overlay.astype(np.float32) * (1 - mask_3ch)
final_result = np.clip(final_result, 0, 255).astype(np.uint8)

cv2.imshow("Final Result with White Overlay", final_result)
cv2.waitKey(0)
cv2.destroyAllWindows()