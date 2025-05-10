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

# --- Display result ---
cv2.imshow("Result", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()