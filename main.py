import cv2
import numpy as np
from rembg import remove
from PIL import Image

# --- Step 1: Load original image ---
img = cv2.imread("ax.png")
if img is not None:
    print("Image loaded successfully.")
else:
    print("Image not found.")

original = img.copy()

# --- Step 2: Use rembg to get the mask (alpha channel) ---
input_pil = Image.open("ax.jpg")
output_pil = remove(input_pil)
output_np = np.array(output_pil)

# Extract alpha channel as mask
if output_np.shape[2] == 4:
    mask = output_np[:, :, 3]
else:
    raise ValueError("Alpha channel not found.")

# Normalize mask to 0–1 range
mask_normalized = mask.astype(np.float32) / 255.0
mask_3ch = cv2.merge([mask_normalized]*3)

mask_normalized = cv2.GaussianBlur(mask_normalized, (11, 11), 0)
mask_3ch = cv2.merge([mask_normalized]*3)

# --- Step 3: Create background and foreground layers ---
foreground = original.astype(np.float32) * mask_3ch
background = original.astype(np.float32) * (1 - mask_3ch)

# --- Step 4: Gamma correction on background ---
gamma = 1.9

lut = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(256)]).astype("uint8")
background_dark = cv2.LUT(background.astype(np.uint8), lut)

# --- Step 5: Sharpen the foreground ---
sharpen_kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
foreground_sharp = cv2.filter2D(foreground.astype(np.uint8), -1, sharpen_kernel)

# --- Step 6: Combine the two ---
combined = cv2.addWeighted(foreground_sharp, 1.0, background_dark, 1.0, 0)

# --- Step 7: Display result ---
cv2.imshow("Result", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()