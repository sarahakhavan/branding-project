import cv2
import numpy as np

img = cv2.imread("ax.jpg")
if img is not None:
    print("Image loaded successfully.")
else:
    print("Image not found.")


# Gamma correction for exposure
gamma = 1.8  # <1 is darker, >1 is brighter
look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")
adjusted = cv2.LUT(img, look_up_table)

cv2.imshow("Adjusted", adjusted)




cv2.waitKey(0)
cv2.destroyAllWindows()