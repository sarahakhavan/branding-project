import cv2
import numpy as np

img = cv2.imread("ax.png")
if img is not None:
    print("Image loaded successfully.")
else:
    print("Image not found.")