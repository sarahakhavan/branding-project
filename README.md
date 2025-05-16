# Branding Project: Converting an image to match the brand of Palo Alto Networks

This project automates the conversion of input images to align with Palo Alto Networks' brand guidelines. It uses Python and OpenCV to apply brand-compliant transformations based on the instructions provided in the design guidelines.


## How to Run It 

### 

### On macOS/Linux

Run:
`python -m venv venv`

Then run:

`source venv/bin/activate`

### Install Dependencies

`pip install -r requirements.txt`

### Run the Script

`python main.py path/image.jpg`


## Image Processing Workflow

### 1. Layer Separation
- Separates the image into **background** and **foreground (person)** layers.
- Enables independent editing and manipulation of each layer.

### 2. Edge Refinement
- Applies **erosion** and **Gaussian blur** techniques to the foreground edges.
- Ensures smooth and natural blending between the person and the background.

### 3. Image Enhancements
- Performs general image improvements, such as contrast adjustment, sharpening, and color correction, to enhance overall quality.

### 4. Independent Overlay Integration
- **Background Overlay**:
  - Adds a **security layer** with gradient transparency.
  - Uses **selective blending** to preserve white areas while removing or muting gray tones in the overlay layer.
- **Foreground Overlay**:
  - Applies a second overlay with **adjustable opacity** and **smooth blending**.

### 5. Final Composition
- Merges the enhanced foreground and background layers.
- Applies subtle **branding elements** to finalize the image.
 

## Assumptions and Notes

* **Image Composition:** The image selected for processing should ideally feature the person positioned at the center of the screen, similar to the reference or guide image. This ensures optimal processing and alignment during the image manipulation steps.
* **Brand Colors and Gradients:** The gradients in the final output were designed using the brand's official color palette as specified in their guidelines. However, it is important to note that the results from the gradients did not fully align with the computed results shown in the official guidelines. This discrepancy may be due to differences in color rendering or processing methods used in the original guidelines versus the custom implementation in this project. It is also possible that the brand guidelines may not have been fully committed to their exact brand colors, leading to this difference in the final results.

## Potential Improvements

* Refactor the script into modular functions for better readability and maintainability.
* Dynamically analyze the color composition of input images and adjust edits accordingly (the current version applies a fixed transformation).
* Add better documentation
* Add better error handling
* Add command-line arguments for output path
* Batch process folders of images
* Package the script into a CLI tool



