# Bar-Code Detection and Decoding ║▌║█║▌│║▌║▌█

## Introduction

This project aims to use image processing techniques to analyze images and extract information, such as detecting barcodes and identifying the largest rectangular contour in an image. To implement these tasks, libraries such as OpenCV, NumPy, matplotlib, and pyzbar were used.

## Key Steps of the Project
![Screenshot 2025-01-30 195200](https://github.com/user-attachments/assets/7840e488-372c-4a46-959d-fe00457ea985)

- Reading the Image
- Resizing the Image
- Convert the Image to Grayscale
- Calculate the Gradient Using Sobel Operators
- Compute Gradient Magnitude
- Apply Gaussian Blur
- Apply Adaptive Thresholding
- Perform Morphological Operations
- Find Contours
- Draw the Largest Contour
- Decode Barcodes
- Display the Results

## Detailed Steps

### Reading the Image
- The image is read from a specified path using the `cv2.imread` function.
- If the image is not found, an error is raised.

### Resizing the Image
- The image is resized while maintaining the aspect ratio using the `cv2.resize` function.

### Processing the Image
- The image is converted to grayscale using `cv2.cvtColor`.
- The gradient is calculated using Sobel filters (`cv2.Sobel`).
- The gradient magnitude is computed and converted to an 8-bit image.
- Gaussian blur is applied to reduce noise.
- Adaptive Thresholding is applied to separate the background from the foreground.
- Morphological operations such as closing and erosion are applied to refine the contours.

### Finding the Largest Rectangular Contour
- Contours are detected in the image using `cv2.findContours`.
- The largest rectangular contour is identified using `cv2.minAreaRect` and drawn on the original image.

### Decoding Barcodes
- The Pyzbar library is used to detect and decode barcodes.
- A rectangle is drawn around the barcode, and the decoded text is displayed on the image.

### Displaying Images
- The original and processed images are displayed using matplotlib to compare the results.

## Challenges Faced in the Project

- **Image Quality**: Some images may contain noise or poor lighting, which affects processing accuracy.
- **Contour Detection**: In some cases, contours may not be detected correctly due to complex backgrounds.
- **Barcode Decoding**: Some barcodes may not be clear due to reflections or distortions in the image.
- **Image Resolution**: When the image was resized, the text placement and scaling were not adjusted properly, leading to misaligned or cropped text.

## Results and Evaluation

- **Results**:
  - All steps were successfully executed, including image enhancement, contour detection, and barcode decoding.
  - Results were visually displayed using matplotlib to compare original and processed images.

- **Evaluation**:
  - Contour detection accuracy was high in images with simple backgrounds.
  - Barcode decoding was successful in most cases, but performance needs improvement for low-quality images.

## Conclusion

The project was successfully implemented using basic and advanced image processing techniques. The results were satisfactory in most cases, but there is room for improvement in low-quality or complex background images.

This project can be a foundation for developing more advanced applications in image processing and computer vision, such as automated barcode scanning systems or object detection tools.

## Project Structure
- `BarCode_Detection_And_Decoding.ipynb`: Contains all the functions with testing the code on all the images attached in the `input_image` folder.
- `requirements.txt`: Contains the libraries used in the project.

## Installation

To install the required libraries, run the following:

```bash
pip install -r requirements.txt
