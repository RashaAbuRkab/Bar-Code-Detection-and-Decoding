"""Image Processing techniques """

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pyzbar.pyzbar import decode

def read_image(image_path):
    """Reads an image from a given path."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Error: Unable to load image at {image_path}.")
    return image

def resize_image(image, width=1000):
    """Resize the image to a specific width while maintaining the aspect ratio."""
    aspect_ratio = float(image.shape[1]) / float(image.shape[0])
    new_height = int(width / aspect_ratio)
    resized_image = cv2.resize(image, (width, new_height))
    return resized_image

def process_image(image):
    """Processes the image to detect rectangular contours."""
    
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate gradient using Sobel operators
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitude 
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)

    # Apply Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gradient_magnitude, (15, 5), 0)
    
    # Apply automatic Otsu's thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_OTSU)

    # Morphological operations to close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))  # Kernel size adjusted
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Erosion followed by dilation to refine edges
    closed = cv2.erode(closed, None, iterations=2)
    closed = cv2.dilate(closed, None, iterations=2)

    return gradient_magnitude, blurred, thresh, closed

def find_largest_contour(image, closed):
    """Finds and draws the largest rectangular contour in the image."""
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        raise ValueError("No contours found in the image.")

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    # Draw the contour on the original image
    result_image = image.copy()
    cv2.drawContours(result_image, [box], -1, (0, 165, 255), 2)
    return result_image, box

def decode_barcode(result_image):
    """Decodes barcodes in the image using pyzbar."""
    barcodes = decode(result_image)
    decoded_info = []
    
    if not barcodes:
        print("No barcodes found.")
    else:
        for barcode in barcodes:
            barcode_data = barcode.data.decode("utf-8")
            decoded_info.append(barcode_data)
            print(f"Decoded Barcode: {barcode_data}")
            
            rect = barcode.rect
            x, y, w, h = rect[0], rect[1], rect[2], rect[3]
        
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 8)  

            text = barcode_data
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.2
            font_thickness = 5
            text_color = (255, 0, 0)  
            background_color = (255, 255, 255)  

            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

            text_x = x + 2
            text_y = y - 5  

            cv2.rectangle(result_image, (text_x, text_y - text_height - 10), (text_x + text_width, text_y + 10), background_color, -1)

            cv2.putText(result_image, text, (text_x, text_y), font, font_scale, text_color, font_thickness)

    return decoded_info, result_image

def show_images(images, titles):
    """Helper function to display multiple images."""
    num_images = len(images)
    plt.figure(figsize=(16, 12))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.title(titles[i])
        plt.imshow(images[i], cmap='gray' if len(images[i].shape) == 2 else None)
        plt.axis('off')
    plt.show()

def analyze_image(image_path):
    """Full pipeline to analyze an image and display intermediate results."""
    try:
        image = read_image(image_path)
        
        # Resize the image for better barcode detection if necessary
        resized_image = resize_image(image)

        original_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        gradient, blurred, thresh, closed = process_image(resized_image)

        # Detect largest contour
        result_image, _ = find_largest_contour(resized_image, closed)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        # Decode barcodes
        decoded_info, result_image_with_text = decode_barcode(resized_image)
        if decoded_info:
            print(f"Decoded Barcodes: {decoded_info}")
        else:
            print("No barcodes detected.")

        show_images(
            [original_image, gradient, blurred, thresh, closed, result_image_with_text],
            ["Original Image", "Gradient", "Blurred", "Threshold", "Morphology", "Final Result"]
        )
    except Exception as e:
        print(f"An error occurred: {e}")



