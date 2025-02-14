# CONTOUR BASED SHAPE DESCRIPTOR

import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_contour_descriptor(image_path):
    # Read image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Threshold the image
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Draw contours
    contour_image = cv2.drawContours(np.zeros_like(image), contours, -1, (255, 255, 255), 1)

    # Generate chain code for contours
    chain_codes = []
    directions = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
    for contour in contours:
        chain_code = []
        for i in range(len(contour) - 1):
            diff = contour[i + 1][0] - contour[i][0]
            angle = np.arctan2(diff[1], diff[0])
            direction_index = int((angle + np.pi) / (np.pi / 4)) % 8
            chain_code.append(directions[direction_index])
        chain_codes.append(''.join(chain_code))
        
    return image, chain_codes, contour_image

# Provide the path to your image
image_path = r'C:\Users\cl501_29\Desktop\Vamsidhar\MV\Cameraman.jpg'

# Get descriptors and display images
original_image, chain_codes, contour_image = get_contour_descriptor(image_path)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(original_image, cmap='gray')

plt.subplot(1, 2, 2)
plt.title("Contours")
plt.imshow(contour_image, cmap='gray')
plt.show()

print("Contour Chain Codes:")
for i, code in enumerate(chain_codes):
    print(f"Contour {i+1}: {code}")
