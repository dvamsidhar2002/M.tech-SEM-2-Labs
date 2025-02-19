import cv2
import numpy as np
import matplotlib.pyplot as plt


def harris_corner_detection(image_path, block_size=2, ksize=3, k=0.04, threshold=0.01):
    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert to float32
    gray = np.float32(gray)
    
    # Apply Harris Corner Detector
    dst = cv2.cornerHarris(gray, block_size, ksize, k)
    
    # Dilate to mark the corners
    dst = cv2.dilate(dst, None)
    
    # Threshold for an optimal value
    img[dst > threshold * dst.max()] = [0, 0, 255]
    
    # Display the result
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Harris Corners')
    plt.axis('off')
    plt.show()

    # Display images
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[1].set_title("Detected Corners")
    axs[1].axis("off")
    
# Example usage
image_path =r'C:\Users\cl501_29\Desktop\Vamsidhar\MV\chess-board-template-printable-vector-46845543.jpg'  # Replace with your image path
harris_corner_detection(image_path)