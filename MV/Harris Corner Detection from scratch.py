import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to the local image
image_path = r"C:\Users\cl501_29\Desktop\Vamsidhar\MV\chess-board-template-printable-vector-46845543.jpg"

# Load the image
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Error: Unable to load image from {image_path}")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Harris corner detection
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Dilate for better visibility
dst = cv2.dilate(dst, None)

# Threshold and mark corners
threshold = 0.01 * dst.max()
image_with_corners = image.copy()
image_with_corners[dst > threshold] = [0, 255, 0]  # Mark corners in green

# Convert to binary for connected components
binary_image = np.uint8(dst > threshold) * 255

# Apply connected components analysis
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

# Draw detected centroids on the image
for (x, y) in centroids:
    x, y = int(x), int(y)
    cv2.circle(image_with_corners, (x, y), radius=5, color=(255, 0, 0), thickness=-1)

# Display images
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB))
axs[1].set_title("Detected Corners")
axs[1].axis("off")

plt.show()