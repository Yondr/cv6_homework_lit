import numpy as np
import cv2
from matplotlib import pyplot as plt

img_bgr = cv2.imread("img/test2.png")
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

row, col = 350, 400
half = 5  # окно 11x11
patch = img[max(0,row-half):row+half+1, max(0,col-half):col+half+1, :]



mask = np.all(patch < 250, axis=2)
patch_valid = patch[mask]
if patch_valid.size == 0:
    patch_valid = patch.reshape(-1,3)

white = patch_valid.reshape(-1,3).mean(axis=0)

eps = 1e-6
coeffs = 255.0 / np.maximum(white, eps)

balanced = img.copy()
for c in range(3):
    balanced[..., c] *= coeffs[c]

balanced = np.clip(balanced, 0, 255).astype(np.uint8)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.imshow(img.astype(np.uint8)); plt.title("Original"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(balanced); plt.title("White patch"); plt.axis("off")
plt.show()

######

# Load your image
img = cv2.imread('img/test2.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Compute the mean values for all three colour channels (red, green, blue)
mean_r = np.mean(img[..., 0])
mean_g = np.mean(img[..., 1])
mean_b = np.mean(img[..., 2])

# Compute the coefficients kr, kg, kb
# Note: there are 3 coefficients to compute but we only have 2 equations.
# Therefore, you have to make an assumption, fix the value of one of the
# coefficients and compute the remining two
# Hint: You can fix the coefficient of the brightest colour channel to 1.

means = np.array([mean_r, mean_g, mean_b])
max_idx = np.argmax(means)
coeffs = np.ones(3)
for i in range(3):
    if i != max_idx:
        coeffs[i] = means[max_idx] / means[i]

kr, kg, kb = coeffs

# Apply color balancing and generate the balanced image
balanced = img*coeffs
balanced = np.clip(balanced, 0, 255).astype(np.uint8)
# Show the original and the balanced image side by side
plt.figure(figsize=(12,5))
plt.subplot(1,2,1); plt.imshow(img.astype(np.uint8)); plt.title("Original"); plt.axis("off")
plt.subplot(1,2,2); plt.imshow(balanced); plt.title("Grayworld"); plt.axis("off")
plt.show()


