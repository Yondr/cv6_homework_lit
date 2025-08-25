import cv2
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = [15, 10]


# Load an image (you can freely chose any image you like)
img = cv2.imread("img/test.jpg")
# Convert it to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# Plot it
plt.imshow(img)
plt.show()
# Split the image into the three colour channels
red, green, blue = cv2.split(img)

# Compose the image in the RGB colour space
img1 = cv2.merge([red, green, blue])

# Compose the image in the RBG colour space
img2 = cv2.merge([red, blue, green])

# Compose the image in the GRB colour space
img3 = cv2.merge([green, red, blue])

# Compose the image in the BGR colour space
img4 = cv2.merge([blue, green, red])

# Create the collage
out1 = np.hstack([img1, img2])
out2 = np.hstack([img3, img4])
out = np.vstack([out1, out2])

# Plot the collage
plt.imshow(out)
plt.show()
plt.axis(False)


# Load an image (you can freely chose any image you like)
img = cv2.imread("img/test.jpg")
# Convert it to RGB

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Make a collage
img_h = np.fliplr(img)
img_v = np.flipud(img)
img_hv = np.flipud(img_h)

top_row = np.hstack([img, img_h])
bottom_row = np.hstack([img_v, img_hv])
collage = np.vstack([top_row, bottom_row])

# Plot the collage
plt.imshow(collage)
plt.axis("off")
plt.show()
