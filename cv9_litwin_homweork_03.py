import cv2
import numpy as np
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = [15, 10]

def unsharp_rgb(img2, radius=1.0, amount=1.0):
    img = img2.astype(np.float32)
    blur = cv2.GaussianBlur(img, (0,0), sigmaX=radius, sigmaY=radius)  # ksize=(0,0) => берёт sigma
    sharp = cv2.addWeighted(img, 1+amount, blur, -amount, 0)           # img + amount*(img-blur)
    return np.clip(sharp, 0, 255).astype(np.uint8)

img = cv2.imread("img/test2.png")
img = img[..., ::-1]
unsharp = unsharp_rgb(img, radius=2.2, amount=2.8)
#cv2.imwrite("usm_simple.png", out)

plt.imshow(unsharp)
plt.show()

diff = img.astype(np.float32) - unsharp.astype(np.float32)
diff = np.clip(diff, 0, 255).astype(np.uint8)

plt.imshow(diff)
plt.show()

amount = -1.8
sharpened = img.astype(np.float32) + (img.astype(np.float32) - unsharp.astype(np.float32)) * amount
sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

plt.imshow(sharpened)
plt.show()
