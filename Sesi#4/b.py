import imageio as img
import numpy as np
import matplotlib.pyplot as plt

def hist_equal(image):
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0,256])
    cdf = hist.cumsum()
    cdf_normal = cdf / cdf.max()

    image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normal)

    return image_equalized.reshape(image.shape).astype(np.uint8)

image = img.imread('D:\\source_d.jpg')

result = hist_equal(image)

hist_img, bins = np.histogram(image.flatten(), bins=256, range=[0,256])
hist_result, bins = np.histogram(result.flatten(), bins=256, range=[0,256])

plt.figure(figsize=(10,10))

plt.subplot(2,2,1)
plt.imshow(image)

plt.subplot(2,2,2)
plt.imshow(result)

plt.subplot(2,2,3)
plt.plot(hist_img)

plt.subplot(2,2,4)
plt.plot(hist_result)

plt.show()
