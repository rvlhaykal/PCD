
import os
import imageio.v3 as imageio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

folder = os.path.dirname(os.path.abspath(__file__))
img = imageio.imread(os.path.join(folder, "plat.jpg"))

# Memastikan gambar memiliki 3 channel warna
if img.ndim == 2:
    original = np.stack([img, img, img], axis=-1)
else:
    original = img[..., :3]


gray = np.dot(
    original[..., :3],
    [0.299, 0.587, 0.114]
).astype(np.uint8)

def hist_equal(image):
    hist, bins = np.histogram(
        image.flatten(),
        bins=256,
        range=[0, 256]
    )

    cdf = hist.cumsum()

    # Menghilangkan nilai histogram kosong di awal
    cdf_min = cdf[cdf > 0][0]

    # Normalisasi CDF
    cdf_normal = (
        (cdf - cdf_min) / (cdf[-1] - cdf_min) * 255
    )

    cdf_normal = np.clip(cdf_normal, 0, 255)

    # Menghasilkan gambar ekualisasi
    result = np.interp(
        image.flatten(),
        bins[:-1],
        cdf_normal
    )

    return result.reshape(image.shape).astype(np.uint8)


equalized = hist_equal(gray)

# Membuat kernel 3x3
fp = np.ones((3, 3), dtype=bool)

# Menghilangkan titik tengah
fp[1, 1] = False

# Menyiapkan gambar hasil filter berwarna
filtered = np.zeros_like(original)

# Melakukan filter pada setiap channel warna
for c in range(3):

    mn = ndimage.minimum_filter(
        original[:, :, c],
        footprint=fp
    )

    mx = ndimage.maximum_filter(
        original[:, :, c],
        footprint=fp
    )

    filtered[:, :, c] = np.clip(
        original[:, :, c],
        mn,
        mx
    )

plt.figure(figsize=(14, 9))

# Gambar asli
plt.subplot(2, 3, 1)
plt.imshow(original)
plt.title("Gambar Asli")
plt.axis("on")

# Ekualisasi histogram
plt.subplot(2, 3, 2)
plt.imshow(equalized, cmap="gray", vmin=0, vmax=255)
plt.title("Ekualisasi Histogram")
plt.axis("on")

# Filter batas
plt.subplot(2, 3, 3)
plt.imshow(filtered)
plt.title("Filter Batas")
plt.axis("on")

# Histogram asli
plt.subplot(2, 3, 4)
plt.hist(gray.flatten(), bins=256, range=[0, 256])
plt.yscale("log")
plt.title("Histogram Asli")

# Histogram hasil ekualisasi
plt.subplot(2, 3, 5)
plt.hist(equalized.flatten(), bins=256, range=[0, 256])
plt.yscale("log")
plt.title("Histogram Hasil")

# Mengatur tata letak
plt.tight_layout(h_pad=3)

plt.savefig(
    os.path.join(folder, "hasil.png"),
    dpi=150
)

# Menampilkan gambar
plt.show()
