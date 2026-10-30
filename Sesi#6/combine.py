import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Membaca citra
img_bgr = cv2.imread("monyet.jpg")
if img_bgr is None:
    raise FileNotFoundError("File monyet.jpg tidak ditemukan. Letakkan satu folder dengan file Python ini.")

img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
tinggi, lebar = img.shape[:2]

# 2. Translation: geser 100 piksel ke kanan dan 50 piksel ke bawah
M_translation = np.float32([
    [1, 0, 100],
    [0, 1, 50]
])
hasil_translation = cv2.warpAffine(img, M_translation, (lebar, tinggi))

# 3. Scaling dengan faktor 0,5 dan 2
hasil_scaling_05 = cv2.resize(
    img, None, fx=0.5, fy=0.5,
    interpolation=cv2.INTER_LINEAR
)
hasil_scaling_2 = cv2.resize(
    img, None, fx=2, fy=2,
    interpolation=cv2.INTER_LINEAR
)

# 4. Rotation 45, 90, dan 180 derajat
def rotasi(citra, sudut):
    h, w = citra.shape[:2]
    pusat = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(pusat, sudut, 1.0)
    return cv2.warpAffine(citra, M, (w, h))

hasil_rotasi_45 = rotasi(img, 45)
hasil_rotasi_90 = rotasi(img, 90)
hasil_rotasi_180 = rotasi(img, 180)

# 5. Horizontal Flip dan Vertical Flip
hasil_horizontal_flip = cv2.flip(img, 1)
hasil_vertical_flip = cv2.flip(img, 0)

# 6. Menampilkan citra asli dan seluruh hasil transformasi
daftar_hasil = [
    ("Citra Asli", img),
    ("Translation (+100, +50)", hasil_translation),
    ("Scaling 0,5x", hasil_scaling_05),
    ("Scaling 2x", hasil_scaling_2),
    ("Rotation 45 derajat", hasil_rotasi_45),
    ("Rotation 90 derajat", hasil_rotasi_90),
    ("Rotation 180 derajat", hasil_rotasi_180),
    ("Horizontal Flip", hasil_horizontal_flip),
    ("Vertical Flip", hasil_vertical_flip)
]

plt.figure(figsize=(14, 10))
for i, (judul, citra) in enumerate(daftar_hasil):
    plt.subplot(3, 3, i + 1)
    plt.imshow(citra)
    plt.title(judul)
    plt.axis("off")

plt.tight_layout()
plt.savefig("hasil_transformasi_monyet.jpg", dpi=160, bbox_inches="tight")
plt.show()