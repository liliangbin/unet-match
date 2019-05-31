# -*- coding:utf-8 -*-

import SimpleITK as sitk
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array, load_img


def to_hu(image):
    image = np.dot(image, 1) - 1024
    print(image)
    MIN_BLOOD = -400
    MAX_BLOOD = np.max(image)
    print(MAX_BLOOD, "===", MIN_BLOOD)
    image = (image - MIN_BLOOD) / (MAX_BLOOD - MIN_BLOOD)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


image = sitk.ReadImage("20001.dcm")
image_array = sitk.GetArrayFromImage(image)  # z, y, x
print(image_array)
print(image_array.shape)
image_array = image_array.swapaxes(0, 2)
image_array = image_array.swapaxes(0, 1)
# 可以使用 transpose
# 维数变化
print("image_array==shape==>", image_array.shape)
image_array[image_array == -2000] = 0

image_array = to_hu(image_array)

print(np.mean(image_array*255))
cv2.imwrite("info.png", image_array * 255)

# images = np.squeeze(image_array)
#
# plt.imshow(images, cmap="gray")
# plt.axis("off")
# plt.savefig("test.png")

# plt.show()
print("image_array done")
img = load_img("20001_mask.png", color_mode="grayscale")
# img = img_to_array(img)
label = img_to_array(img)
print("label==shape ==", label.shape)
