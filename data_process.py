# -*- coding:utf-8 -*-

import SimpleITK as sitk
import cv2
from keras.preprocessing.image import img_to_array, load_img


def to_hu(image):
    MIN_BLOOD = -10
    MAX_BLOOD = 400
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
# np.savetxt("test.txt", image_array)
image_array = to_hu(image_array)*2
image_array = image_array * 255
#image_array[image_array < 40] = 0
print(image_array[255][255])
print(image_array[160][160])
cv2.imwrite("info.png", image_array)
# np.savetxt("test2.txt", image_array * 255)
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
