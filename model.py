# -*- coding:utf-8 -*-

import glob
import os

import SimpleITK as sitk
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import img_to_array, load_img
from matplotlib import pyplot as plt

# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 指定GPU的第二种方法

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'  # A "Best-fit with coalescing" algorithm, simplified from a version of dlmalloc.
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 定量
config.gpu_options.allow_growth = True  # 按需
set_session(tf.Session(config=config))


def to_hu(image):
    MIN_BLOOD = -10
    MAX_BLOOD = 400
    image = (image - MIN_BLOOD) / (MAX_BLOOD - MIN_BLOOD)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


print('-' * 30)
print('Creating training images...')
print('-' * 30)
train_path = "../train/dataset/"
cnt = 1000
for j in range(74, 109):
    number = cnt + j
    filname = str(number)
    file = train_path + filname + "/venous phase/200"
    if not os.path.lexists(train_path + filname):
        print(train_path + filname + "===> 无当前文件夹")
        continue

    imgs = glob.glob(train_path + filname + "/venous phase" + "//*." + "dcm")
    print(len(imgs))
    imgdatas = np.ndarray((len(imgs), 512, 512, 1), dtype=np.uint8)
    imglabr = np.ndarray((len(imgs), 512, 512, 1), dtype=np.uint8)
    i = 1
    for imgname in imgs:
        name = ""
        if i < 10:
            numname = "0" + str(i)
        else:
            numname = str(i)
        name = file + numname + ".dcm"
        pngname = file + numname + "_mask.png"
        print("dcmname===>" + name + "\n" + "pngname==>" + pngname)
        if not os.path.exists(name):
            continue
        image = sitk.ReadImage(name)
        image_array = sitk.GetArrayFromImage(image)  # z, y, x
        print(image_array.shape)
        image_array = image_array.swapaxes(0, 2)
        image_array = image_array.swapaxes(0, 1)
        # 可以使用 transpose
        # 维数变化
        print(image_array.shape)

        image_array = to_hu(image_array) * 2
        image_array[image_array > 1] = 1.
        images = np.squeeze(image_array)

        plt.imshow(images, cmap="gray")
        plt.axis("off")
        #plt.savefig(name + "_new.png")
        print("image_array done")
        # plt.show()
        imgdatas[i - 1] = image_array

        print(imgname)
        img = load_img(pngname, grayscale=True)
        # img = img_to_array(img)
        label = img_to_array(img)

        imglabr[i - 1] = label

        i += 1

    print('loading done')

    np.save('../train/npy/' + filname + '_imgs_mask_train.npy', imglabr)
    print("imgs_mas_train_1001.npy done ")
    np.save('../train/npy/' + filname + '_imgs_train.npy', imgdatas)
    print('Saving to .npy files done.')
