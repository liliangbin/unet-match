import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img = cv.imread("../testImages/4/img1.jpg", 0)
h, w = img.shape[:2]
pixelSequence = img.reshape([h * w, ])
numberBins = 256
histogram, bins, patch = plt.hist(pixelSequence, numberBins,
                                  facecolor='black', histtype='bar')
plt.xlabel("gray label")
plt.ylabel("number of pixels")
plt.axis([0, 255, 0, np.max(histogram)])
plt.show()
cv.imshow("img", img)
cv.waitKey()
