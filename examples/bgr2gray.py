import sys
sys.path.append('../')
from pygpu import *
import cv2

image_bgr = cv2.imread("Images/src/The_Fellowship_of_the_Ring1.jpg")

gpu = GPU()
gpu.set_program("kernels/bgr2gray.cl", "bgr2gray")
gpu.set_return(image_bgr[:, :, 0])
image_gray = gpu(image_bgr)

cv2.imshow("bgr", image_bgr)
cv2.imshow("gray", image_gray)
cv2.waitKey(0)