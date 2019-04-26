import sys
sys.path.append('../')
from pygpu import *
import cv2
from math import *

def gauss_mat(win_width, sigma):
	n = (win_width-1)/2
	x, y = np.meshgrid(range(win_width), range(win_width))
	mat = 1/sqrt(2*pi)/sigma*np.exp(-0.5*((x-n)**2+(y-n)**2)/sigma**2)
	return mat/np.sum(mat)

image_src = cv2.imread("Images/src/The_Fellowship_of_the_Ring1.jpg")

gpu = GPU()
gpu.set_program("kernels/blur.cl", "blur")
gpu.set_return(image_src)
image_dest = gpu(image_src, image_src.shape[0], image_src.shape[1], gauss_mat(9, 3), 9)

cv2.imshow("src", image_src)
cv2.imshow("dest", image_dest)
cv2.waitKey(0)