import sys
sys.path.append('../')
from pygpu import *
import numpy as np
import cv2
from math import *
import os

def remove_all_dest():
	print("removing...")
	files = [i for i in os.listdir('Images/dest_gray')]
	for file in files:
		os.remove('Images/dest_gray/' + file)

	files = [i for i in os.listdir('Images/dest_blur')]
	for file in files:
		os.remove('Images/dest_blur/' + file)

def gauss_mat(win_width, sigma):
	n = (win_width-1)/2
	x, y = np.meshgrid(range(win_width), range(win_width))
	mat = 1/sqrt(2*pi)/sigma*np.exp(-0.5*((x-n)**2+(y-n)**2)/sigma**2)
	return mat/np.sum(mat)

def batch_blur():
	image_src = cv2.imread("Images/src/The_Fellowship_of_the_Ring1.jpg")

	gpus = AllGPUs()
	gpus.set_program("kernels/blur.cl", "blur")
	gpus.set_return(image_src)
	gpus.set_args(image_src, image_src.shape[0], image_src.shape[1], gauss_mat(5, 3), 5)
	print("reading...")
	files = [i for i in os.listdir('Images/src')]
	for file in files:
		# print("reading", 'Images/' + folder + "/" + file, "...")
		gpus.add_mission(cv2.imread('Images/src/' + file))

	print("running...")
	gpus.run()

	print("writing...")
	i = 0
	for file in files:
		# print("writing ", 'Images/' + folder + "/" + file[:-4] + "-blur.jpg ...")
		cv2.imwrite('Images/dest_blur/' + file[:-4] + "-blur.jpg", gpus.result(i))
		i += 1

	print("All done!")
	print("See the results in \'Images/dest_blur\' folder.")
	print()
	gpus.print_performance()

remove_all_dest()
batch_blur()