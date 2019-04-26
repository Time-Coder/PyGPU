import sys
sys.path.append('../')
from pygpu import *

def small_array_test():
	a = [5, 8, 2, 9, 0, 3]
	b = [3, 0, 6, 2, 5, 2]

	gpu = GPU()
	gpu.set_program("kernels/sum.cl", "sum")
	gpu.set_return(a)
	c = gpu(a, b)

	print("a =", a)
	print("b =", b)
	print("c =", c)

def large_array_test():
	n = 1000000
	a = np.random.rand(n)
	b = np.random.rand(n)

	gpu = GPU()
	gpu.set_program("kernels/sum.cl", "sum")
	gpu.set_return(a)
	result = gpu(a, b)

	gpu.print_performance()

# small_array_test()
large_array_test()