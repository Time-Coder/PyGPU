# PyGPU: A very simple interface for calling GPU

## 1. Preview
### 1.1 Sum two arrays
If you want to sum two arrays with GPU, write code just like this:  
in *main.py*:
```python
# main.py
from pygpu import *
import numpy as np

n = 1000000
a = np.random.rand(n)
b = np.random.rand(n)

gpu = GPU()
gpu.set_program("sum.cl", "sum")
gpu.set_return(a)
# tell him the return value's shape is just like a

result = gpu(a, b)
```
in *sum.cl*:
```c
// sum.cl
__kernel void sum(__global float* result,
				  __global float* a,
				  __global float* b)
{
	int i = get_global_id(0);
	result[i] = a[i] + b[i];
}
```
### 1.2 Convert Colored Image into Gray
If you want to convert a colored image into gray mode with GPU, write code just like this:  
in *main.py*:
```python
# main.py
from pygpu import *
import numpy as np
import cv2

image_bgr = cv2.imread("1.jpg")

gpu = GPU()
gpu.set_program("bgr2gray.cl", "bgr2gray")
gpu.set_return(image_bgr[:, :, 0])
# tell him the return value's shape is just like the first channel of input image
# because the output image will be a gray image and has only one channel

image_gray = gpu(image_bgr)
```
in *bgr2gray.cl*:
```c
// bgr2gray.cl
__kernel void bgr2gray(__global uchar*  image_gray,
					   __global uchar3* image_bgr)
{
	int i = get_global_id(0);
	image_gray[i] = (uchar)(0.11 * image_bgr[i].x + 
							0.59 * image_bgr[i].y +
							0.3  * image_bgr[i].z);
}
```
Easy enough, isn't it?

## 2. Installing
### 2.1 Installing Numpy and OpenCV-Python
Just use the following command:
```shell
pip install numpy
pip install opencv-python
```
### 2.2 Installing PyOpenCL
1. Update OpenCL driver:   

	* For Intel graphics card, go to website [OpenCL™ Runtimes for Intel® Processors](https://software.intel.com/en-us/articles/opencl-drivers#win64), download the correct driver for your operating system then install it.

	* For Nvidia graphics card, go to website [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx?lang=en-us), download the correct driver for your operating system then install it.

	* For AMD graphics card, go to website [Installing The AMD Catalyst Software Driver](https://www.amd.com/en/support/kb/faq/ccc-install), follow the guide in the website and finish your installation.

2. Install PyOpenCL from prebuild binary:

	1. Go to website [PyOpenCL prebuild binary](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopencl), download the correct version for your operating system.

	2. Use command ``pip install <the file name you've just download>`` to install PyOpenCL

### 2.3 Installing PyGPU:
* If you just want to import pygpu temporarily, put the file *pygpu.py* in your work directory and then you can import it in your program.

* If you want to import pygpu from anywhere, put the file *pygpu.py* in the directory *Path\to\Python3\Lib\site-packages* or *Path\to\Anaconda3\Lib\site-packages* if you has Anaconda installed.

## 3. Usage
First of all, you need to import pygpu in the head using:
```python
from pygpu import *
```
### 3.1 process a single mission
1. **Define a GPU class' variable**(for example, name it as ``gpu``)  
There are three ways to do this: 
	1. ``gpu = GPU()``: This will use the default GPU device. In detail, it use the GPU on platform 0 and device 0.

	2. ``gpu = GPU(1, 0)``: Tell GPU class to use the GPU device on platform 1 and device 0.

	3. ``gpu = GPU("nvidia")``: Use a name string can also indicate GPU class to use which device.
To see what devices do you have, execute command ``AllGPUs.list_devices()``. In my computer, it print this result:
```python
>>> AllGPUs.list_device()
( 0 , 0 ): GeForce GTX 960M
( 1 , 0 ): Intel(R) HD Graphics 530
>>> 
```

2. **Write kernel function**  
The kernel function define the operation that you want to eval at each single data of your data set. You must write it in another file and in the rules of OpenCL kernel function. An also, PyGPU set several special rules rules for kernel function. In detail,
	1. Begin your kernel function with a modifier ``__kernel``.

	2. The return type of kernel must be ``void``.

	3. The first argument of kernel function must be the only one output argument. And must be in type of one-dimensional pointer, modified by ``__global``. For example, ``__global float*``. So in your kernel function's logic, you must store your final computing results in the first argument and can only store in this argument.

	4. Other arguments can be in base type or one-dimensional pointer. They can not be in two or more dimensional pointer or user defined type or class. And they are all input arguments.

	5. Never use ``__local`` to be the modifier of an argument.

	6. If an argument is a pointer, he must have a modifier. The modifier can only be ``__global`` or ``__constant``.
		* Use ``__global`` when this argument will vary each time when you call the variable ``gpu``.

		* Use ``__constant`` when this argument stay the same value each time when you call the variable ``gpu``

	7. In the body of the kernel function, use ``int i = get_global_id(0);`` to get the current work position. ``i`` means the kernel function now is generating the ``i``th data of the only output argument. ``i`` will vary from 0 to the length of first argument.(That comes the question: the first argument is a pure pointer, so how do program know the size of it's content? This will tell in step **Set return template**)

	8. Write how to generate ``i``th output data in normal C language way.
Maybe you will get a little confused about these rules. It dosen't matter. The examples in section **Preview** and **Examples** and in *examples* folder will help you understand them.

3. **Tell ``gpu`` to use your kernel function**  
Return to python file where you defined ``gpu`` variable. Write this line to indicate GPU to use your kernel function:
```python
gpu.set_program('file_name', 'function_name')
```
Remember to replace ``'file_name'`` with your kernel file's real file name and ``'function_name'`` with your real kernel function name.

4. **Set return template**  
In the kernel function, the first argument is the output argument. It serve as the return value. There is a limitation that it can only be in one-dimensional pointer. So what if I want the return result to be a two or more dimensional numpy.ndarray? Don't worry, ``set_return`` method will help with it. Use ``set_return`` to transfer a template to tell ``gpu`` you want the result in this shape. Then ``gpu`` variable will interprete the raw one-dimensional array into the shape you want.  
For example, in kernel function the first argument is in type ``float*`` and you use ``gpu.set_return(np.zeros((100, 200, 3)))`` in the host side. ``gpu`` variable will reshape the raw one-dimensional ``float*`` into 100 rows 200 colums and 3 channels image liked matrix.  
Be attention: ``gpu.set_return(a)`` not means ``gpu``'s return result will store in variable ``a``. ``a`` just give a template to ``gpu`` variable.

5. **Call ``gpu`` just like a function**
Now it's time to pass input arguments to ``gpu`` variable. Just in this way:
```
result = gpu(arg1, arg2, arg3, ...)
```
Here are some rules to choose type of each arguments. For example, your kernel function is like:
```
__kernel void func(Type0 result, Type1 arg1, Type2 arg2, Type3 arg3, ...)
```
To satisfy the limitation of PyGPU, the arguments type in kernel function can only be in scalar types, vector types or their one-dimensional pointer.  

For scalar types or vector types, in kernel function you can choose from the following table:
Scalar Type | Vector2 Type | Vector3 Type | Vector4 Type | Vector8 Type | Vector16 Type
------------|--------------|--------------|--------------|--------------|--------------
char        |  char2       |  char3       |  char4       |  char8       |  char16
uchar       |  uchar2      |  uchar3      |  uchar4      |  uchar8      |  uchar16
short       |  short2      |  short3      |  short4      |  short8      |  short16
ushort      |  ushort2     |  ushort3     |  ushort4     |  ushort8     |  ushort16
int         |  int2        |  int3        |  int4        |  int8        |  int16
uint        |  uint2       |  uint3       |  uint4       |  uint8       |  uint16
long        |  long2       |  long3       |  long4       |  long8       |  long16
ulong       |  ulong2      |  ulong3      |  ulong4      |  ulong8      |  ulong16
half        |  half2       |  half3       |  half4       |  half8       |  half16
float       |  float2      |  float3      |  float4      |  float8      |  float16
double      |  double2     |  double3     |  double4     |  double8     |  double16

For their pointers, in kernel function you can choose from the following table:
Scalar Pointer | Vector2 pointer | Vector3 pointer | Vector4 pointer | Vector8 pointer | Vector16 pointer
---------------|-----------------|-----------------|-----------------|-----------------|-----------------
char*          | char2*          | char3*          | char4*          | char8*          | char16*
uchar*         | uchar2*         | uchar3*         | uchar4*         | uchar8*         | uchar16*
short*         | short2*         | short3*         | short4*         | short8*         | short16*
ushort*        | ushort2*        | ushort3*        | ushort4*        | ushort8*        | ushort16*
int*           | int2*           | int3*           | int4*           | int8*           | int16*
uint*          | uint2*          | uint3*          | uint4*          | uint8*          | uint16*
long*          | long2*          | long3*          | long4*          | long8*          | long16*
ulong*         | ulong2*         | ulong3*         | ulong4*         | ulong8*         | ulong16*
half*          | half2*          | half3*          | half4*          | half8*          | half16*
float*         | float2*         | float3*         | float4*         | float8*         | float16*
double*        | double2*        | double3*        | double4*        | double8*        | double16*

	* if you use **scalar type** in kernel function, for example the first input argument is ``double arg1``, in host program, you can set ``arg1`` a single value(means not numpy array or list or tuple or other things, just a single value). You don't need to transform the type of ``arg1``. That means if you want to set ``arg1 = 1``, then just pass ``1`` to ``gpu``. You don't need to transform 1 to specific type such as ``np.float64(1)`` or ``cl.cltypes.double(1)``, no need.

	* if you use **vector type** in kernel function, for example the second input argument is ``float3 arg2``, in host program, then you can set ``arg2`` one of following type value:
		* a one row numpy.ndarray with size 3, such as ``np.random.rand(3)``
		* a list with 3 scalar value, such as ``[1, 2, 3]``
		* a tuple with 3 scalar value, such as ``(1, 2, 3)``
	But you can't let ``arg2`` maked by ``cl.cltypes.make_float3(...)``. Forget the old type transform way, forget them.

	* if you use **pointer** in kernel function, for example the third input argument is ``__global float*``, in the host program, you can set ``arg3`` one of the following type:
		* a list of single value, such as ``[1, 2, 3, 4, 5, 6, ...]``
		* a list of list or more nesting, such as ``[[1,2,3], [3,5,2], [9,3,6], ...]``
		* a numpy.ndarray, such as ``np.random.rand(3)``, ``np.random.rand(3, 3)``, ``np.random.rand(3, 3, 3)``
	All multi-dimension matrix liked data will be flatten into one dimension. And in the kernel side, you need to do some index tranform. You will see it in examples.

6. **Next time you call ``gpu``**
If you have call ``gpu`` once in this way: ``result = gpu(arg1, arg2, arg3, ...)``, next time you call ``gpu`` if some arguments are the same as first time, use ``None`` can avoid copying large data from host to device. For example, if ``arg1`` is a very large matrix and you called ``gpu`` once just like:
```
result1 = gpu(large_matrix, 3)
```
Next time you also want to process this matrix with another int value 4, avoid using
```
result2 = gpu(large_matrix, 4)
```
instead, using
```
result2 = gpu(None, 4)
```

### 3.2 process multi-missions with a single GPU
If you want to do some same process for different arrays, such as for different images, but these images have the same size, you can copy these images to device together, process them together and copy them from device to host together. This will save a lot of time. You can do this in the following way:
1. **Define a GPU class variable ``gpu``**  
2. **Write kernel program**  
3. **Tell ``gpu`` to use your kernel program**  
4. **Set return template**  
5. **Set arguments**:  
This is the first difference from processing single mission. For each mission, all the matrix liked or array liked arguments must have the same size. So you need to sent each input argument a template before add missions. You can set arguments in this way:
```
gpu.set_args(arg1, arg2, arg3, ...)
```
For each mission, ``arg1`` or ``arg2`` will vary from time to time, but every one have the same size.
6. **Add missions**
For example, if you set arguments template using ``gpu.set_args(arg1, arg2, arg3)`` and ``arg1`` is a image, it will change for each mission and ``arg2`` and ``arg3`` will be fixed. You can add missions in this way:
```
gpu.add_mission(image1)
gpu.add_mission(image2)
gpu.add_mission(image3)
gpu.add_mission(image4)
```
Be attention, their are some special rules for adding mission:
	* image1 to 4 must have the same size
	* varying arguments in kernel function must be in global pointer type. And global pointer arguments must be set in each ``add_mission`` time.
	* fixed arguments in kernel function must be non-pointer type or local pointer
7. **Process all the missions at the same time**
Just use ``gpu.run()`` is OK.
8. **Get results**
You can get ``i``th mission's result by using ``result = gpu.result(i)``

### 3.3 process multi-missions with multi-GPUs
If you change the declaration in the first step ``gpu = GPU()`` into ``gpu = AllGPUs()``, it will automatically distribute missions to all GPUs on your computer and let them start computation at the same time.

### 3.4 other functions
1. ``GPU.device_name()`` return the current GPU name.
2. ``GPU.print_info()`` or ``AllGPUs.print_info()`` print current GPU's detail information or all GPUs' detail information.
3. ``GPU.clear()`` or ``AllGPUs.clear()`` make GPU or AllGPUs class' instance return to the state before set_program.
4. ``GPU.clear_missions()`` or ``AllGPUs.clear_missions()`` make GPU or AllGPUs class' instance return to the state before add first mission.
5. ``GPU.print_performance()`` or ``AllGPUs.print_performance()`` print the performace of last computation. It includes:
	* total time
	* time of copying data from host to device
	* time of computing
	* time of copying data from device to host
	* computing/total time ratio
	* computing/copying time ratio
6. ``GPU.device2host_time()`` return the time of copying data from device to host of last computation(in second).
7. ``GPU.compute_time()`` return computing time of last computation(in second).
8. ``GPU.host2device_time()`` return the time of copying data from host to device of last computation(in second).
9. ``GPU.total_time()`` return the total time of last computation(in second).

## 4. Examples
In **Preview** section, there are already two examples. In ``examples`` folder, there are two more complex examples:
	* Gaussain Blur a image(*blur.py*, it will teach you how to transform index between 2-dimensional matrix and one-dimensional array)
	* Gaussain Blur a lot of images(*batch_process.py*, it will teach you how to use multi-processing method of PyGPU)
You can run them directly.

## 5. Limitation
This library's usage is simple enough. Simple means the degree of freedom is low. So their are much limitation. Here list some limitation I known:
1. There can only be one output argument in kernel function.
2. The output argument in kernel function must be the first argument in kernel function.
3. You cannot use ``__local`` memory
4. You can only use one-dimensional pointer in kernel function.
5. You can't distribute work groups or work items by your self, you can only let OpenCL do this automatically for you.
6. In multi-missions processing, you can only change global pointer type arguments and global pointer type arguments must be set for each mission even if they are same.
7. You can only use the types in the table in kernel function. You cannot use user defined class or structure or other types.
There are also many other limitiations. But for some simple parallel processing, I think these functions are enough.
