import pyopencl as cl
import pyopencl.cltypes
import numpy as np
import copy
from string import digits
import time
from threading import Thread

class GPU():
	def __init__(self, Platform_id = 0, Device_id = 0):
		self._cl_basenames = ['uchar', 'char', 'ushort', 'short', \
							  'uint', 'int', 'ulong', 'long', \
							  'half', 'float', 'double']
		self._np_basenames = ['np.uint8', 'np.int8', 'np.uint16', 'np.int16', \
							  'np.uint32', 'np.int32', 'np.uint64', 'np.int64', \
							  'np.float16', 'np.float32', 'np.float64']

		if isinstance(Platform_id, str):
			platforms = cl.get_platforms()
			for i in range(len(platforms)):
				if Platform_id.lower() in cl.get_platforms()[i].get_info(cl.platform_info.NAME).lower():
					Platform_id = i
					break

		if isinstance(Platform_id, str):
			Platform_id = 0

		# PyOpenCL variables
		self._platform = cl.get_platforms()[Platform_id]
		self._device = self._platform.get_devices(cl.device_type.GPU)[Device_id]
		self._context = cl.Context([self._device])
		self._queue = cl.CommandQueue(self._context, self._device)
		self._program = None
		self._kernel = None
		self._worksize = None

		# some flags
		self._program_settle = False
		self._return_settle = False
		self._args_settle = False
		self._is_three = False
		self._is_called = False
		self._is_image = False

		# arguments related
		self._fname = ''
		self._cl_typenames = []
		self._default_args = []
		self._default_args_origine = []
		self._real_args = []

		# single times process related
		self._dest_buffer = None

		# batch process related
		self._n_missions = 0
		self._varying_args = []
		self._mission_global_args = []
		self._mission_finished = False
		self._mission_global_buffers = []

		# time record
		self._host2device_time = 0
		self._device2host_time = 0
		self._calculate_time = 0

	def set_program(self, file_name, function_name):
		self._clear_program()
		self._fname = function_name
		self._get_cl_typenames_in_kernel(file_name, function_name)
		self._program = cl.Program(self._context, open(file_name).read()).build()
		self._kernel = cl.Kernel(self._program, function_name)
		self._mission_global_buffers = [None] * len(self._varying_args)
		self._mission_global_args    = [None] * len(self._varying_args)
		for i in range(len(self._mission_global_args)):
			self._mission_global_args[i] = []
		self._program_settle = True

	def set_return(self, arg):
		if not self._program_settle:
			print("Error in GPU::set_return(self, arg):")
			print("Please set_program before set_return!")
			exit(-1)

		self._clear_return()
		if isinstance(arg, np.ndarray) and arg.dtype == 'uint8':
			self._is_image = True
		count = self._get_number(self._cl_typenames[0])
		arg = np.array(arg, dtype=eval(self._dtype(self._cl_typenames[0])))
		self._worksize = (int(arg.size/count),)
		if count == 3 and len(arg.shape) == 3 and arg.shape[2] == 3:
			arg = np.insert(arg, 3, values=0, axis=2)
			self._is_three = True
		
		self._dest_buffer = self._write_only_buffer(arg.nbytes)
		self._default_args = []
		self._default_args.append(self._dest_buffer)
		self._default_args_origine.append(arg)
		self._kernel.set_arg(0, self._dest_buffer)
		self._return_settle = True
		self._args_settle = False

	def set_args(self, *args):
		if not self._return_settle:
			print("Error in GPU::set_args(self, *args):")
			print("Please setReturn before setArgs!")
			exit(-1)

		if len(args) != len(self._cl_typenames)-1:
			print("Error in GPU::set_args(self, *args):")
			print("need", len(self._cl_typenames)-1, "arguments, but you passed", len(args))
			exit(-1)

		self._clear_args()
		for i, host_arg in enumerate(args):
			if isinstance(host_arg, np.ndarray):
				origine_shape = host_arg.shape
				host_arg = host_arg.flatten()
				host_arg = np.reshape(host_arg, origine_shape)
				
			device_arg = self._type_transform(host_arg, self._cl_typenames[i+1])
			self._default_args.append(device_arg)
			self._kernel.set_arg(i+1, device_arg)

		self._real_args = copy.copy(self._default_args)
		self._args_settle = True

	def add_mission(self, *args):
		if not self._args_settle:
			self.set_args(*args)

		if len(args) != len(self._varying_args)-1:
			print("Error in GPU::add_mission(self, *args):")
			print(len(self._varying_args)-1, "varied arguments must be passed!")
			exit(-1)

		self._host2device_time = 0
		self._calculate_time = 0
		self._device2host_time = 0

		return_arg = self._default_args_origine[0]
		self._mission_global_args[0].append(np.zeros_like(return_arg, dtype=return_arg.dtype))
		for i, host_arg in enumerate(args):
			self._mission_global_args[i+1].append(self._np2cl(host_arg, self._varying_args[i+1]))
		self._n_missions += 1

	def run(self):
		self._is_called = False
		if self._n_missions == 0:
			return

		for j in range(len(self._cl_typenames)):
			if j not in self._varying_args:
				self._kernel.set_arg(j, self._default_args[j])

		t1 = time.perf_counter()
		self._mission_global_args[0] = np.array(self._mission_global_args[0])
		self._mission_global_buffers[0] = self._buffer(self._mission_global_args[0].nbytes)
		self._kernel.set_arg(0, self._mission_global_buffers[0])
		for j in range(len(self._varying_args)):
			self._mission_global_buffers[j] = self._buffer(np.array(self._mission_global_args[j]))
			self._kernel.set_arg(self._varying_args[j], self._mission_global_buffers[j])

		t2 = time.perf_counter()
		cl.enqueue_nd_range_kernel(self._queue, self._kernel, (self._worksize[0]*self._n_missions,), None)
		t3 = time.perf_counter()
		cl.enqueue_copy(self._queue, self._mission_global_args[0], self._mission_global_buffers[0])
		t4 = time.perf_counter()

		if self._is_three and len(self._mission_global_args[0][0].shape) == 3 and self._mission_global_args[0][0].shape[2] == 4:
			self._mission_global_args[0] = np.delete(self._mission_global_args[0], -1, axis=3)

		if self._is_image and self._mission_global_args[0].dtype != 'uint8':
			self._mission_global_args[0] = self._mission_global_args[0].astype(np.uint8)

		self._host2device_time = t2 - t1
		self._calculate_time = t3 - t2
		self._device2host_time = t4 - t3
		self._mission_finished = True

	def __call__(self, *args):
		if not self._args_settle:
			self.set_args(*args)

		self._is_called = True
		if len(args) != len(self._cl_typenames)-1:
			print("Error in GPU::__call__(self, *args):")
			print("need", len(self._cl_typenames)-1, "arguments, but you passed", len(args))
			exit(-1)

		for i, arg in enumerate(args):
			if arg is not None:
				self._set_arg(i+1, arg)

		t1 = time.perf_counter()
		cl.enqueue_nd_range_kernel(self._queue, self._kernel, self._worksize, None)
		t2 = time.perf_counter()
		cl.enqueue_copy(self._queue, self._default_args_origine[0], self._dest_buffer)
		t3 = time.perf_counter()
		self._calculate_time = t2 - t1
		self._device2host_time = t3 - t2

		if self._is_image and self._default_args_origine[0].dtype != 'uint8':
			self._default_args_origine[0] = self._default_args_origine[0].astype(np.uint8)

		if self._is_three and len(self._default_args_origine[0].shape) == 3 and self._default_args_origine[0].shape[2] == 4:
			return np.delete(self._default_args_origine[0], -1, axis=2)
		else:
			return self._default_args_origine[0]

	def result(self, i):
		if not self._mission_finished:
			print("Error in GPU::result(i):")
			print("mission donot finished!")
			exit(-1)

		return self._mission_global_args[0][i]

	def _clear_program(self):
		self._fname = ''
		self._cl_typenames = []
		self._kernel = None
		self._program = None
		self._program_settle = False
		self._varying_args = []
		self._clear_return()

	def _clear_return(self):
		self._worksize = None
		self._is_three = False
		self._default_args_origine = []
		self._dest_buffer = None
		self._default_args = []
		self._return_settle = False
		self._is_image = False
		self._clear_args()

	def _clear_args(self):
		self._real_args = []
		self._is_called = False
		self._args_settle = False
		self._host2device_time = 0
		self._device2host_time = 0
		self._calculate_time = 0
		self.clear_missions()

	def clear_missions(self):
		self._n_missions = 0
		self._mission_global_buffers = [None] * len(self._varying_args)
		self._mission_global_args    = [None] * len(self._varying_args)
		for i in range(len(self._mission_global_args)):
			self._mission_global_args[i] = []
		self._mission_finished = False

	def clear(self):
		self._clear_program()

	def device2host_time(self):
		if not self._mission_finished and not self._is_called:
			print("Error in GPU::device2host_time(self):")
			print("There is no compution has done!")
			exit(-1)
		return self._device2host_time

	def host2device_time(self):
		if not self._mission_finished and not self._is_called:
			print("Error in GPU::host2device_time(self):")
			print("There is no compution has done!")
			exit(-1)
		return self._host2device_time

	def calculate_time(self):
		if not self._mission_finished and not self._is_called:
			print("Error in GPU::calculate_time(self):")
			print("There is no compution has done!")
			exit(-1)
		return self._calculate_time

	def total_time(self):
		if not self._mission_finished and not self._is_called:
			print("Error in GPU::total_time(self):")
			print("There is no compution has done!")
			exit(-1)
		return self._device2host_time + self._host2device_time + self._calculate_time

	def print_performance(self):
		if not self._mission_finished and not self._is_called:
			print("Error in GPU::print_performance(self):")
			print("There is no compution has done!")
			exit(-1)

		total_time = self._host2device_time + self._calculate_time + self._device2host_time
		if self._is_called:
			num = 1
		else:
			num = self._n_missions
		print(self.device_name(), "has processed", num, "missions,")
		print("           total time:", round(1000*total_time, 2), "ms")
		print(" mission average time:", round(1000*total_time/num, 2), "ms")
		print("  host -> device time:", round(1000*self._host2device_time, 2), "ms")
		print("       calculate time:", round(1000*self._calculate_time, 2), "ms")
		print("  device -> host time:", round(1000*self._device2host_time, 2), "ms")
		print("   calculate/total ratio:", round(100*self._calculate_time/total_time, 2), "%")
		print("calculate/transfer ratio:", round(100*self._calculate_time/(self._host2device_time+self._device2host_time), 2), "%")

	def device_name(self):
		return self._device.get_info(cl.device_info.NAME)

	def print_info(self):
		print("PyOpenCL Version:", cl.VERSION)
		print("OpenCL Version:", cl.get_cl_header_version())
		print()
		print("Platform Name:", self._platform.get_info(cl.platform_info.NAME))
		print("Platform Profile:", self._platform.get_info(cl.platform_info.PROFILE))
		print("Platform Vendor:", self._platform.get_info(cl.platform_info.VENDOR))
		print("Platform Version:", self._platform.get_info(cl.platform_info.VERSION))
		print()
		print("GPU Name:", self._device.get_info(cl.device_info.NAME))
		print("OpenCL Version:", self._device.get_info(cl.device_info.OPENCL_C_VERSION))
		print("GPU Vendor:", self._device.get_info(cl.device_info.VENDOR))
		print("GPU Version:", self._device.get_info(cl.device_info.VERSION))
		print("GPU Driver Version:", self._device.get_info(cl.device_info.DRIVER_VERSION))
		print("Max Work Group Size:", self._device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE))
		print("Max Compute Units:", self._device.get_info(cl.device_info.MAX_COMPUTE_UNITS))
		print("Max Work Item Size:", self._device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES))
		print("Local Memory Size:", self._device.get_info(cl.device_info.LOCAL_MEM_SIZE)/1024, 'KB')		

	def _get_number(self, string):
		num = ''
		for i_start in range(len(string)):
			if str.isdigit(string[i_start]):
				break
		for i_end in range(i_start, len(string)):
			if not str.isdigit(string[i_end]):
				break
		num_str = string[i_start:i_end]
		if num_str == '':
			return 1
		else:
			return int(num_str)

	def _np2cl(self, arg, i):
		counts = [2, 3, 4, 8, 16]
		for base in self._cl_basenames:
			for count in counts:
				typename = base + str(count)
				ptr_typename = typename + "*"
				if ptr_typename in self._cl_typenames[i]:
					arg = np.array(arg, dtype=eval('cl.cltypes.' + base))
					if count == 3 and len(arg.shape) == 3 and arg.shape[2] == 3:
						return np.insert(arg, 3, values=0, axis=2)

				if typename in self._cl_typenames[i]:
					if (not isinstance(arg, np.ndarray) and len(arg) != count) or \
					   (	isinstance(arg, np.ndarray) and arg.size != count):
						print('Error in GPU::__call__(self, *args):')
						print(typename + ' need ' + str(count) + ' elements')
						print('But you passed ' + str(len(arg)) + ' elements')
						exit(-1)
					func_name = 'cl.cltypes.make_' + typename
					str_args = str(arg[0])
					for i in range(1, count):
						str_args += (', ' + str(arg[i]))
					return eval(func_name + '(' + str_args + ')')

			typename = base
			ptr_typename = typename + '*'
			if ptr_typename in self._cl_typenames[i]:
				return np.array(arg, dtype=eval('cl.cltypes.' + typename))

			if typename in self._cl_typenames[i]:
				return eval('cl.cltypes.' + base + '(' + str(arg) + ')')
		exit(-1)

	def _set_arg(self, i, arg):
		counts = [2, 3, 4, 8, 16]
		for base in self._cl_basenames:
			for count in counts:
				typename = base + str(count)
				ptr_typename = typename + "*"
				if ptr_typename in self._cl_typenames[i]:
					arg = np.array(arg, dtype=eval('cl.cltypes.' + base))
					if count == 3 and len(arg.shape) == 3 and arg.shape[2] == 3:
						arg = np.insert(arg, 3, values=0, axis=2)
					if arg.nbytes == self._default_args_origine[i].nbytes:
						cl.enqueue_copy(self._queue, self._real_args[i], arg)
					else:
						self._real_args[i] = self._buffer(arg)
					self._kernel.set_arg(i, self._real_args[i])
					return

				if typename in self._cl_typenames[i]:
					if (not isinstance(arg, np.ndarray) and len(arg) != count) or \
					   (	isinstance(arg, np.ndarray) and arg.size != count):
						print('Error in GPU::__call__(self, *args):')
						print(typename + ' need ' + str(count) + ' elements')
						print('But you passed ' + str(len(arg)) + ' elements')
						exit(-1)
					func_name = 'cl.cltypes.make_' + typename
					str_args = str(arg[0])
					for i in range(1, count):
						str_args += (', ' + str(arg[i]))
					self._real_args[i] = eval(func_name + '(' + str_args + ')')
					self._kernel.set_arg(i, self._real_args[i])
					return

			typename = base
			ptr_typename = typename + '*'
			if ptr_typename in self._cl_typenames[i]:
				arg = np.array(arg, dtype=eval('cl.cltypes.' + typename))
				if arg.nbytes == self._default_args_origine[i].nbytes:
					cl.enqueue_copy(self._queue, self._real_args[i], arg)
				else:
					self._real_args[i] = self._buffer(arg)
				self._kernel.set_arg(i, self._real_args[i])
				return

			if typename in self._cl_typenames[i]:
				self._real_args[i] = eval('cl.cltypes.' + base + '(' + str(arg) + ')')
				self._kernel.set_arg(i, self._real_args[i])
				return
		exit(-1)

	def _get_cl_typenames_in_kernel(self, file_name, function_name):
		code = open(file_name).read()
		it_function_name = code.find(function_name, 1)
		if it_function_name == -1:
			print("Error in GPU::setProgram(file_name, function_name):")
			print("There are no function named \"", function_name, "\" in file \"", file_name, "\"")
			exit(-1)
		it_left_brace = code.find("(", it_function_name)
		it_right_brace = code.find(")", it_left_brace)
		variables = code[it_left_brace+1 : it_right_brace].split(",")
		self._cl_typenames = []
		self._varying_args = []
		for j in range(len(variables)):
			variable = variables[j]
			if "__global" in variable:
				self._varying_args.append(j)

			i = variable.find('*')
			if i != -1:
				i -= 1
				while variable[i] == ' ':
					variable = variable[:i] + variable[i+1:]
					i -= 1

			for it_end in range(len(variable)-1, -1, -1):
				if variable[it_end] != " ":
					break

			flag = False
			for it_typename_end in range(it_end, -1, -1):
				if variable[it_typename_end] == " ":
					flag = True
				if flag and variable[it_typename_end] != " ":
					it_typename_end += 1
					break
			for it_typename_begin in range(it_typename_end-1, -1, -1):
				if variable[it_typename_begin] == " ":
					it_typename_begin += 1
					break
			self._cl_typenames.append(variable[it_typename_begin:it_typename_end])

	def _dtype(self, cl_type):
		remove_digits = str.maketrans('', '', digits)
		cl_type = cl_type.translate(remove_digits)
		return self._np_basenames[self._cl_basenames.index(cl_type.replace('*', ''))]

	def _buffer(self, arg):
		if isinstance(arg, int):
			return cl.Buffer(self._context, cl.mem_flags.READ_WRITE, arg)
		elif isinstance(arg, np.ndarray):
			return cl.Buffer(self._context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=arg)

	def _write_only_buffer(self, nbytes):
		return cl.Buffer(self._context, cl.mem_flags.WRITE_ONLY, nbytes)

	def _read_only_buffer(self, arg):
		if isinstance(arg, int):
			return cl.Buffer(self._context, cl.mem_flags.READ_ONLY, arg)
		elif isinstance(arg, np.ndarray):
			return cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=arg)
		
	def _type_transform(self, arg, cl_typename):
		counts = [2, 3, 4, 8, 16]
		for base in self._cl_basenames:
			for count in counts:
				typename = base + str(count)
				ptr_typename = typename + "*"
				if ptr_typename in cl_typename:
					arg = np.array(arg, dtype=eval('cl.cltypes.' + base))
					if count == 3 and len(arg.shape) == 3 and arg.shape[2] == 3:
						arg = np.insert(arg, 3, values=1, axis=2)
					t1 = time.perf_counter()
					device_buffer = self._buffer(arg)
					t2 = time.perf_counter()
					self._host2device_time += (t2 - t1)
					self._default_args_origine.append(arg)
					return device_buffer

				if typename in cl_typename:
					if (not isinstance(arg, np.ndarray) and len(arg) != count) or \
					   (	isinstance(arg, np.ndarray) and arg.size != count):
						print('Error in GPU::set_args(self, *args):')
						print(typename + ' need ' + str(count) + ' elements')
						print('But you passed ' + str(len(arg)) + ' elements')
						exit(-1)
					func_name = 'cl.cltypes.make_' + typename
					str_args = str(arg[0])
					for i in range(1, count):
						str_args += (', ' + str(arg[i]))
					self._default_args_origine.append(0)
					return eval(func_name + '(' + str_args + ')')
			typename = base
			ptr_typename = typename + '*'
			if ptr_typename in cl_typename:
				arg = np.array(arg, dtype=eval('cl.cltypes.' + typename))
				t1 = time.perf_counter()
				device_buffer = self._buffer(arg)
				t2 = time.perf_counter()
				self._host2device_time += (t2 - t1)
				self._default_args_origine.append(arg)
				return device_buffer

			if typename in cl_typename:
				self._default_args_origine.append(0)
				return eval('cl.cltypes.' + base + '(' + str(arg) + ')')
		exit(-1)

class AllGPUs:
	def __init__(self):
		Intel_GPUs = []
		AMD_GPUs = []
		Nvidia_GPUs = []

		platforms  = cl.get_platforms()
		for i in range(len(platforms)):
		    devices = platforms[i].get_devices(cl.device_type.GPU)
		    for j in range(len(devices)):
		        if "intel" in devices[j].get_info(cl.device_info.NAME).lower():
		        	Intel_GPUs.append(GPU(i, j))
		        elif "amd" in devices[j].get_info(cl.device_info.NAME).lower():
		        	AMD_GPUs.append(GPU(i, j))
		        else:
		        	Nvidia_GPUs.append(GPU(i, j))
		        	
		self.GPUs = Nvidia_GPUs + Intel_GPUs + AMD_GPUs
		self.n_GPUs = len(self.GPUs)
		self._n_missions = 0
		self.current_gpu = 0
		self.mission_map_to_gpu = []
		self._is_called = False

	def set_program(self, file_name, function_name):
		for gpu in self.GPUs:
			gpu.set_program(file_name, function_name)

	def set_return(self, *args):
		for gpu in self.GPUs:
			gpu.set_return(*args)

	def set_args(self, *args):
		for gpu in self.GPUs:
			gpu.set_args(*args)

	def add_mission(self, *args):
		self.mission_map_to_gpu.append((self.current_gpu, self.GPUs[self.current_gpu]._n_missions))
		self.GPUs[self.current_gpu].add_mission(*args)
		self.current_gpu += 1
		if self.current_gpu >= self.n_GPUs:
			self.current_gpu = 0
		self._n_missions += 1

	def __call__(self, *args):
		self._is_called = True
		return self.GPUs[0](*args)

	def run(self):
		self._is_called = False

		thread_list = []
		for gpu in self.GPUs:
			if gpu._n_missions > 0:
				t = Thread(target=gpu.run, args=())
				thread_list.append(t)
				t.start()
		for t in thread_list:
			t.join()

	def result(self, i):
		gpu_mission = self.mission_map_to_gpu[i]
		return self.GPUs[gpu_mission[0]].result(gpu_mission[1])

	def clear(self):
		for gpu in self.GPUs:
			gpu.clear()
		self._n_missions = 0
		self.current_gpu = 0
		self.mission_map_to_gpu = []

	def clear_missions(self):
		for gpu in self.GPUs:
			gpu.clear_missions()
		self._n_missions = 0
		self.current_gpu = 0
		self.mission_map_to_gpu = []

	def print_performance(self):
		total_time = 0
		for gpu in self.GPUs:
			if gpu._n_missions > 0:
				total_time += gpu.total_time()

		if self._is_called:
			num = 1
		else:
			num = self._n_missions
		print("Processed", num, "missions in", round(1000*total_time, 2), "ms")
		print("Total   time:", round(1000*total_time, 2), "ms")
		print("Average time:", round(1000*total_time/self._n_missions, 2), "ms")
		print("For each device:")
		for gpu in self.GPUs:
			if gpu._n_missions > 0:
				print()
				gpu.print_performance()

	@staticmethod
	def list_devices():
		platforms  = cl.get_platforms()
		for i in range(len(platforms)):
		    devices = platforms[i].get_devices(cl.device_type.GPU)
		    for j in range(len(devices)):
		        print("(", i, ",", j, "):", devices[j].get_info(cl.device_info.NAME))

	@staticmethod
	def print_info():
		print("PyOpenCL Version:", cl.VERSION)
		print("OpenCL Head Version:", cl.get_cl_header_version())
		print()
		 
		platforms  = cl.get_platforms()
		print("Platforms Amount:", len(platforms))
		 
		for plat in platforms:
		    print("Platform:", plat.get_info(cl.platform_info.NAME))
		    print("--Platform Profile:", plat.get_info(cl.platform_info.PROFILE))
		    print("--Platform Vendor:", plat.get_info(cl.platform_info.VENDOR))
		    print("--Platform Version:", plat.get_info(cl.platform_info.VERSION))
		 
		    devices = plat.get_devices(cl.device_type.GPU)
		    print("--GPU Amount:", len(devices))
		 
		    for device in devices:
		        print("--GPU:", device.get_info(cl.device_info.NAME))
		        print("----OpenCL Version:",device.get_info(cl.device_info.OPENCL_C_VERSION))
		        print("----GPU Vendor:",device.get_info(cl.device_info.VENDOR))
		        print("----GPU Version:",device.get_info(cl.device_info.VERSION))
		        print("----GPU Driver Version:",device.get_info(cl.device_info.DRIVER_VERSION))
		        print("----Max Work Group Size:",device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE))
		        print("----Max Compute Units:",device.get_info(cl.device_info.MAX_COMPUTE_UNITS))
		        print("----Max Work Item Size:",device.get_info(cl.device_info.MAX_WORK_ITEM_SIZES))
		        print("----Local Memory Size:",device.get_info(cl.device_info.LOCAL_MEM_SIZE)/1024, 'KB')
		        print()