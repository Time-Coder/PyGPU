__kernel void sum(__global float* result,
				  __global float* a,
				  __global float* b)
{
	int i = get_global_id(0);
	result[i] = a[i] + b[i];
}