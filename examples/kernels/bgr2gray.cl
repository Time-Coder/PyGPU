__kernel void bgr2gray(__global uchar*  image_gray,
					   __global uchar3* image_bgr)
{
	int i = get_global_id(0);
	image_gray[i] = (uchar)(0.11 * image_bgr[i].x + 
							0.59 * image_bgr[i].y +
							0.3  * image_bgr[i].z);
}