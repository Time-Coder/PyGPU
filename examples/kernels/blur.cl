__kernel void blur(__global uchar3* image_dest, __global uchar3* image_src, int n_rows, int n_cols,
				   __constant float* mat, int win_width)
{
	int size = n_rows * n_cols;
	int it = get_global_id(0);
	int it_image = it / size;
	int it_in_image = it % size;
	int row = it_in_image / n_cols;
	int col = it_in_image % n_cols;

	int n = (win_width - 1) / 2;
	float3 sum = (float3)(0.0f, 0.0f, 0.0f);
	for(int i = 0; i < win_width; i++)
	{
		int sub_row = row-n+i;
		sub_row = sub_row >= 0 ? (sub_row < n_rows ? sub_row : n_rows-1) : 0;

		for(int j = 0; j < win_width; j++)
		{
			int sub_col = col-n+j;
			sub_col = sub_col >= 0 ? (sub_col < n_cols ? sub_col : n_cols-1) : 0;
			
			int sub_it = it_image * size + sub_row * n_cols + sub_col;
			float k = mat[i*win_width+j];
			sum.x += k * image_src[sub_it].x;
			sum.y += k * image_src[sub_it].y;
			sum.z += k * image_src[sub_it].z;
		}
	}
	image_dest[it].x = (uchar)sum.x;
	image_dest[it].y = (uchar)sum.y;
	image_dest[it].z = (uchar)sum.z;
}