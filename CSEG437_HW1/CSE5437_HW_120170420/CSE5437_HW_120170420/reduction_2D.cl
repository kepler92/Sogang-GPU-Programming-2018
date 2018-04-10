__kernel
void reduction_global(__global float* data, __global float* output) {
	//int local_x = get_local_id(0);
	//int local_y = get_local_id(1);

	//int group_x_size = get_local_size(0);
	//int group_y_size = get_local_size(1);

	//int global_x = get_global_id(0);
	//int global_y = get_global_id(1);

	//int global_x_size = get_global_size(0);
	//int global_y_size = get_global_size(1);


	//barrier(CLK_GLOBAL_MEM_FENCE);

	//for (int i = group_y_size / 2; i > 0; i >>= 1) {
	//	if (local_y < i) {
	//		data[global_x] += data[global_x + ];
	//	}
	//}
}


__kernel
void reduction_local(__global float* data, __local float* partial_sums, __global float* output) {
	int local_x = get_local_id(0);
	int local_y = get_local_id(1);

	int group_x_size = get_local_size(0);
	int group_y_size = get_local_size(1);

	int global_x = get_global_id(0);
	int global_y = get_global_id(1);

	int global_x_size = get_global_size(0);
	int global_y_size = get_global_size(1);

	int local_index = local_y * group_x_size + local_x;
	int global_index = global_y * global_x_size + global_x;

	partial_sums[local_index] = data[global_index];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = group_y_size / 2; i > 0; i >>= 1) {
		if (local_y < i) {
			partial_sums[local_index] += partial_sums[local_index + i * group_x_size];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (local_y == 0) {
		int group_id = (get_group_id(1) * get_num_groups(0) + get_group_id(0)) * group_x_size + local_x;
		output[group_id] = partial_sums[local_x];
	}
}