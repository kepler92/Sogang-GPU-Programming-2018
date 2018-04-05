__kernel
//void reduction_1d_global(__global float *data, __global float *partial_sums, __global float *output) {
void reduction_2d_global(__global float *data, __global float* output) {
	int local_id = get_local_id(0);
	int global_id = get_global_id(0);
	int group_size = get_local_size(0);

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = group_size / 2; i > 0; i >>= 1) {
		if (local_id < i) {
			data[global_id] += data[global_id + i];
		}
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	if (local_id == 0) {
		output[get_group_id(0)] = data[global_id];
	}
}


__kernel
void reduction_2d_local(__global float *data, __local float* partial_sums, __global float* output) {
	int local_id = get_local_id(0);
	int group_size = get_local_size(0);

	partial_sums[local_id] = data[get_global_id(0)];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = group_size / 2; i > 0; i >>= 1) {
		if (local_id < i) {
			partial_sums[local_id] += partial_sums[local_id + i];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (local_id == 0) {
		output[get_group_id(0)] = partial_sums[0];
	}
}
