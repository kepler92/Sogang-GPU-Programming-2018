__kernel
void reduction_global(__global float* data, __global float* output) {
	int global_row = get_global_id(0);
	int global_col = get_global_id(1);
	int local_row = get_local_id(0);
	int local_col = get_local_id(1);

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (local_col != 0)
		printf("%d\n", local_col);
}


__kernel
void reduction_local(__global float* data, __local float* partial_sums, __global float* output) {
	int globalRow = get_global_id(0);
	int globalCol = get_global_id(1);

	if (globalCol == 1)
		printf("%d %d\n", globalRow, globalCol);
}