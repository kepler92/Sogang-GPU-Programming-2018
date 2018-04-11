//
//  main.cpp
//  Simple_SIMT
//
//  Written for CSEG437/CSE5437
//  Department of Computer Science and Engineering
//  Copyright © 2018년 Sogang University. All rights reserved.
//

#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "my_OpenCL_util.h"

//#define COALESCED_GLOBAL_MEMORY_ACCESS  // What happens if this line is commented out?
//
//#ifdef COALESCED_GLOBAL_MEMORY_ACCESS
//#define OPENCL_C_PROG_FILE_NAME "simple_kernel.cl"
//#define KERNEL_NAME "CombineTwoArrays"
//#else
//#define OPENCL_C_PROG_FILE_NAME "simple_kernel2.cl"
//#define KERNEL_NAME "CombineTwoArrays2"
//#endif

//////////////////////////////////////////////////////////////////////////
void generate_random_float_array(float *array, int n) {
	srand((unsigned int)201803); // Always the same input data
	for (int i = 0; i < n; i++) {
		array[i] = 3.1415926f*((float)rand() / RAND_MAX);
	}
}
//////////////////////////////////////////////////////////////////////////

#define GPU_GLOBAL_MEMORY	0
#define GPU_LOCAL_MEMORY	1

#define NUMBER_OF_GPU_METHOD	GPU_LOCAL_MEMORY + 1
#define IS_DEBUG	0


typedef struct _OPENCL_C_PROG_SRC {
	size_t length;
	char *string;
} OPENCL_C_PROG_SRC;

typedef struct _REDUCTION_RESULT {
	char type[20];
	float result;
	float total_time;
	float kernel_time;
} REDUCTION_RESULT;


#include <float.h>		// For FLT_EPSILON
#include "reduction_cpu.h"


void print_result(REDUCTION_RESULT cpu_result, REDUCTION_RESULT* gpu_result, int number_of_gpu_result = NUMBER_OF_GPU_METHOD) {
	bool check_flag = true;
	for (int i = 0; i < number_of_gpu_result; i++) {
		if (fabsf(cpu_result.result - gpu_result[i].result) > FLT_EPSILON) {
			check_flag = false;
			break;
		}
	}

	int number_of_total_result = number_of_gpu_result + 1;
	REDUCTION_RESULT* result = (REDUCTION_RESULT*)malloc(sizeof(REDUCTION_RESULT) * (number_of_total_result));

	strcpy(result[0].type, cpu_result.type);
	result[0].result = cpu_result.result;
	result[0].total_time = cpu_result.total_time;
	result[0].kernel_time = cpu_result.kernel_time;

	for (int i = 0; i < number_of_gpu_result; i++) {
		strcpy(result[i + 1].type, gpu_result[i].type);
		result[i + 1].result = gpu_result[i].result;
		result[i + 1].total_time = gpu_result[i].total_time;
		result[i + 1].kernel_time = gpu_result[i].kernel_time;
	}

	for (int i = 0; i < number_of_total_result; i++) {
		printf("     [%s Execution] \n", result[i].type);
		printf("       Total Time by host clock = %fms \n", result[i].total_time);
		printf("       Kernel Time by host clock = %fms \n", result[i].kernel_time);
		printf("\n");
	}

	printf("     + Check ");
	if (check_flag == true)	printf("PASSED!\n");
	else					printf("FAILED!\n");

	printf("         [");
	for (int i = 0; i < number_of_total_result; i++) {
		if (i == 0)		printf("%f(%s)", result[i].result, result[i].type);
		else			printf("/%f(%s)", result[i].result, result[i].type);
	}
	printf("]\n\n");

	free(result);
}


REDUCTION_RESULT* reduction_1D_OpenCL(float *data, size_t n_elements, size_t work_group_size) {
	cl_int errcode_ret;
	float compute_time;

	size_t n_work_group;
	float *partial_sum, *output;
	OPENCL_C_PROG_SRC prog_src;

	cl_platform_id platform;
	cl_device_id devices;
	cl_context context;
	cl_command_queue cmd_queues;
	cl_program program;
	cl_kernel kernel[NUMBER_OF_GPU_METHOD];
	cl_mem  buffer_data, buffer_output;
	cl_event event_for_timing;

	static REDUCTION_RESULT result[NUMBER_OF_GPU_METHOD];
	float sum_output;
	float total_time;
	float kernel_time;

	n_work_group = n_elements / work_group_size;
	output = (float*)malloc(sizeof(float)*n_work_group);

	total_time = 0.0f;
	kernel_time = 0.0f;

	
	errcode_ret = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR_CODE(errcode_ret);

	errcode_ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &devices, NULL);
	CHECK_ERROR_CODE(errcode_ret);

	//if(IS_DEBUG) fprintf(stdout, "\n^^^ The first GPU device on the platform ^^^\n");
	//print_device_0(devices);

	context = clCreateContext(NULL, 1, &devices, NULL, NULL, &errcode_ret);
	CHECK_ERROR_CODE(errcode_ret);

	cmd_queues = clCreateCommandQueue(context, devices, CL_QUEUE_PROFILING_ENABLE, &errcode_ret);
	CHECK_ERROR_CODE(errcode_ret);

	prog_src.length = read_kernel_from_file("reduction_1D.cl", &prog_src.string);

	program = clCreateProgramWithSource(context, 1, (const char **)&prog_src.string, &prog_src.length, &errcode_ret);
	CHECK_ERROR_CODE(errcode_ret);

	errcode_ret = clBuildProgram(program, 1, &devices, NULL, NULL, NULL);
	CHECK_ERROR_CODE(errcode_ret);


	if(IS_DEBUG) fprintf(stdout, "\n ## Test: On Global Memory\n");

	kernel[GPU_GLOBAL_MEMORY] = clCreateKernel(program, "reduction_global", &errcode_ret);
	CHECK_ERROR_CODE(errcode_ret);
		
	buffer_data = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*n_elements, NULL, &errcode_ret);
	CHECK_ERROR_CODE(errcode_ret);

	buffer_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*n_work_group, NULL, &errcode_ret);
	CHECK_ERROR_CODE(errcode_ret);

	if(IS_DEBUG) fprintf(stdout, "   [Data Transfer Host to Devece(GPU)] \n");

	CHECK_TIME_START;
	// Move the input data from the host memory to the GPU device memory.
	errcode_ret = clEnqueueWriteBuffer(cmd_queues, buffer_data, CL_FALSE, 0, sizeof(float)*n_elements, data, 0, NULL, NULL);
	CHECK_ERROR_CODE(errcode_ret);

	clFinish(cmd_queues); // What if this line is removed?
	CHECK_TIME_END(compute_time); total_time += compute_time;
	CHECK_ERROR_CODE(errcode_ret);

	if(IS_DEBUG) fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);

	errcode_ret = clSetKernelArg(kernel[GPU_GLOBAL_MEMORY], 0, sizeof(cl_mem), &buffer_data);
	CHECK_ERROR_CODE(errcode_ret);
	errcode_ret = clSetKernelArg(kernel[GPU_GLOBAL_MEMORY], 1, sizeof(cl_mem), &buffer_output);
	CHECK_ERROR_CODE(errcode_ret);

	if(IS_DEBUG) fprintf(stdout, "   [Kernel Execution] \n");

	CHECK_TIME_START;
	errcode_ret = clEnqueueNDRangeKernel(cmd_queues, kernel[GPU_GLOBAL_MEMORY], 1, NULL, &n_elements, &work_group_size, 0, NULL, &event_for_timing);
	CHECK_ERROR_CODE(errcode_ret);
	clFinish(cmd_queues);  // What would happen if this line is removed?
									  // or clWaitForEvents(1, &event_for_timing);
	CHECK_TIME_END(compute_time); total_time += compute_time; kernel_time = compute_time;
	CHECK_ERROR_CODE(errcode_ret);

	if(IS_DEBUG) fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	if(IS_DEBUG) print_device_time(event_for_timing);

	if(IS_DEBUG) fprintf(stdout, "   [Data Transfer Device to Host] \n");

	memset(output, 0, sizeof(float)*n_work_group);

	CHECK_TIME_START;
	errcode_ret = clEnqueueReadBuffer(cmd_queues, buffer_output, CL_TRUE, 0, sizeof(float)*n_work_group, output, 0, NULL, &event_for_timing);
	CHECK_TIME_END(compute_time); total_time += compute_time;
	CHECK_ERROR_CODE(errcode_ret);

	sum_output = 0.0f;
	reduction_1d_on_the_cpu_reduction(output, &sum_output, n_work_group);

	if(IS_DEBUG) fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	if(IS_DEBUG) print_device_time(event_for_timing);

	if(IS_DEBUG) fprintf(stdout, "   [Check Results] \n");
	if(IS_DEBUG) fprintf(stdout, "     * Output = %f\n\n", sum_output);


	strcpy(result[GPU_GLOBAL_MEMORY].type, "GLOBAL");
	result[GPU_GLOBAL_MEMORY].result = sum_output;
	result[GPU_GLOBAL_MEMORY].total_time = total_time; total_time = 0.0f;
	result[GPU_GLOBAL_MEMORY].kernel_time = kernel_time;	


	if(IS_DEBUG) fprintf(stdout, "\n ## Test: On Local Memory\n");

	kernel[GPU_LOCAL_MEMORY] = clCreateKernel(program, "reduction_local", &errcode_ret);
	CHECK_ERROR_CODE(errcode_ret);

	//buffer_data = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*n_elements, NULL, &errcode_ret);
	//CHECK_ERROR_CODE(errcode_ret);

	//buffer_partial_sum = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*work_items, NULL, &errcode_ret);
	//CHECK_ERROR_CODE(errcode_ret);

	//buffer_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*n_work_group, NULL, &errcode_ret);
	//CHECK_ERROR_CODE(errcode_ret);

	if(IS_DEBUG) fprintf(stdout, "   [Data Transfer Host to Devece(GPU)] \n");

	CHECK_TIME_START;
	// Move the input data from the host memory to the GPU device memory.
	errcode_ret = clEnqueueWriteBuffer(cmd_queues, buffer_data, CL_FALSE, 0, sizeof(float)*n_elements, data, 0, NULL, NULL);
	CHECK_ERROR_CODE(errcode_ret);

	clFinish(cmd_queues); // What if this line is removed?
	CHECK_TIME_END(compute_time); total_time += compute_time;
	CHECK_ERROR_CODE(errcode_ret);

	if(IS_DEBUG) fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);

	errcode_ret = clSetKernelArg(kernel[GPU_LOCAL_MEMORY], 0, sizeof(cl_mem), &buffer_data);
	CHECK_ERROR_CODE(errcode_ret);
	errcode_ret = clSetKernelArg(kernel[GPU_LOCAL_MEMORY], 1, sizeof(cl_float) * work_group_size, NULL);
	CHECK_ERROR_CODE(errcode_ret);
	errcode_ret = clSetKernelArg(kernel[GPU_LOCAL_MEMORY], 2, sizeof(cl_mem), &buffer_output);
	CHECK_ERROR_CODE(errcode_ret);

	if(IS_DEBUG) fprintf(stdout, "   [Kernel Execution] \n");

	CHECK_TIME_START;
	errcode_ret = clEnqueueNDRangeKernel(cmd_queues, kernel[GPU_LOCAL_MEMORY], 1, NULL, &n_elements, &work_group_size, 0, NULL, &event_for_timing);
	CHECK_ERROR_CODE(errcode_ret);
	clFinish(cmd_queues);  // What would happen if this line is removed?
						   // or clWaitForEvents(1, &event_for_timing);
	CHECK_TIME_END(compute_time); total_time += compute_time; kernel_time = compute_time;
	CHECK_ERROR_CODE(errcode_ret);
	
	if(IS_DEBUG) fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	if(IS_DEBUG) print_device_time(event_for_timing);

	if(IS_DEBUG) fprintf(stdout, "   [Data Transfer Device to Host] \n");

	memset(output, 0, sizeof(float)*n_work_group);

	CHECK_TIME_START;
	errcode_ret = clEnqueueReadBuffer(cmd_queues, buffer_output, CL_TRUE, 0, sizeof(float)*n_work_group, output, 0, NULL, &event_for_timing);
	CHECK_TIME_END(compute_time); total_time += compute_time;
	CHECK_ERROR_CODE(errcode_ret);

	sum_output = 0.0f;
	reduction_1d_on_the_cpu_reduction(output, &sum_output, n_work_group);

	if(IS_DEBUG) fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	if(IS_DEBUG) print_device_time(event_for_timing);

	if(IS_DEBUG) fprintf(stdout, "   [Check Results] \n");
	if(IS_DEBUG) fprintf(stdout, "     * Output = %f\n\n", sum_output);


	strcpy(result[GPU_LOCAL_MEMORY].type, "LOCAL");
	result[GPU_LOCAL_MEMORY].result = sum_output;
	result[GPU_LOCAL_MEMORY].total_time = total_time; total_time = 0.0f;
	result[GPU_LOCAL_MEMORY].kernel_time = kernel_time;


	/* Free OpenCL resources. */
	clReleaseMemObject(buffer_data);
	//clReleaseMemObject(buffer_partial_sum);
	clReleaseMemObject(buffer_output);
	clReleaseKernel(kernel[GPU_GLOBAL_MEMORY]);
	clReleaseKernel(kernel[GPU_LOCAL_MEMORY]);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmd_queues);
	clReleaseContext(context);

	/* Free host resources. */
	free(output);
	free(prog_src.string);

	return result;
}


REDUCTION_RESULT* reduction_2D_OpenCL(float *data, size_t* elements_size, size_t* work_group_size) {
	cl_int errcode_ret;
	float compute_time;

	size_t n_elements, n_work_group, work_group_area;
	float *partial_sum, *output;
	OPENCL_C_PROG_SRC prog_src;

	cl_platform_id platform;
	cl_device_id devices;
	cl_context context;
	cl_command_queue cmd_queues;
	cl_program program;
	cl_kernel kernel[2];
	cl_mem  buffer_data, buffer_output;
	cl_event event_for_timing;
		
	static REDUCTION_RESULT result[NUMBER_OF_GPU_METHOD];
	float sum_output;
	float total_time;
	float kernel_time;

	n_elements = elements_size[0] * elements_size[1];
	work_group_area = work_group_size[0] * work_group_size[1];
	n_work_group = n_elements / work_group_size[1];

	output = (float*)malloc(sizeof(float)*n_work_group);

	total_time = 0.0f;
	kernel_time = 0.0f;


	errcode_ret = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR_CODE(errcode_ret);

	errcode_ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &devices, NULL);
	CHECK_ERROR_CODE(errcode_ret);

	//if(IS_DEBUG) fprintf(stdout, "\n^^^ The first GPU device on the platform ^^^\n");
	//print_device_0(devices);

	context = clCreateContext(NULL, 1, &devices, NULL, NULL, &errcode_ret);
	CHECK_ERROR_CODE(errcode_ret);

	cmd_queues = clCreateCommandQueue(context, devices, CL_QUEUE_PROFILING_ENABLE, &errcode_ret);
	CHECK_ERROR_CODE(errcode_ret);

	prog_src.length = read_kernel_from_file("reduction_2D.cl", &prog_src.string);

	program = clCreateProgramWithSource(context, 1, (const char **)&prog_src.string, &prog_src.length, &errcode_ret);
	CHECK_ERROR_CODE(errcode_ret);

	errcode_ret = clBuildProgram(program, 1, &devices, NULL, NULL, NULL);
	CHECK_ERROR_CODE(errcode_ret);


	if(IS_DEBUG) fprintf(stdout, "\n ## Test: On Global Memory\n");

	kernel[GPU_GLOBAL_MEMORY] = clCreateKernel(program, "reduction_global", &errcode_ret);
	CHECK_ERROR_CODE(errcode_ret);

	buffer_data = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*n_elements, NULL, &errcode_ret);
	CHECK_ERROR_CODE(errcode_ret);

	buffer_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*n_work_group, NULL, &errcode_ret);
	CHECK_ERROR_CODE(errcode_ret);

	if(IS_DEBUG) fprintf(stdout, "   [Data Transfer Host to Devece(GPU)] \n");

	CHECK_TIME_START;
	// Move the input data from the host memory to the GPU device memory.
	errcode_ret = clEnqueueWriteBuffer(cmd_queues, buffer_data, CL_FALSE, 0, sizeof(float)*n_elements, data, 0, NULL, NULL);
	CHECK_ERROR_CODE(errcode_ret);

	clFinish(cmd_queues); // What if this line is removed?
	CHECK_TIME_END(compute_time); total_time += compute_time;
	CHECK_ERROR_CODE(errcode_ret);

	if(IS_DEBUG) fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);

	errcode_ret = clSetKernelArg(kernel[GPU_GLOBAL_MEMORY], 0, sizeof(cl_mem), &buffer_data);
	CHECK_ERROR_CODE(errcode_ret);
	errcode_ret = clSetKernelArg(kernel[GPU_GLOBAL_MEMORY], 1, sizeof(cl_mem), &buffer_output);
	CHECK_ERROR_CODE(errcode_ret);

	if(IS_DEBUG) fprintf(stdout, "   [Kernel Execution] \n");

	CHECK_TIME_START;
	errcode_ret = clEnqueueNDRangeKernel(cmd_queues, kernel[GPU_GLOBAL_MEMORY], 2, NULL, elements_size, work_group_size, 0, NULL, &event_for_timing);
	CHECK_ERROR_CODE(errcode_ret);
	clFinish(cmd_queues);  // What would happen if this line is removed?
						   // or clWaitForEvents(1, &event_for_timing);
	CHECK_TIME_END(compute_time); total_time += compute_time; kernel_time = compute_time;
	CHECK_ERROR_CODE(errcode_ret);

	if(IS_DEBUG) fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	if(IS_DEBUG) print_device_time(event_for_timing);

	if(IS_DEBUG) fprintf(stdout, "   [Data Transfer Device to Host] \n");

	memset(output, 0, sizeof(float)*n_work_group);

	CHECK_TIME_START;
	errcode_ret = clEnqueueReadBuffer(cmd_queues, buffer_output, CL_TRUE, 0, sizeof(float)*n_work_group, output, 0, NULL, &event_for_timing);
	CHECK_TIME_END(compute_time); total_time += compute_time;
	CHECK_ERROR_CODE(errcode_ret);

	sum_output = 0.0f;
	reduction_1d_on_the_cpu_reduction(output, &sum_output, n_work_group);

	if(IS_DEBUG) fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	if(IS_DEBUG) print_device_time(event_for_timing);

	if(IS_DEBUG) fprintf(stdout, "   [Check Results] \n");
	if(IS_DEBUG) fprintf(stdout, "     * Output = %f\n\n", sum_output);


	strcpy(result[GPU_GLOBAL_MEMORY].type, "GLOBAL");
	result[GPU_GLOBAL_MEMORY].result = sum_output;
	result[GPU_GLOBAL_MEMORY].total_time = total_time; total_time = 0.0f;
	result[GPU_GLOBAL_MEMORY].kernel_time = kernel_time;


	if(IS_DEBUG) fprintf(stdout, "\n ## Test: On Local Memory\n");

	kernel[GPU_LOCAL_MEMORY] = clCreateKernel(program, "reduction_local", &errcode_ret);
	CHECK_ERROR_CODE(errcode_ret);

	//buffer_data = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*n_elements, NULL, &errcode_ret);
	//CHECK_ERROR_CODE(errcode_ret);

	//buffer_partial_sum = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*work_items, NULL, &errcode_ret);
	//CHECK_ERROR_CODE(errcode_ret);

	//buffer_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*n_work_group, NULL, &errcode_ret);
	//CHECK_ERROR_CODE(errcode_ret);

	if(IS_DEBUG) fprintf(stdout, "   [Data Transfer Host to Devece(GPU)] \n");

	CHECK_TIME_START;
	// Move the input data from the host memory to the GPU device memory.
	errcode_ret = clEnqueueWriteBuffer(cmd_queues, buffer_data, CL_FALSE, 0, sizeof(float)*n_elements, data, 0, NULL, NULL);
	CHECK_ERROR_CODE(errcode_ret);

	clFinish(cmd_queues); // What if this line is removed?
	CHECK_TIME_END(compute_time); total_time += compute_time;
	CHECK_ERROR_CODE(errcode_ret);

	if(IS_DEBUG) fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);

	errcode_ret = clSetKernelArg(kernel[GPU_LOCAL_MEMORY], 0, sizeof(cl_mem), &buffer_data);
	CHECK_ERROR_CODE(errcode_ret);
	errcode_ret = clSetKernelArg(kernel[GPU_LOCAL_MEMORY], 1, sizeof(cl_float) * work_group_area, NULL);
	CHECK_ERROR_CODE(errcode_ret);
	errcode_ret = clSetKernelArg(kernel[GPU_LOCAL_MEMORY], 2, sizeof(cl_mem), &buffer_output);
	CHECK_ERROR_CODE(errcode_ret);

	if(IS_DEBUG) fprintf(stdout, "   [Kernel Execution] \n");

	CHECK_TIME_START;
	errcode_ret = clEnqueueNDRangeKernel(cmd_queues, kernel[GPU_LOCAL_MEMORY], 2, NULL, elements_size, work_group_size, 0, NULL, &event_for_timing);
	CHECK_ERROR_CODE(errcode_ret);
	clFinish(cmd_queues);  // What would happen if this line is removed?
						   // or clWaitForEvents(1, &event_for_timing);
	CHECK_TIME_END(compute_time); total_time += compute_time; kernel_time = compute_time;
	CHECK_ERROR_CODE(errcode_ret);

	if(IS_DEBUG) fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	if(IS_DEBUG) print_device_time(event_for_timing);

	if(IS_DEBUG) fprintf(stdout, "   [Data Transfer Device to Host] \n");

	memset(output, 0, sizeof(float)*n_work_group);

	CHECK_TIME_START;
	errcode_ret = clEnqueueReadBuffer(cmd_queues, buffer_output, CL_TRUE, 0, sizeof(float)*n_work_group, output, 0, NULL, &event_for_timing);
	CHECK_TIME_END(compute_time); total_time += compute_time;
	CHECK_ERROR_CODE(errcode_ret);

	sum_output = 0.0f;
	reduction_1d_on_the_cpu_reduction(output, &sum_output, n_work_group);

	if(IS_DEBUG) fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	if(IS_DEBUG) print_device_time(event_for_timing);

	if(IS_DEBUG) fprintf(stdout, "   [Check Results] \n");
	if(IS_DEBUG) fprintf(stdout, "     * Output = %f\n\n", sum_output);


	strcpy(result[GPU_LOCAL_MEMORY].type, "LOCAL");
	result[GPU_LOCAL_MEMORY].result = sum_output;
	result[GPU_LOCAL_MEMORY].total_time = total_time; total_time = 0.0f;
	result[GPU_LOCAL_MEMORY].kernel_time = kernel_time;


	/* Free OpenCL resources. */
	clReleaseMemObject(buffer_data);
	//clReleaseMemObject(buffer_partial_sum);
	clReleaseMemObject(buffer_output);
	clReleaseKernel(kernel[GPU_GLOBAL_MEMORY]);
	clReleaseKernel(kernel[GPU_LOCAL_MEMORY]);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmd_queues);
	clReleaseContext(context);

	/* Free host resources. */
	free(output);
	free(prog_src.string);

	return result;
}


void reduction_1D(void) {
	float *data;
	size_t n_elements, work_group_size;

	float compute_time;
	float output;
	REDUCTION_RESULT general_cpu_result;
	REDUCTION_RESULT *opencl_gpu_result;

	n_elements = 128 * 1024 * 1024;
	work_group_size = 128;

	printf("   Elements size: %d\n", (int)n_elements);
	printf("   Work-group size: %d\n", (int)work_group_size);
	printf("\n\n");

	data = (float*)malloc(sizeof(float)*n_elements);

	if (IS_DEBUG) fprintf(stdout, "^^^ Generating random input array with %d elements...\n", (int)n_elements);
	generate_random_float_array(data, (int)n_elements);
	if (IS_DEBUG) fprintf(stdout, "^^^ Done!\n");

	if (IS_DEBUG) fprintf(stdout, "\n^^^ General CPU computation ^^^\n");
	if (IS_DEBUG) fprintf(stdout, "   [CPU Execution] \n");
	CHECK_TIME_START;
	//reduition_1d_on_the_cpu(data, &general_cpu_result, (int)n_elements);
	reduction_1d_on_the_cpu_reduction(data, &output, (int)n_elements);
	//reduction_1d_on_the_cpu_KahanSum(data, &general_cpu_result, (int)n_elements);
	CHECK_TIME_END(compute_time);

	if (IS_DEBUG) fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	if (IS_DEBUG) fprintf(stdout, "   [Check Results] \n");
	if (IS_DEBUG) fprintf(stdout, "     * Output = %f\n\n", output);

	strcpy(general_cpu_result.type, "CPU");
	general_cpu_result.total_time = compute_time;
	general_cpu_result.kernel_time = compute_time;
	general_cpu_result.result = output;

	if (IS_DEBUG) fprintf(stdout, "\n^^^ Computing on OpenCL GPU Device ^^^\n");
	if (IS_DEBUG) fprintf(stdout, "     * Work Group Size: %d", work_group_size);
	opencl_gpu_result = reduction_1D_OpenCL(data, n_elements, work_group_size);

	print_result(general_cpu_result, opencl_gpu_result);

	free(data);
}


void reduction_2D() {
	float *data;
	size_t elements_size[2], work_group_size[2];
	size_t n_row, n_col;

	float compute_time;
	float output;
	REDUCTION_RESULT general_cpu_result;
	REDUCTION_RESULT *opencl_gpu_result;

	elements_size[0] = n_row = 16 * 1024;	// Matrix Row
	elements_size[1] = n_col = 8 * 1024;	// Matrix Column
	work_group_size[0] = 32;
	work_group_size[1] = 32;

	printf("   Elements size: (%d, %d)\n", (int)n_row, (int)n_col);
	printf("   Work-group size: (%d, %d)\n", (int)work_group_size[0], (int)work_group_size[1]);
	printf("\n\n");

	data = (float*)malloc(sizeof(float*)*n_row*n_col);

	if(IS_DEBUG) fprintf(stdout, "^^^ Generating random input matrix with (%d, %d) elements...\n", (int)n_row, (int)n_col);
	generate_random_float_array(data, (int)n_row*n_col);
	if(IS_DEBUG) fprintf(stdout, "^^^ Done!\n");

	if(IS_DEBUG) fprintf(stdout, "\n^^^ General CPU computation ^^^\n");
	if(IS_DEBUG) fprintf(stdout, "   [CPU Execution] \n");
	CHECK_TIME_START;
	//reduction_2d_on_the_cpu(data, &general_cpu_result, n_row, n_col);
	reduction_2d_on_the_cpu_reduction(data, &output, n_row, n_col);
	//reduction_2d_on_the_cpu_KahanSum(data, &general_cpu_result, n_row, n_col);
	CHECK_TIME_END(compute_time);

	if(IS_DEBUG) fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	if(IS_DEBUG) fprintf(stdout, "   [Check Results] \n");
	if(IS_DEBUG) fprintf(stdout, "     * Output = %f\n\n", output);

	strcpy(general_cpu_result.type, "CPU");
	general_cpu_result.total_time = compute_time;
	general_cpu_result.kernel_time = compute_time;
	general_cpu_result.result = output;

	if(IS_DEBUG) fprintf(stdout, "\n^^^ Computing on OpenCL GPU Device ^^^\n");
	opencl_gpu_result = reduction_2D_OpenCL(data, elements_size, work_group_size);

	print_result(general_cpu_result, opencl_gpu_result);

	free(data);
}


int main(void) {
	printf("=== Reduction; One-Dimmension Data ===\n\n");
	reduction_1D();

	printf("\n\n");

	printf("=== Reduction; Two-Dimmension Data ===\n\n");
	reduction_2D();

	printf("\n\n");

	return 0;
}

