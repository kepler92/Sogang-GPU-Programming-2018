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
#include "reduction_cpu.h"
//////////////////////////////////////////////////////////////////////////

#define USING_GLOBAL_MEMORY	0
#define USING_LOCAL_MEMORY	1


typedef struct _OPENCL_C_PROG_SRC {
	size_t length;
	char *string;
} OPENCL_C_PROG_SRC;


float* reduction_1D_OpenCL(float *data, size_t n_elements, size_t work_group_size) {
	cl_int errcode_ret;
	float compute_time;

	size_t n_work_group;
	float *partial_sum, *output;
	float result[2];
	OPENCL_C_PROG_SRC prog_src;

	cl_platform_id platform;
	cl_device_id devices;
	cl_context context;
	cl_command_queue cmd_queues;
	cl_program program;
	cl_kernel kernel[2];
	//cl_mem  buffer_data, buffer_partial_sum, buffer_output;
	cl_mem  buffer_data, buffer_output;
	cl_event event_for_timing;

	n_work_group = n_elements / work_group_size;
	output = (float*)malloc(sizeof(float)*n_work_group);

	errcode_ret = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR_CODE(errcode_ret);

	errcode_ret = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &devices, NULL);
	CHECK_ERROR_CODE(errcode_ret);

	//fprintf(stdout, "\n^^^ The first GPU device on the platform ^^^\n");
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


	fprintf(stdout, "\n > On Global Memory\n");

	kernel[USING_GLOBAL_MEMORY] = clCreateKernel(program, "reduction_1d_global", &errcode_ret);
	CHECK_ERROR_CODE(errcode_ret);
		
	buffer_data = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*n_elements, NULL, &errcode_ret);
	CHECK_ERROR_CODE(errcode_ret);

	buffer_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*n_work_group, NULL, &errcode_ret);
	CHECK_ERROR_CODE(errcode_ret);

	fprintf(stdout, "   [Data Transfer Host to Devece(GPU)] \n");

	CHECK_TIME_START;
	// Move the input data from the host memory to the GPU device memory.
	errcode_ret = clEnqueueWriteBuffer(cmd_queues, buffer_data, CL_FALSE, 0, sizeof(float)*n_elements, data, 0, NULL, NULL);
	CHECK_ERROR_CODE(errcode_ret);

	clFinish(cmd_queues); // What if this line is removed?
	CHECK_TIME_END(compute_time);
	CHECK_ERROR_CODE(errcode_ret);

	fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);

	errcode_ret = clSetKernelArg(kernel[USING_GLOBAL_MEMORY], 0, sizeof(cl_mem), &buffer_data);
	CHECK_ERROR_CODE(errcode_ret);
	errcode_ret = clSetKernelArg(kernel[USING_GLOBAL_MEMORY], 1, sizeof(cl_mem), &buffer_output);
	CHECK_ERROR_CODE(errcode_ret);

	fprintf(stdout, "   [Kernel Execution] \n");

	CHECK_TIME_START;
	errcode_ret = clEnqueueNDRangeKernel(cmd_queues, kernel[USING_GLOBAL_MEMORY], 1, NULL, &n_elements, &work_group_size, 0, NULL, &event_for_timing);
	CHECK_ERROR_CODE(errcode_ret);
	clFinish(cmd_queues);  // What would happen if this line is removed?
									  // or clWaitForEvents(1, &event_for_timing);
	CHECK_TIME_END(compute_time);
	CHECK_ERROR_CODE(errcode_ret);

	fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	print_device_time(event_for_timing);

	fprintf(stdout, "   [Data Transfer Device to Host] \n");

	memset(output, 0, sizeof(float)*n_work_group);

	CHECK_TIME_START;
	errcode_ret = clEnqueueReadBuffer(cmd_queues, buffer_output, CL_TRUE, 0, sizeof(float)*n_work_group, output, 0, NULL, &event_for_timing);
	CHECK_TIME_END(compute_time);
	CHECK_ERROR_CODE(errcode_ret);

	result[0] = 0.0f;
	reduction_on_the_cpu_reduction(output, &result[0], n_work_group);

	fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	print_device_time(event_for_timing);

	fprintf(stdout, "   [Check Results] \n");
	fprintf(stdout, "     * Output = %f\n\n", result[0]);


	fprintf(stdout, "\n > On Local Memory\n");

	kernel[USING_LOCAL_MEMORY] = clCreateKernel(program, "reduction_1d_local", &errcode_ret);
	CHECK_ERROR_CODE(errcode_ret);

	//buffer_data = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float)*n_elements, NULL, &errcode_ret);
	//CHECK_ERROR_CODE(errcode_ret);

	//buffer_partial_sum = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*work_items, NULL, &errcode_ret);
	//CHECK_ERROR_CODE(errcode_ret);

	//buffer_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*n_work_group, NULL, &errcode_ret);
	//CHECK_ERROR_CODE(errcode_ret);

	fprintf(stdout, "   [Data Transfer Host to Devece(GPU)] \n");

	CHECK_TIME_START;
	// Move the input data from the host memory to the GPU device memory.
	errcode_ret = clEnqueueWriteBuffer(cmd_queues, buffer_data, CL_FALSE, 0, sizeof(float)*n_elements, data, 0, NULL, NULL);
	CHECK_ERROR_CODE(errcode_ret);

	clFinish(cmd_queues); // What if this line is removed?
	CHECK_TIME_END(compute_time);
	CHECK_ERROR_CODE(errcode_ret);

	fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);

	errcode_ret = clSetKernelArg(kernel[USING_LOCAL_MEMORY], 0, sizeof(cl_mem), &buffer_data);
	CHECK_ERROR_CODE(errcode_ret);
	errcode_ret = clSetKernelArg(kernel[USING_LOCAL_MEMORY], 1, sizeof(cl_float) * work_group_size, NULL);
	CHECK_ERROR_CODE(errcode_ret);
	errcode_ret = clSetKernelArg(kernel[USING_LOCAL_MEMORY], 2, sizeof(cl_mem), &buffer_output);
	CHECK_ERROR_CODE(errcode_ret);

	fprintf(stdout, "   [Kernel Execution] \n");

	CHECK_TIME_START;
	errcode_ret = clEnqueueNDRangeKernel(cmd_queues, kernel[USING_LOCAL_MEMORY], 1, NULL, &n_elements, &work_group_size, 0, NULL, &event_for_timing);
	CHECK_ERROR_CODE(errcode_ret);
	clFinish(cmd_queues);  // What would happen if this line is removed?
						   // or clWaitForEvents(1, &event_for_timing);
	CHECK_TIME_END(compute_time);
	CHECK_ERROR_CODE(errcode_ret);
	
	fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	print_device_time(event_for_timing);

	fprintf(stdout, "   [Data Transfer Device to Host] \n");

	memset(output, 0, sizeof(float)*n_work_group);

	CHECK_TIME_START;
	errcode_ret = clEnqueueReadBuffer(cmd_queues, buffer_output, CL_TRUE, 0, sizeof(float)*n_work_group, output, 0, NULL, &event_for_timing);
	CHECK_TIME_END(compute_time);
	CHECK_ERROR_CODE(errcode_ret);

	result[1] = 0.0f;
	reduction_on_the_cpu_reduction(output, &result[1], n_work_group);

	fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	print_device_time(event_for_timing);

	fprintf(stdout, "   [Check Results] \n");
	fprintf(stdout, "     * Output = %f\n\n", result[1]);

	/* Free OpenCL resources. */
	clReleaseMemObject(buffer_data);
	//clReleaseMemObject(buffer_partial_sum);
	clReleaseMemObject(buffer_output);
	clReleaseKernel(kernel[USING_GLOBAL_MEMORY]);
	clReleaseKernel(kernel[USING_LOCAL_MEMORY]);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmd_queues);
	clReleaseContext(context);

	/* Free host resources. */
	free(output);
	free(prog_src.string);

	return result;
}


void reduction_1D(size_t work_group_size) {
	float *data;
	size_t n_elements;

	float compute_time;
	float general_cpu_result;
	float *opencl_gpu_result;

	n_elements = 128 * 1024 * 1024;
	//n_elements = 512 * 1024;

	data = (float*)malloc(sizeof(float)*n_elements);

	fprintf(stdout, "^^^ Generating random input arrays with %d elements each...\n", (int)n_elements);
	srand((unsigned int)201803); // Always the same input data
	for (int i = 0; i < (int)n_elements; i++)
		data[i] = 3.1415926f*((float)rand() / RAND_MAX);
	fprintf(stdout, "^^^ Done!\n");

	fprintf(stdout, "\n- General CPU computation ^^^\n");
	fprintf(stdout, "   [CPU Execution] \n");
	CHECK_TIME_START;
	//reduition_on_the_cpu(data, output, (int)n_elements);
	//reduction_on_the_cpu_reduction(data, output, (int)n_elements);
	reduction_on_the_CPU_KahanSum(data, &general_cpu_result, (int)n_elements);
	CHECK_TIME_END(compute_time);

	fprintf(stdout, "     * Time by host clock = %.3fms\n\n", compute_time);
	fprintf(stdout, "   [Check Results] \n");
	fprintf(stdout, "     * Output = %f\n\n", general_cpu_result);

	fprintf(stdout, "\n- Computing on OpenCL GPU Device ^^^\n");
	opencl_gpu_result = reduction_1D_OpenCL(data, n_elements, work_group_size);

	free(data);
}


float* reduction_2D_OpenCL(float **data, size_t* elements_size, size_t work_group_size) {
	return NULL;
}


void reduction_2D(size_t work_group_size) {

}


int main(void) {
	size_t work_group_size;
	work_group_size = 128;

	fprintf(stdout, "=== Reduction On One Dimmension ===\n\n");
	reduction_1D(work_group_size);
}

