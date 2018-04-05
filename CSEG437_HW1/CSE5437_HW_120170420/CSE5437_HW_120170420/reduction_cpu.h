#pragma once

#include<stdlib.h>
#include<string.h>

void reduition_1d_on_the_cpu(float *data, float *output, int n) {
	int i;
	float sum = 0.0f;

	for (i = 0; i < n; i++)
		sum += data[i];
	
	output[0] = sum;
}

void reduction_1d_on_the_cpu_reduction(float* data, float *output, int n) {
	int i, j;
	float sum = 0.0f;

	float* data_b = (float*)malloc(sizeof(float)*n);
	memcpy(data_b, data, sizeof(float)*n);

	for (i = n / 2; i > 0; i >>= 1) {
		for (j = 0; j < i; j++) {
			data_b[j] += data_b[j + i];
		}
	}

	sum = data_b[0];
	free(data_b);
	
	output[0] = sum;
}

void reduction_1d_on_the_cpu_KahanSum(float *data, float *output, int n) {
	int i;
	float sum = 0.0f, c = 0.0f, t, y;

	for (i = 0; i < n; i++) {
		y = data[i] - c;
		t = sum + y;
		c = (t - sum) - y;
		sum = t;
	}
	
	output[0] = sum;
}


void reduition_2d_on_the_cpu(float **data, float *output, int row, int col) {
	int i, j;
	float sum = 0.0f;

	for (i = 0; i < row; i++)
		for (j = 0; j < col; j++)
			sum += data[i][j];

	output[0] = sum;
}

void reduction_2d_on_the_cpu_reduction(float **data, float *output, int row, int col) {
	int i, j;
	float sum = 0.0f;

	float* data_b = (float*)malloc(sizeof(float)*row*col);
	for (i = 0; i < row; i++) {
		memcpy((data_b + col * i), data[i], sizeof(float)*col);
	}

	for (i = (row * col) / 2; i > 0; i >>= 1) {
		for (j = 0; j < i; j++) {
			data_b[j] += data_b[j + i];
		}
	}

	sum = data_b[0];
	free(data_b);

	output[0] = sum;
}

void reduction_2d_on_the_cpu_KahanSum(float **data, float *output, int row, int col) {
	int i, j;
	float sum = 0.0f, c = 0.0f, t, y;

	for (i = 0; i < row; i++) {
		for (j = 0; j < col; j++) {
			y = data[i][j] - c;
			t = sum + y;
			c = (t - sum) - y;
			sum = t;
		}
	}

	output[0] = sum;
}