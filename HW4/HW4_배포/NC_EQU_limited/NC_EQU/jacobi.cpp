#include "jacobi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


void get_sigma(int N, double *B, double *X, int NELT, int *IA, int *JA, double *A, double *sigma, double *a) {
	for (int n = 0; n < NELT; n++) {
		int i = IA[n] - 1;
		int j = JA[n] - 1;

		if (i != j)
			sigma[i] += A[n] * X[j];

		else
			a[i] = A[n];
	}
}

void get_value(int N, double *B, double *X, double *sigma, double *a) {
	for (int i = 0; i < N; i++)
		X[i] = (B[i] - sigma[i]) / a[i];
}



void jacobi_method(int N, double *B, double *X, int NELT, int *IA, int *JA, double *A, int iter) {
	double *sigma = (double*)malloc(sizeof(double) * N);
	double *a = (double*)malloc(sizeof(double) * N);

	for (int i = 0; i < iter; i++) {
		memset(sigma, 0, sizeof(double) * N);
		memset(a, 0, sizeof(double) * N);

		get_sigma(N, B, X, NELT, IA, JA, A, sigma, a);
		get_value(N, B, X, sigma, a);
	}

	free(sigma);
	free(a);
}



