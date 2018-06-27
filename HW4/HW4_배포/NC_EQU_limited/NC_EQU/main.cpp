#include <stdlib.h>

#define Iteration_Number 100000

extern "C" {
    int dsgs_(int *n, double *b, double *x, int *nelt, int *ia, int *ja, double *a, int *isym, int *itol, double *tol, int *itmax,
        int *iter, double *err, int *ierr, int *iunit, double *rwork, int *lenw, int *iwork, int *leniw);
}


#include<stdio.h>
#include<string.h>
#include<math.h>
#include "jacobi.h"

#define BUFFER_SIZE	1000
#define FILES		4

#define INIT_X_VALUE	1.0f
#define PRINT_X_LENGTH	10


char datafiles[FILES][BUFFER_SIZE] = {
	"data/mat0000.txt",
	"data/mat0013.txt",
	"data/mat0020.txt",
	"data/mat0029.txt"
};


#include <Windows.h>
#include <winnt.h>
__int64 _start, _freq, _end;
#define CHECK_TIME_START QueryPerformanceFrequency((LARGE_INTEGER*)&_freq); QueryPerformanceCounter((LARGE_INTEGER*)&_start)
#define CHECK_TIME_END(a) QueryPerformanceCounter((LARGE_INTEGER*)&_end); a = (float)((float)(_end - _start) / (_freq * 1.0e-3f))


void choose_data(char* filename) {	
	printf("File List\n");
	for (int i = 0; i < FILES; i++) {
		printf(" %d: %s\n", i + 1, datafiles[i]);
	}

	int choose;
	printf("Choose file number: ");
	scanf("%d", &choose);

	if (1 <= choose && choose <= FILES)
		strcpy(filename, datafiles[choose - 1]);

	else {
		if (choose == 0)
			strcpy(filename, "data/test.txt");

		else
			exit(-1);
	}

	//switch (choose)
	//{
	//case 1:
	//	strcpy(filename, "data/mat0000.txt");
	//	break;
	//case 2:
	//	strcpy(filename, "data/mat0013.txt");
	//	break;
	//case 3:
	//	strcpy(filename, "data/mat0020.txt");
	//	break;
	//case 4:
	//	strcpy(filename, "data/mat0029.txt");
	//	break;
	//default:
	//	exit(-1);
	//}
}


void read_lines_data(char* filename, int *N, int *NELT) {
	*N = *NELT = 0;

	FILE *fp = fopen(filename, "r");
	fscanf(fp, "%d\n", N);

	int nelt_line = 0;

	char buffer[BUFFER_SIZE];
	while (!feof(fp)) {
		fgets(buffer, BUFFER_SIZE, fp);
		if (strstr(buffer, "*") != NULL)
			break;
		nelt_line++;
	}

	fclose(fp);

	fp = fopen(filename, "r");
	fgets(buffer, BUFFER_SIZE, fp);
	for (int i = 0; i < nelt_line; i++) {
		double value;
		int row, column;
		fscanf(fp, "%lf %d %d\n", &value, &row, &column);

		if (row == column)
			(*NELT) += 1;
		else
			(*NELT) += 2;
	}
	fclose(fp);
}


void read_matrix_data(char* filename, int N, int NELT,
						double *A, int *IA, int* JA,
						double *B, double *X) {
	FILE *fp = fopen(filename, "r");
	char buffer[BUFFER_SIZE];
	fgets(buffer, BUFFER_SIZE, fp);
	
	int i = 0, j = 0;
	while (i < NELT) {
		double value;
		int row, column;
		fscanf(fp, "%lf %d %d\n", &value, &row, &column);

		A[i] = value; IA[i] = row; JA[i] = column;
		i++;
		
		if (row != column) {
			A[i] = value; IA[i] = column; JA[i] = row;
			i++;
		}
	}

	fgets(buffer, BUFFER_SIZE, fp);

	for (int i = 0; i < N; i++)
		fscanf(fp, "%lf\n", &B[i]);

	fgets(buffer, BUFFER_SIZE, fp);

	for (int i = 0; i < N; i++)
		fscanf(fp, "%lf\n", &X[i]);

	fclose(fp);
}


void print_X(const char* sign, double *X, int length) {
	printf("旨收收收收收收收收收收收收收收收收收收收收收收旬\n");
	printf("早 %-20s 早\n", sign);
	printf("早式式式式式式式式式式式式式式式式式式式式式式早\n");
	for (int i = 0; i < length; i++)
		printf("早 %5d: %-13lf 早\n", i + 1, X[i]);
	printf("曲收收收收收收收收收收收收收收收收收收收收收收旭\n");
}


void print_INFO(const char* sign, int iter, double e, double r, float compute_time) {
	printf("旨收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收旬\n");
	printf("早 %-30s 早\n", sign);
	printf("早式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式式早\n");
	printf("早 Total Iteration: %13d 早\n", iter);
	printf("早 Difference e : %15lf 早\n", e);
	printf("早 Difference r : %15lf 早\n", r);
	printf("早 Compute Time: %14.3fms 早\n", compute_time);
	printf("曲收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收收旭\n");
}


double difference_X(double *A, double *B, int length) {
	double* differ = (double*)malloc(sizeof(double) * length);
	for (int i = 0; i < length; i++)
		differ[i] = fabs(A[i] - B[i]);

	for (int i = length / 2; i > 0; i >>= 1) {
		for (int j = 0; j < i; j++) {
			differ[j] += differ[j + 1];
		}
	}

	double result = differ[0];
	free(differ);
	return result;
}


double difference_A(int NELT, int *IA, int *JA, double *A, int N, double *X, double *X_) {
	double* differ = (double*)malloc(sizeof(double) * N);
	memset(differ, 0, sizeof(double) * N);

	for (int n = 0; n < NELT; n++) {
		int i = IA[n] - 1;
		int j = JA[n] - 1;
		double value = A[n];

		differ[i] += (
			fabs(A[n] * X[j] - A[n] * X_[j])
			);
	}

	for (int i = N / 2; i > 0; i >>= 1) {
		for (int j = 0; j < i; j++) {
			differ[j] += differ[j + 1];
		}
	}

	double result = differ[0];
	free(differ);
	return result;
}




int main() {
    int N;
    double* B;
    double* X;
    int NELT;
    int* IA;
    int* JA;
    double* A;
    int ISYM;
    int ITOL;
    double TOL;
    int ITMAX;
    int ITER;
    double ERR;
    int IERR;
    int IUNIT;
    double* RWORK;
    int LENW;
    int* IWORK;
    int LENIW;


	char filename[BUFFER_SIZE];
	choose_data(filename);
	read_lines_data(filename, &N, &NELT);

	int iter;
	printf("Total Interation: ");
	scanf("%d", &iter);	


    //N;
    B = (double *)malloc(N * sizeof(double));
    X = (double *)malloc(N * sizeof(double));
    //NELT;
    IA = (int *)malloc(NELT * sizeof(int));
    JA = (int *)malloc(NELT * sizeof(int));
    A = (double *)malloc(NELT * sizeof(double));
    ISYM = 0;
    ITOL = 1;
    //TOL = 0.00000001;
	TOL = 0.000000000000000001;
    ITMAX = Iteration_Number;
    IUNIT = 0;
    RWORK = (double *)malloc((NELT + 3 * N) * sizeof(double));
    LENW = NELT + 3 * N + 1;
    IWORK = (int *)malloc((NELT + 2 * N + 11) * sizeof(int));
    LENIW = NELT + 2 * N + 11;

	read_matrix_data(filename, N, NELT, A, IA, JA, B, X);
	print_X("Solution", X, PRINT_X_LENGTH);
	
	float time_GS, time_Jacobi;
	double differ_X_GS, differ_A_GS;
	double differ_X_Jacobi, differ_A_Jacobi;
	int iter_GS, iter_Jacobi;

	//
	//
	// Gauss-Seidel Method
	//
	//

	int *IA_GSMethod = (int *)malloc(NELT * sizeof(int));
	int *JA_GSMethod = (int *)malloc(NELT * sizeof(int));
	double *A_GSMethod = (double *)malloc(NELT * sizeof(double));

	memcpy(IA_GSMethod, IA, NELT * sizeof(int));
	memcpy(JA_GSMethod, JA, NELT * sizeof(int));
	memcpy(A_GSMethod, A, NELT * sizeof(double));

	double *X_GSMethod = (double *)malloc(N * sizeof(double));
	for (int i = 0; i < N; i++) X_GSMethod[i] = INIT_X_VALUE;
	//memset(X_GSMethod, 0, N * sizeof(double));

	ITMAX = iter;
		
	CHECK_TIME_START;
    dsgs_(&N, B, X_GSMethod, &NELT, IA_GSMethod, JA_GSMethod, A_GSMethod, &ISYM, &ITOL, &TOL, &ITMAX, &ITER, &ERR, &IERR, &IUNIT, RWORK, &LENW, IWORK, &LENIW);
	CHECK_TIME_END(time_GS);

	free(IA_GSMethod);
	free(JA_GSMethod);
	free(A_GSMethod);
	
	differ_X_GS = difference_X(X, X_GSMethod, N); 
	differ_A_GS = difference_A(NELT, IA, JA, A, N, X, X_GSMethod);
	iter_GS = ITER - 1;
	print_X("Gauss-Seidel Method", X_GSMethod, PRINT_X_LENGTH);	
	free(X_GSMethod);

	//
	//
	// Jacobi Method
	//
	//

	double *X_JacobiMethod = (double *)malloc(N * sizeof(double));
	for (int i = 0; i < N; i++) X_JacobiMethod[i] = INIT_X_VALUE;

	CHECK_TIME_START;
	jacobi_method(N, B, X_JacobiMethod, NELT, IA, JA, A, iter);
	CHECK_TIME_END(time_Jacobi);

	differ_X_Jacobi = difference_X(X, X_JacobiMethod, N); 
	differ_A_Jacobi = difference_A(NELT, IA, JA, A, N, X, X_JacobiMethod);
	iter_Jacobi = iter;
	print_X("Jacobi Method", X_JacobiMethod, PRINT_X_LENGTH);
	free(X_JacobiMethod);

	print_INFO("Gauss-Seidel Method", iter_GS, differ_X_GS, differ_A_GS, time_GS);
	print_INFO("Jacobi Method", iter_Jacobi, differ_X_Jacobi, differ_A_Jacobi, time_Jacobi);


    free(B);
    free(X);
    free(IA);
    free(JA);
    free(A);
    free(RWORK);
    free(IWORK);

    return 0;
}