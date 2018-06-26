#include <stdlib.h>

#include<stdio.h>
#include<string.h>
#define BUFFER_SIZE	1000

#define Iteration_Number 100000

extern "C" {
    int dsgs_(int *n, double *b, double *x, int *nelt, int *ia, int *ja, double *a, int *isym, int *itol, double *tol, int *itmax,
        int *iter, double *err, int *ierr, int *iunit, double *rwork, int *lenw, int *iwork, int *leniw);
}


void choose_data(char* filename) {
	int choose;
	printf("file choose: ");
	scanf("%d", &choose);

	switch (choose)
	{
	case 1:
		strcpy(filename, "data/mat0000.txt");
		break;
	case 2:
		strcpy(filename, "data/mat0013.txt");
		break;
	case 3:
		strcpy(filename, "data/mat0020.txt");
		break;
	case 4:
		strcpy(filename, "data/mat0029.txt");
		break;
	default:
		exit(-1);
	}
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
	
	for (int i = 0; i < NELT; ) {
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
	double *InitX;

	char filename[BUFFER_SIZE];
	choose_data(filename);

	read_lines_data(filename, &N, &NELT);

    //N;
    B = (double *)malloc(N * sizeof(double));
    X = (double *)malloc(N * sizeof(double));
    //NELT;
    IA = (int *)malloc(NELT * sizeof(int));
    JA = (int *)malloc(NELT * sizeof(int));
    A = (double *)malloc(NELT * sizeof(double));
    ISYM = 0;
    ITOL = 1;
    TOL = 0.00000001;
    ITMAX = Iteration_Number;
    IUNIT = 0;
    RWORK = (double *)malloc((NELT + 3 * N) * sizeof(double));
    LENW = NELT + 3 * N + 1;
    IWORK = (int *)malloc((NELT + 2 * N + 11) * sizeof(int));
    LENIW = NELT + 2 * N + 11;
	
	InitX = (double *)malloc(N * sizeof(double));
	memset(InitX, 0, N * sizeof(double));
	read_matrix_data(filename, N, NELT, A, IA, JA, B, X);

	for (int i = 0; i < 10; i++)
		printf("%lf ", X[i]);
	printf("%\n");

    dsgs_(&N, B, InitX, &NELT, IA, JA, A, &ISYM, &ITOL, &TOL, &ITMAX, &ITER, &ERR, &IERR, &IUNIT, RWORK, &LENW, IWORK, &LENIW);

	for (int i = 0; i < 10; i++)
		printf("%lf ", InitX[i]);
	printf("%\n");





    free(B);
    free(X);
    free(IA);
    free(JA);
    free(A);
    free(RWORK);
    free(IWORK);

    return 0;
}