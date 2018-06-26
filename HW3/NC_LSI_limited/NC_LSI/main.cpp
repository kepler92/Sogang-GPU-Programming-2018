#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define min(X,Y) ((X) < (Y) ? (X) : (Y))  
#define max(X,Y) ((X) > (Y) ? (X) : (Y)) 

extern "C" {
    void dsvdc_(double* x, int* ldx, int* n, int* p, double* s, double* e, double* u, int* ldu, double* v, int* ldv, double* work, int* job, int* info);
}


#define WORD_SIZE	50
#define DOCU_SIZE	200
#define BUFFER_SIZE	max(WORD_SIZE, DOCU_SIZE)
#define LSI_TOP		5


#include <Windows.h>
#include <winnt.h>
__int64 _start, _freq, _end;
#define CHECK_TIME_START QueryPerformanceFrequency((LARGE_INTEGER*)&_freq); QueryPerformanceCounter((LARGE_INTEGER*)&_start)
#define CHECK_TIME_END(a) QueryPerformanceCounter((LARGE_INTEGER*)&_end); a = (float)((float)(_end - _start) / (_freq * 1.0e-3f))


void file_lines(const char *filename, int *length) {
	FILE *fp = fopen(filename, "r");
	char buffer[BUFFER_SIZE];
	*length = 0;
	while (!feof(fp)) {
		memset(buffer, 0, BUFFER_SIZE * sizeof(char));
		fgets(buffer, BUFFER_SIZE, fp);
		if (strlen(buffer) != 0)
			(*length)++;
		else
			break;
	}
	fclose(fp);
}


void read_wordlist(char*** wordlist, const char* filename, int length) {
	FILE* fp = fopen(filename, "r");
	(*wordlist) = (char**)malloc(sizeof(char*) * length);
	for (int i = 0; i < length; i++) {
		(*wordlist)[i] = (char*)malloc(sizeof(char) * WORD_SIZE);
		fgets((*wordlist)[i], WORD_SIZE, fp);
		(*wordlist)[i][strlen((*wordlist)[i]) - 1] = '\0';
	}
	fclose(fp);
}


int find_wordlist(char *word, char **wordlist, int length) {
	for (int i = 0; i < length; i++) {
		if (strstr(word, wordlist[i]) != NULL)
			return i;
	}
	return -1;
}


void read_document(char*** document, const char* filename, int length) {
	FILE* fp = fopen(filename, "r");
	char buffer[DOCU_SIZE], *token;
	(*document) = (char**)malloc(sizeof(char*) * length);
	for (int i = 0; i < length; i++) {
		(*document)[i] = (char*)malloc(sizeof(char) * DOCU_SIZE);
		memset(buffer, 0, DOCU_SIZE * sizeof(char)); token = NULL;
		fgets(buffer, DOCU_SIZE, fp);
		token = strtok(buffer, ":");
		token = token + strlen(buffer) + 1;
		if (token[0] == ' ')
			token = token + 1;
		token[strlen(token) - 1] = '\0';
		memcpy((*document)[i], token, DOCU_SIZE);
	}
	fclose(fp);
}


int read_query(double *Q, char **wordlist, int wordlist_length) {
	char query[BUFFER_SIZE];
	memset(query, 0, sizeof(char)*BUFFER_SIZE);
	fgets(query, BUFFER_SIZE, stdin);

	if (query[0] == '\0')
		return -1;

	query[strlen(query) - 1] = '\0';
	memset(Q, 0, sizeof(double) * wordlist_length);
	char* token = strtok(query, " \t");
	do {
		int i = find_wordlist(token, wordlist, wordlist_length);
		if (i >= 0)
			Q[i] = 1;
	} while (token = strtok(NULL, " \t"));
	return 1;
}


void make_A(double *A, char** document, int document_length, char ** wordlist, int wordlist_length) {
	for (int d = 0; d < document_length; d++) {
		int index = d * wordlist_length;
		for (int w = 0; w < wordlist_length; w++) {	// C/C++: A[word][docu] = 1, F: A[docu][word] = 1
			if (strstr(document[d], wordlist[w]) != NULL) {
				A[index + w] = 1;
			}
			else {
				A[index + w] = 0;
			}
		}
	}
}


void load_matrix(const char* filename, double *a, int row, int column) {
	fprintf(stdout, "\t- Load %s\n", filename);
	FILE *fp = fopen(filename, "r");
	for (int i = 0; i < column; i++) {
		for (int j = 0; j < row; j++) {
			fscanf(fp, "%lf", &a[i * row + j]);
		}
	}
	fclose(fp);
}


void save_matrix(const char* filename, double *a, int row, int column) {
	fprintf(stdout, "\t- Save %s\n", filename);
	FILE *fp = fopen(filename, "w");
	for (int i = 0; i < column; i++) {
		for (int j = 0; j < row; j++) {
			fprintf(fp, "%lf\t", a[i * row + j]);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n");
	fclose(fp);
}


void print_matrix(const char* filename, double *a, int row, int column) {
	fprintf(stdout, "Print %s\n", filename);
	FILE *fp = fopen(filename, "w");
	for (int j = 0; j < row; j++) {
		for (int i = 0; i < column; i++) {
			fprintf(fp, "%lf\t", a[j * column + i]);
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n");
	fclose(fp);
	fprintf(stdout, "> Done\n");
}


void transpose_matrix(double *matrix, int row, int col)
{
	for (int i = 0; i < row; i++)
	{
		for (int j = i + 1; j < col; j++)
		{
			double temp = matrix[i * col + j];
			matrix[i * col + j] = matrix[j * col + i];
			matrix[j * col + i] = temp;
		}
	}
}


void norm_matrix(double *A, int row, int column) {
	for (int i = 0; i < column; i++) {
		double sum = 0;
		for (int j = 0; j < row; j++)
			sum += A[i * row + j];
		
		sum = sqrt(sum);
		for (int j = 0; j < row; j++)
			A[i * row + j] /= sum;
	}
}


void document_collection_matrix(double *V, double *D, double *C, int row, int column) {
	for (int r = 0; r < row; r++) {
		int index = r * column;
		for (int c = 0; c < column; c++) {
			C[index + c] = V[index + c] * D[c];
		}
	}
}


void matrix_multiplication(double *A, double *B, double *C, int row, int K_value, int column) {
	memset(C, 0, sizeof(double) * row * column);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < column; j++) {
			C[i * column + j] = 0;
			for (int k = 0; k < K_value; k++) {
				C[i * column + j] += (A[i * K_value + k] * B[k * column + j]);
			}
		}
	}
}


void maximum_vector(double *v, int length, int *p, int top) {
	double* numbers = (double*)malloc(sizeof(double) * length);
	for (int i = 0; i < length; i++)
		numbers[i] = v[i];

	int *indexes = (int*)malloc(sizeof(int) * length);
	for (int i = 0; i < length; i++)
		indexes[i] = i;
	
	int i, j;
	int max;
	double temp;

	for (i = 0; i < top; i++)
	{
		max = i;
		for (j = i + 1; j < length; j++)
		{
			if (numbers[j] > numbers[max]) {
				max = j;
			}
		}

		temp = numbers[i];
		numbers[i] = numbers[max];
		numbers[max] = temp;

		temp = indexes[i];
		indexes[i] = indexes[max];
		indexes[max] = temp;
	}

	for (i = 0; i < top; i++) {
		p[i] = indexes[i];
	}
}


void reduce_dimension_column_matrix(double **a, int row, int column, int K) {
	double *dst = (double*)malloc(row * K * sizeof(double));
	for (int r = 0; r < row; r++) {
		for (int c = 0; c < K; c++) {
			dst[r * K + c] = (*a)[r * column + c];
		}
	}
	free(*a);
	(*a) = dst;
}


void reduce_dimension_row_matrix(double **a, int row, int column, int K) {
	double *dst = (double*)malloc(K * column * sizeof(double));
	for (int r = 0; r < K; r++) {
		for (int c = 0; c < column; c++) {
			dst[r * column + c] = (*a)[r * column + c];
		}
	}
	free(*a);
	(*a) = dst;
}


void hello_print(const char* wordlist, const char* documentkey) {
	printf("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n");
	printf("┃ CSE5437 수치 컴퓨팅 및 GPU 프로그래밍 ┃\n");
	printf("┃                 숙제 3                ┃\n");
	printf("┃   LSI(Latent Semantic Indexing)       ┃\n");
	printf("┃                                       ┃\n");
	printf("┃ wordlist: %-27s ┃\n", wordlist);
	printf("┃ document: %-27s ┃\n", documentkey);
	printf("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n");
	printf("\n\n");
}


int main() {
    int K;
    int row;        //# of keyword
    int column;     //# of docu
	
	const char wordlist_file[] = "documents/wordlist.txt";
	const char documentkey_file[] = "documents/documentkey.txt";

	hello_print(wordlist_file, documentkey_file);

	char** wordlist = NULL;
	char** document = NULL;

	file_lines(wordlist_file, &row);
	file_lines(documentkey_file, &column);

	read_wordlist(&wordlist, wordlist_file, row);
	read_document(&document, documentkey_file, column);

    double* A;
    double* U;
    double* D;
    double* V;

    //A = new double[row*column];
	A = (double*)malloc(row * column * sizeof(double));
	
    int LDX = row;
    int N = row;
    int P = column;
    int LDU = row;
    int LDV = column;
    U = (double*)malloc(row * row * sizeof(double));
    D = (double*)malloc(min(row + 1, column) * sizeof(double));
    V = (double*)malloc(column * column * sizeof(double));
    double* E = (double*)malloc(row * sizeof(double));
    double* WORK = (double*)malloc(row * sizeof(double));
    int JOB = 11;
    int INFO;

	printf("SVD 불러오기 [y/n]: ");
	switch (fgetc(stdin)) {
	case 'y':
	case 'Y':
		load_matrix("U.txt", U, row, row);
		load_matrix("D.txt", D, min(row + 1, column), 1);
		load_matrix("V.txt", V, column, column);
		break;
	case 'N':
	case 'n':
		make_A(A, document, column, wordlist, row);
		//print_matrix("A.txt", A, row, column);

		norm_matrix(A, row, column);
		//save_matrix("An.txt", A, row, column);

		printf(" > SVD 진행 중\n");
		dsvdc_(A, &LDX, &N, &P, D, E, U, &LDU, V, &LDV, WORK, &JOB, &INFO);
		
		save_matrix("U.txt", U, row, row);
		save_matrix("D.txt", D, min(row + 1, column), 1);
		save_matrix("V.txt", V, column, column);
		break;
	default:
		exit(-1);
	}

	printf("\nK값 입력: ");
	scanf("%d", &K);
	getchar();
		
	//transpose_matrix(U, row, row);
	reduce_dimension_row_matrix(&U, row, row, K);
	/*print_matrix("Uu.txt", U, K, row);*/

	transpose_matrix(V, column, column);
	reduce_dimension_column_matrix(&V, column, column, K);
	//print_matrix("Vv.txt", V, column, K);

	double *C = (double*)malloc(column * K * sizeof(double));
	document_collection_matrix(V, D, C, column, K);
	//print_matrix("C.txt", C, column, K);
	printf(" > 설정 완료\n");

	double* Q = (double*)malloc(sizeof(double) * row);
	double* uq = (double*)malloc(sizeof(double) * K * 1);
	double* result = (double*)malloc(sizeof(double) * row);
	int* sort = (int*)malloc(sizeof(int) * LSI_TOP);
	float compute_time;

	while (1) {
		printf("\n=========================\n\n");
		printf("검색어: ");
		if (read_query(Q, wordlist, row) == -1)
			break;
		
		CHECK_TIME_START;

		matrix_multiplication(U, Q, uq, K, row, 1);
		//print_matrix("UQ.txt", uq, K, 1);
		
		matrix_multiplication(C, uq, result, column, K, 1);
		//print_matrix("R.txt", result, column, 1);

		maximum_vector(result, column, sort, LSI_TOP);

		CHECK_TIME_END(compute_time);
		printf(" > compute_time: %.3fms\n", compute_time);
		
		for (int i = 0; i < LSI_TOP; i++) {
			int index = sort[i];
			printf(" - %04d(%lf): %s\n", index, result[index], document[index]);
		}
	}

	free(A);
	free(U);
	free(D);
	free(V);

	free(C);
	free(Q);
	free(uq);
	free(result);
	free(sort);
	
	return 0;
}
