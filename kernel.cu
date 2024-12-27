
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>

#include <iomanip>
#include <chrono>
#include <iostream>

#include <stdio.h>
#include <stdint.h>
#include <cstdlib>
#include <string.h>
#include <time.h>

using namespace cooperative_groups;
using namespace std;
using namespace std::chrono;

//#define S2 "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABA"

//#define S1 "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"

#define LEN 10238

#define ALPHLEN 26

#define ALPHSTART 'A'

#define ALPHEND 'Z'

#define WARPSIZE 32

#define WARPSINBLOCK 32

// Struktura listy jednokierunkowej, dla opisywania operacji zmiany s1 na s2
// Typ operacji
typedef enum {
	ADD,
	DEL,
	REPLACE
} OperationType;

// Węzeł listy
typedef struct Node
{
	uint16_t ind;
	char letter;
	OperationType type;
	struct Node* next;
} Node;

// Struktura dla argumentów kernela dla Cooperative Group
typedef struct {
	uint16_t* D;
	uint16_t* X;
	char* s1;
	char* s2;
	uint16_t s1Len;
	uint16_t s2Len;
} KernelArgs;

cudaError_t LevenshteinGPU(char* s1, char* s2, const uint16_t s1Len, const uint16_t s2Len, uint16_t* D, long long* gpu_alloc_time, long long* calculateD_time, long long* copy_to_h_time, long long* calculateX_time);

__host__ uint16_t checkWord(char* s);

__host__ __device__ uint32_t GetDInd(uint16_t i, uint16_t j, uint16_t width);

__host__ __device__ uint16_t Min(uint16_t num1, uint16_t num2, uint16_t num3);

__host__ void CPULevenshtein(char* s1, uint16_t s1Len, char* s2, uint16_t s2Len, uint16_t* D);

__host__ void PrintD(uint16_t* D, uint16_t height, uint16_t width, char* s1, char* s2);

__host__ void PrintX(uint16_t* X, uint16_t height, uint16_t width, char* s2);

__host__ Node* RetrievePath(uint16_t* D, uint16_t height, uint16_t width, char* s1, char* s2);

__host__ bool EasyCheck(uint16_t* hD, uint16_t* dD, uint16_t height, uint16_t width);

__global__ void calculateX(uint16_t* X, char* s2, const uint16_t s2Len);

__global__ void calculateD(uint16_t* D, uint16_t* X, char* s1, char* s2, const uint16_t s1Len, const uint16_t s2Len, uint16_t* globalDiagArray);

void saveToFile(uint16_t* dD, uint16_t* hD, char* s1, char* s2, const uint16_t s1Len, const uint16_t s2Len);

// Operacje na liście jednokierunkowej, dla opisywania operacji zmiany s1 na s2
Node* createNode(uint16_t ind, char letter, OperationType type);
void addToFrontList(Node** head, uint16_t ind, char letter, OperationType type);
void addToEndList(Node** tail, Node** head, uint16_t ind, char letter, OperationType type);
void printList(Node* head);
void freeList(Node* head);

int main(int argc, char* argv[])
{
	cudaError_t cudaStatus;
	long long cpu_time = 0, gpu_time = 0, gpu_calculateD_time = 0, gpu_prepare_time = 0, gpu_copy_to_h_time = 0, gpu_calculateX_time = 0, cpu_path_time = 0, gpu_path_time = 0;

	char S1[LEN], S2[LEN], help;

	srand(time(NULL));

	for (int i = 0; i < LEN - 1; i++)
	{
		S1[i] = 'A' + rand() % 26;
		S2[i] = 'A' + rand() % 26;
	}
	S1[LEN - 1] = '\0';
	S2[LEN - 1] = '\0';

	char* s1, * s2;

	if (argc > 2)
		s2 = argv[2];
	else
		s2 = S2;
	if (argc > 1)
		s1 = argv[1];
	else
	{
		s1 = S1; s2 = S2;
	}

	uint16_t s1Len = checkWord(s1), s2Len = checkWord(s2);
	if (s1Len == 0 || s2Len == 0)
	{
		printf("Podane slowa sa niepoprawne!\n");
		return 0;
	}

	uint16_t* D_CPU = (uint16_t*)malloc(sizeof(uint16_t) * (s1Len + 1) * (s2Len + 1));

	auto cpu_ts = high_resolution_clock::now();
	CPULevenshtein(s1, s1Len, s2, s2Len, D_CPU);
	auto cpu_te = high_resolution_clock::now();
	cpu_time += 0.001 * duration_cast<microseconds> (cpu_te - cpu_ts).count();

	auto cpu_path_ts = high_resolution_clock::now();
	Node* CPU_result = RetrievePath(D_CPU, s1Len + 1, s2Len + 1, s1, s2);
	auto cpu_path_te = high_resolution_clock::now();
	cpu_path_time += duration_cast<microseconds> (cpu_path_te - cpu_path_ts).count();

	uint16_t* D_GPU = (uint16_t*)malloc(sizeof(uint16_t) * (s1Len + 1) * (s2Len + 1));

	auto gpu_ts = high_resolution_clock::now();
	cudaStatus = LevenshteinGPU(s1, s2, s1Len, s2Len, D_GPU, &gpu_prepare_time, &gpu_calculateD_time, &gpu_copy_to_h_time, &gpu_calculateX_time);
	auto gpu_te = high_resolution_clock::now();
	gpu_time += 0.001 * duration_cast<microseconds> (gpu_te - gpu_ts).count();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "LevenshteinGPU failed! %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}

	if (EasyCheck(D_CPU, D_GPU, s1Len + 1, s2Len + 1))
		printf("Macierze D sa takie same :)\n");
	else
		printf("Macierze D sa inne!!\n");

	auto gpu_path_ts = high_resolution_clock::now();
	Node* GPU_result = RetrievePath(D_GPU, s1Len + 1, s2Len + 1, s1, s2);
	auto gpu_path_te = high_resolution_clock::now();
	gpu_path_time += duration_cast<microseconds> (gpu_path_te - gpu_path_ts).count();

	//PrintD(D_CPU, s1Len + 1, s2Len + 1, s1, s2);
	//printList(CPU_result);
	//PrintD(D_GPU, s1Len + 1, s2Len + 1, s1, s2);
	//printList(GPU_result);
	//saveToFile(D_GPU, D_CPU, s1, s2, s1Len, s2Len);

	std::cout << "CPU time: " << setw(7) << cpu_time << " nsec" << endl;
	std::cout << "Whole GPU time: " << setw(7) << gpu_time << " nsec" << endl;
	std::cout << "GPU memory alloc + copy + SetDevice time: " << setw(7) << gpu_prepare_time << " nsec" << endl;
	std::cout << "CalculateX time: " << setw(7) << gpu_calculateX_time << " nsec" << endl;
	std::cout << "CalculateD time: " << setw(7) << gpu_calculateD_time << " nsec" << endl;
	std::cout << "Algorithm(CalculateD + CalculateX) GPU time: " << setw(7) << gpu_calculateD_time + gpu_calculateX_time << " nsec" << endl;
	std::cout << "Copy to host GPU time: " << setw(7) << 0.001 * gpu_copy_to_h_time << " nsec" << endl;
	std::cout << "RetrievePath CPU time: " << setw(7) << 0.001 * cpu_path_time << " nsec" << endl;
	std::cout << "RetrievePath GPU time: " << setw(7) << 0.001 * gpu_path_time << " nsec" << endl;

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	freeList(CPU_result);
	freeList(GPU_result);
	free(D_CPU);
	free(D_GPU);
	return 0;
}

__host__ uint16_t checkWord(char* s)
{
	int i = 0;
	while (s[i] != '\0')
	{
		if (s[i] < ALPHSTART || s[i] > ALPHEND)
			return 0;
		i++;
	}
	return i;
}

__host__ __device__ uint32_t GetDInd(uint16_t i, uint16_t j, uint16_t width)
{
	return i * width + j;
}

__host__ __device__ uint16_t Min(uint16_t num1, uint16_t num2, uint16_t num3)
{
	if (num1 <= num2 && num1 <= num3)
		return num1;
	else if (num2 <= num1 && num2 <= num3)
		return num2;
	else return num3;
}

__host__ void CPULevenshtein(char* s1, uint16_t s1Len, char* s2, uint16_t s2Len, uint16_t* D)
{
	for (uint16_t i = 0; i < s1Len + 1; i++)
	{
		for (uint16_t j = 0; j < s2Len + 1; j++)
		{
			if (i == 0)
				D[GetDInd(i, j, s2Len + 1)] = j;
			else if (j == 0)
				D[GetDInd(i, j, s2Len + 1)] = i;
			else if (s1[i - 1] == s2[j - 1])
				D[GetDInd(i, j, s2Len + 1)] = D[GetDInd(i - 1, j - 1, s2Len + 1)];
			else
				D[GetDInd(i, j, s2Len + 1)] = 1 + Min(D[GetDInd(i - 1, j, s2Len + 1)], D[GetDInd(i, j - 1, s2Len + 1)], D[GetDInd(i - 1, j - 1, s2Len + 1)]);
		}
	}
}

__host__ void PrintD(uint16_t* D, uint16_t height, uint16_t width, char* s1, char* s2)
{
	printf("   |");
	for (int i = 0; i < width; i++)
	{
		if (i == 0)
			printf("   |");
		else
			printf("%2c |", s2[i - 1]);
	}
	printf("\n");

	for (int i = 0; i < width + 1; i++)
	{
		printf("---+");
	}
	printf("\n");

	for (int i = 0; i < height; i++)
	{
		if (i == 0)
			printf("   |");
		else
			printf("%2c |", s1[i - 1]);

		for (int j = 0; j < width; j++)
		{
			printf("%2d |", D[GetDInd(i, j, width)]);
		}
		printf("\n");

		for (int i = 0; i < width + 1; i++)
		{
			printf("---+");
		}
		printf("\n");
	}
}

__host__ void PrintX(uint16_t* X, uint16_t height, uint16_t width, char* s2)
{
	printf("  ");
	for (int i = 0; i < width; i++)
	{
		if (i == 0)
			printf("  ");
		else
			printf("%2c", s2[i - 1]);
	}
	printf("\n");

	for (int i = 0; i < height; i++)
	{
		printf("%2c", ALPHSTART + i);

		for (int j = 0; j < width; j++)
		{
			printf("%2d", X[i + ALPHLEN * j]);
		}
		printf("\n");
	}
}

__host__ Node* RetrievePath(uint16_t* D, uint16_t height, uint16_t width, char* s1, char* s2)
{
	int i = height - 1, j = width - 1, added = 0, maxToAdd;
	Node* listHead = NULL, *listTail = NULL;
	maxToAdd = D[GetDInd(i, j, width)];
	int curCellVal = D[GetDInd(i, j, width)];

	while ((i != 0 || j != 0) && added != maxToAdd)
	{
		if (i == 0)
		{
			while (j != 0 && added != maxToAdd)
			{
				addToEndList(&listTail, &listHead, i, s2[j - 1], ADD);
				added++;
				j--;
			}
		}
		else if (j == 0)
		{
			while (i != 0 && added != maxToAdd)
			{
				addToEndList(&listTail, &listHead, i - 1, s1[i - 1], DEL);
				added++;
				i--;
			}
		}
		else
		{
			int min = Min(D[GetDInd(i - 1, j, width)], D[GetDInd(i, j - 1, width)], D[GetDInd(i - 1, j - 1, width)]);
			if (min == D[GetDInd(i - 1, j - 1, width)])
			{
				if (min != curCellVal)
				{
					addToEndList(&listTail, &listHead, i - 1, s2[j - 1], REPLACE);
					added++;
				}
				i--;
				j--;
			}
			else if(min == D[GetDInd(i, j - 1, width)])
			{
				addToEndList(&listTail, &listHead, i, s2[j - 1], ADD);
				added++;
				j--;
			}
			else
			{
				addToEndList(&listTail, &listHead, i - 1, s1[i - 1], DEL);
				added++;
				i--;
			}
			curCellVal = min;
		}
	}

	return listHead;
}

__host__ bool EasyCheck(uint16_t* hD, uint16_t* dD, uint16_t height, uint16_t width)
{
	int len = height * width;
	for (int i = 0; i < len; i++)
	{
		if (hD[i] != dD[i])
		{
			uint16_t hel1 = hD[i], hel2 = dD[i];
			return false;
		}
	}
	return true;
}

// Tworzenie nowego węzła listy
Node* createNode(uint16_t ind, char letter, OperationType type)
{
	Node* newNode = (Node*)malloc(sizeof(Node));
	if (!newNode)
	{
		printf("Error: alokacja pamięci przy tworzeniu node.\n");
		exit(EXIT_FAILURE);
	}
	newNode->ind = ind;
	newNode->letter = letter;
	newNode->type = type;
	newNode->next = NULL;
	return newNode;
}

// Dodawanie na początek listy
void addToFrontList(Node** head, uint16_t ind, char letter, OperationType type)
{
	Node* newNode = createNode(ind, letter, type);
	if (*head == NULL)
	{
		*head = newNode;
		return;
	}

	newNode->next = *head;
	*head = newNode;
}

// Dodawanie na koniec listy
void addToEndList(Node** tail, Node** head, uint16_t ind, char letter, OperationType type)
{
	Node* newNode = createNode(ind, letter, type);
	if (*tail == NULL)
	{
		*head = newNode;
		*tail = newNode;
		return;
	}

	(*tail)->next = newNode;
	*tail = (*tail)->next;
}

// Wypisywanie listy wynikowej
void printList(Node* head)
{
	Node* current = head;
	while (current != NULL)
	{
		printf("Index: %d, Letter: %c, Type: ", current->ind, current->letter);
		switch (current->type)
		{
		case ADD:
			printf("ADD\n");
			break;
		case DEL:
			printf("DELETE\n");
			break;
		case REPLACE:
			printf("REPLACE\n");
			break;
		}
		current = current->next;
	}
}

// Czyszczenie listy
void freeList(Node* head)
{
	Node* current = head;
	while (current != NULL)
	{
		Node* temp = current;
		current = current->next;
		free(temp);
	}
}

// Helper function for using CUDA
cudaError_t LevenshteinGPU(char* s1, char* s2, const uint16_t s1Len, const uint16_t s2Len, uint16_t* D, long long* gpu_prepare_time, long long* calculateD_time, long long* copy_to_h_time, long long* calculateX_time)
{
	uint16_t* d_X;
	uint16_t* d_D;
	char* d_s1, * d_s2;
	cudaError_t cudaStatus;
	uint16_t* d_globalDiagArray;

	auto gpu_preapare_ts = high_resolution_clock::now();
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	int dev = 0, blocks = 1, threads = 1;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);

	if ((s2Len + 1) % (WARPSINBLOCK * WARPSIZE) != 0)
		blocks = ((s2Len + 1) - ((s2Len + 1) % (WARPSINBLOCK * WARPSIZE))) / (WARPSINBLOCK * WARPSIZE) + 1;
	else
		blocks = (s2Len + 1) / (WARPSINBLOCK * WARPSIZE);

	if (blocks > deviceProp.multiProcessorCount)
	{
		printf("Za długie słowo s2 dla tego GPU.\n");
		goto Error;
	}

	if ((s2Len + 1) >= (WARPSINBLOCK * WARPSIZE))
		threads = (WARPSINBLOCK * WARPSIZE);
	else if ((s2Len + 1) % WARPSIZE == 0)
		threads = ((s2Len + 1) / WARPSIZE) * WARPSIZE;
	else
		threads = (((s2Len + 1) - (s2Len + 1) % WARPSIZE) / WARPSIZE + 1) * WARPSIZE;

	cudaStatus = cudaMalloc(&d_X, (s2Len + 1) * ALPHLEN * sizeof(uint16_t));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "d_X cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_D, (s1Len + 1) * (s2Len + 1) * sizeof(uint16_t));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "d_D cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_globalDiagArray, blocks * sizeof(uint16_t));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "d_globalDiagArray cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_s1, s1Len * sizeof(char));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "d_s1 cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_s2, s2Len * sizeof(char));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "d_s2 cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_s1, s1, s1Len * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "s1 cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(d_s2, s2, s2Len * sizeof(char), cudaMemcpyHostToDevice);
	auto gpu_preapare_te = high_resolution_clock::now();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "s2 cudaMemcpy failed!");
		goto Error;
	}

	*gpu_prepare_time = 0.001 * duration_cast<microseconds> (gpu_preapare_te - gpu_preapare_ts).count();

	auto gpu_calculateX_ts = high_resolution_clock::now();
	calculateX<<<1, 32>>>(d_X, d_s2, s2Len);

	cudaStatus = cudaDeviceSynchronize();
	auto gpu_calculateX_te = high_resolution_clock::now();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calculateX!\n", cudaStatus);
		goto Error;
	}

	*calculateX_time = 0.001 * duration_cast<microseconds> (gpu_calculateX_te - gpu_calculateX_ts).count();

	void* args[] = {(void*)&d_D, (void*)&d_X, (void*)&d_s1, (void*)&d_s2, (void*)&s1Len, (void*)&s2Len, (void*)&d_globalDiagArray};

	// initialize, then launch
	auto gpu_calculateD_ts = high_resolution_clock::now();
	cudaLaunchCooperativeKernel((void*)calculateD, blocks, threads, args);

	cudaStatus = cudaDeviceSynchronize();
	auto gpu_calculateD_te = high_resolution_clock::now();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	*calculateD_time = 0.001 * duration_cast<microseconds> (gpu_calculateD_te - gpu_calculateD_ts).count();

	auto gpu_memory_back_ts = high_resolution_clock::now();
	cudaStatus = cudaMemcpy(D, d_D, (s1Len + 1) * (s2Len + 1) * sizeof(uint16_t), cudaMemcpyDeviceToHost);
	auto gpu_memory_back_te = high_resolution_clock::now();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "D cudaMemcpy failed!");
		goto Error;
	}

	*copy_to_h_time += duration_cast<microseconds>(gpu_memory_back_te - gpu_memory_back_ts).count();

Error:
	cudaFree(d_D);
	cudaFree(d_X);
	cudaFree(d_s1);
	cudaFree(d_s2);

	return cudaStatus;
}

__global__ void calculateX(uint16_t* X, char* s2, const uint16_t s2Len)
{
	__shared__ uint16_t buffer[ALPHLEN];

	if (threadIdx.x < ALPHLEN)
	{
		uint16_t prev = 0;
		uint16_t num = 0;
		for (int i = 0; i < s2Len + 1; i++)
		{
			if (i == 0)
				num = 0;
			else if (s2[i - 1] == ALPHSTART + threadIdx.x)
				num = i;
			else
				num = prev;

			buffer[threadIdx.x] = num;
			prev = num;

			if (threadIdx.x == 0)
				memcpy(X + ALPHLEN * i, buffer, ALPHLEN * sizeof(uint16_t));
		}
	}
}

__global__ void calculateD(uint16_t* D, uint16_t* X, char* s1, char* s2, const uint16_t s1Len, const uint16_t s2Len, uint16_t* globalDiagArray)
{
	grid_group grid = this_grid();
	__shared__ uint16_t sharedDiagArray[WARPSINBLOCK - 1];

	if (threadIdx.x + blockIdx.x * blockDim.x < s2Len + 1)
	{
		uint16_t Xrow[ALPHLEN];
		char s1c, s2c;
		uint16_t foundVal = threadIdx.x + blockDim.x * blockIdx.x, prevVal = threadIdx.x + blockDim.x * blockIdx.x, diagVal, x;

		if (threadIdx.x + blockDim.x * blockIdx.x != 0)
			memcpy(&s2c, s2 + threadIdx.x + blockDim.x * blockIdx.x - 1, sizeof(char));
		memcpy(Xrow, X + ALPHLEN * (threadIdx.x + blockDim.x * blockIdx.x), ALPHLEN * sizeof(uint16_t));

		for (int i = 0; i < s1Len + 1; i++)
		{
			if (i > 0)
				s1c = s1[i - 1];

			x = Xrow[s1c - ALPHSTART];

			diagVal = __shfl_up_sync(0xffffffff, prevVal, 1);

			if (threadIdx.x % WARPSIZE == 0 && threadIdx.x != 0)
				diagVal = sharedDiagArray[(threadIdx.x / WARPSIZE) - 1];
			else if (threadIdx.x == 0 && blockIdx.x != 0)
				diagVal = globalDiagArray[blockIdx.x - 1];

			grid.sync();

			if (i == 0)
				foundVal = threadIdx.x + blockDim.x * blockIdx.x;
			else if (threadIdx.x + blockDim.x * blockIdx.x == 0)
				foundVal = i;
			else if (s1c == s2c)
				foundVal = diagVal;
			else if (x == 0)
				foundVal = 1 + Min(prevVal, diagVal, i + threadIdx.x + blockDim.x * blockIdx.x - 1);
			else
				foundVal = 1 + Min(prevVal, diagVal, D[GetDInd(i - 1, x - 1, s2Len + 1)] + threadIdx.x + blockDim.x * blockIdx.x - 1 - x);	

			D[GetDInd(i, threadIdx.x + blockDim.x * blockIdx.x, s2Len + 1)] = foundVal;
			prevVal = foundVal;

			if (threadIdx.x % WARPSIZE == WARPSIZE - 1 && threadIdx.x != WARPSIZE * WARPSINBLOCK - 1)
				sharedDiagArray[(threadIdx.x - (WARPSIZE - 1)) / WARPSIZE] = prevVal;
			else if (threadIdx.x == WARPSIZE * WARPSINBLOCK - 1 && blockIdx.x != gridDim.x - 1)
				globalDiagArray[blockIdx.x] = prevVal;

			grid.sync();
		}
	}
}

void saveToFile(uint16_t* dD, uint16_t* hD, char* s1, char* s2, const uint16_t s1Len, const uint16_t s2Len)
{
	FILE* cpu_outputfile = fopen("CPU_OUTPUTFILE.txt", "w");
	if (cpu_outputfile == NULL)
	{
		fprintf(stderr, "Nie mozna otworzyc pliku do zapisu wynikow z cpu\n");
	}

	FILE* gpu_outputfile = fopen("GPU_OUTPUTFILE.txt", "w");
	if (gpu_outputfile == NULL)
	{
		fprintf(stderr, "Nie mozna otworzyc pliku do zapisu wynikow z gpu\n");
		fclose(cpu_outputfile);
	}

	fwrite(" ", sizeof(char), 1, cpu_outputfile);
	fwrite(",", sizeof(char), 1, cpu_outputfile);
	fwrite(" ", sizeof(char), 1, gpu_outputfile);
	fwrite(",", sizeof(char), 1, gpu_outputfile);
	
	for (int i = 0; i < s2Len + 1; i++)
	{
		if (i == 0)
		{
			fwrite(" ", sizeof(char), 1, cpu_outputfile);
			fwrite(",", sizeof(char), 1, cpu_outputfile);
			fwrite(" ", sizeof(char), 1, gpu_outputfile);
			fwrite(",", sizeof(char), 1, gpu_outputfile);
		}
		else
		{
			fwrite(s2 + i - 1, sizeof(char), 1, cpu_outputfile);
			fwrite(",", sizeof(char), 1, cpu_outputfile);
			fwrite(s2 + i - 1, sizeof(char), 1, gpu_outputfile);
			fwrite(",", sizeof(char), 1, gpu_outputfile);
		}
	}
	fprintf(cpu_outputfile, "\n");
	fprintf(gpu_outputfile, "\n");

	for (int i = 0; i < s1Len + 1; i++)
	{
		if (i == 0)
		{
			fwrite(" ", sizeof(char), 1, cpu_outputfile);
			fwrite(",", sizeof(char), 1, cpu_outputfile);
			fwrite(" ", sizeof(char), 1, gpu_outputfile);
			fwrite(",", sizeof(char), 1, gpu_outputfile);
		}
		else
		{
			fwrite(s1 + i - 1, sizeof(char), 1, cpu_outputfile);
			fwrite(",", sizeof(char), 1, cpu_outputfile);
			fwrite(s1 + i - 1, sizeof(char), 1, gpu_outputfile);
			fwrite(",", sizeof(char), 1, gpu_outputfile);
		}

		for (int j = 0; j < s2Len + 1; j++)
		{
			fprintf(cpu_outputfile, "%u,", hD[GetDInd(i, j, s2Len + 1)]);
			fprintf(gpu_outputfile, "%u,", dD[GetDInd(i, j, s2Len + 1)]);
		}
		fprintf(cpu_outputfile, "\n");
		fprintf(gpu_outputfile, "\n");
	}

	fclose(cpu_outputfile);
	fclose(gpu_outputfile);
}