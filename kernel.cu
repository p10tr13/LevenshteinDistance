
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

#define ALPHLEN 26

#define ALPHSTART 'A'

#define ALPHEND 'Z'

#define WARPSIZE 32

#define WARPSINBLOCK 32

#define GPU_OUTPUTFILEPATH "GPU_OUTPUTFILE.txt"

#define CPU_OUTPUTFILEPATH "CPU_OUTPUTFILE.txt"

// Structure of a single linked list for operations of changing s1 to s2
// Operation type
typedef enum {
	ADD,
	DEL,
	REPLACE
} OperationType;

// List Node
typedef struct Node
{
	uint32_t ind;
	char letter;
	OperationType type;
	struct Node* next;
} Node;

cudaError_t LevenshteinGPU(char* s1, char* s2, const uint32_t s1Len, const uint32_t s2Len, uint32_t* D, long long* gpu_alloc_time, long long* calculateD_time, long long* copy_to_h_time, long long* calculateX_time);
__host__ uint32_t checkWord(char* s);
__host__ __device__ uint32_t GetDInd(uint32_t i, uint32_t j, uint32_t width);
__host__ __device__ uint32_t Min(uint32_t num1, uint32_t num2, uint32_t num3);
__host__ void CPULevenshtein(char* s1, uint32_t s1Len, char* s2, uint32_t s2Len, uint32_t* D);
__host__ void PrintD(uint32_t* D, uint32_t height, uint32_t width, char* s1, char* s2);
__host__ void PrintX(uint32_t* X, uint32_t height, uint32_t width, char* s2);
__host__ Node* RetrievePath(uint32_t* D, uint32_t height, uint32_t width, char* s1, char* s2);
__host__ bool EasyCheck(uint32_t* hD, uint32_t* dD, uint32_t height, uint32_t width);
__global__ void calculateX(uint32_t* X, char* s2, const uint32_t s2Len);
__global__ void calculateD(uint32_t* D, uint32_t* X, char* s1, char* s2, const uint32_t s1Len, const uint32_t s2Len, uint32_t* globalDiagArray);
__global__ void warmup(int i);
__host__ void wordsLen(char* filepath, int* strLen1, int* strLen2);
__host__ char* getLineFromFile(FILE* file, int strLen);
__host__ void saveDToFile(uint32_t* dD, uint32_t* hD, char* s1, char* s2, const uint32_t s1Len, const uint32_t s2Len, char* cpu_outputfilepath, char* gpu_outputfilepath);
__host__ void savePathToFile(Node* head, char* result_file_path);
__host__ void howToUse();

// Functions prototypes for single linked list operations
__host__ Node* createNode(uint32_t ind, char letter, OperationType type);
__host__ void addToFrontList(Node** head, uint32_t ind, char letter, OperationType type);
__host__ void addToEndList(Node** tail, Node** head, uint32_t ind, char letter, OperationType type);
__host__ void printList(Node* head);
__host__ void freeList(Node* head);

int main(int argc, char* argv[])
{
	cudaError_t cudaStatus;
	long long cpu_time = 0, gpu_time = 0, gpu_calculateD_time = 0, gpu_prepare_time = 0, gpu_copy_to_h_time = 0, gpu_calculateX_time = 0, cpu_path_time = 0, gpu_path_time = 0, save_to_file_time = 0;
	int mode = -1, print_mode = -1;
	char* s1, * s2, * s1_s2_file, * result_file_path;

	if (argc > 4)
	{
		print_mode = atoi(argv[4]);
		if (print_mode < 0 || print_mode > 6)
		{
			howToUse();
			return 0;
		}
	}

	if (argc > 3)
	{
		mode = atoi(argv[1]);
		switch (mode)
		{
		case 1:
		{
			s1 = argv[2];
			s2 = argv[3];
			break;
		}
		case 2:
		{
			srand(time(NULL));
			int len1 = atoi(argv[2]), len2 = atoi(argv[3]);
			if (len1 < 2 || len2 < 2)
			{
				howToUse();
				return 0;
			}

			s1 = (char*)malloc(sizeof(char) * len1);
			s2 = (char*)malloc(sizeof(char) * len2);

			for (int i = 0; i < len1 - 1; i++)
				s1[i] = 'A' + rand() % 26;

			for (int i = 0; i < len2 - 1; i++)
				s2[i] = 'A' + rand() % 26;

			s1[len1 - 1] = '\0';
			s2[len2 - 1] = '\0';
			break;
		}
		default:
		{
			howToUse();
			return 0;
		}
		}
	}
	else if (argc == 3)
	{
		s1_s2_file = argv[1];
		result_file_path = argv[2];
		mode = 0;
		print_mode = 0;
		int strLen1 = 0, strLen2 = 0;
		wordsLen(s1_s2_file, &strLen1, &strLen2);

		FILE* file = fopen(s1_s2_file, "r");
		if (file == NULL)
		{
			fprintf(stderr, "Nie mozna otworzyc pliku ze slowami.\n");
		}

		s1 = getLineFromFile(file, strLen1);
		s2 = getLineFromFile(file, strLen2);

		fclose(file);
	}
	else
	{
		howToUse();
		return 0;
	}

	// Checking if the words are valid
	uint32_t s1Len = checkWord(s1), s2Len = checkWord(s2);
	if (s1Len == 0 || s2Len == 0)
	{
		printf("Podane slowa sa niepoprawne!\n");
		return 0;
	}

	printf("\nLength of words s1: %d, s2: %d\n", s1Len, s2Len);

	uint32_t* D_CPU = (uint32_t*)malloc(sizeof(uint32_t) * (s1Len + 1) * (s2Len + 1));
	if (D_CPU == NULL)
	{
		std::cout << "D_CPU Memory Allocation Failed";
		exit(1);
	}

	// Calculating D matrix on CPU
	auto cpu_ts = high_resolution_clock::now();
	CPULevenshtein(s1, s1Len, s2, s2Len, D_CPU);
	auto cpu_te = high_resolution_clock::now();
	cpu_time += duration_cast<microseconds> (cpu_te - cpu_ts).count();

	// Extraction of transformations of string s1 from matrix calculated on the CPU
	auto cpu_path_ts = high_resolution_clock::now();
	Node* CPU_result = RetrievePath(D_CPU, s1Len + 1, s2Len + 1, s1, s2);
	auto cpu_path_te = high_resolution_clock::now();
	cpu_path_time += duration_cast<microseconds> (cpu_path_te - cpu_path_ts).count();

	uint32_t* D_GPU = (uint32_t*)malloc(sizeof(uint32_t) * (s1Len + 1) * (s2Len + 1));
	if (D_GPU == NULL)
	{
		std::cout << "D_GPU Memory Allocation Failed";
		free(D_CPU);
		freeList(CPU_result);
		return 0;;
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Warm up function to ensure the GPU is ready for calculations
	warmup <<<1, 32 >>> (1);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching warmup!\n", cudaStatus);
		goto Error;
	}

	// Calculating D matrix on GPU
	auto gpu_ts = high_resolution_clock::now();
	cudaStatus = LevenshteinGPU(s1, s2, s1Len, s2Len, D_GPU, &gpu_prepare_time, &gpu_calculateD_time, &gpu_copy_to_h_time, &gpu_calculateX_time);
	auto gpu_te = high_resolution_clock::now();
	gpu_time += duration_cast<microseconds> (gpu_te - gpu_ts).count();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "LevenshteinGPU failed! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// Checking if the D matrices calculated on CPU and GPU are the same
	if (EasyCheck(D_CPU, D_GPU, s1Len + 1, s2Len + 1))
		printf("D matrices from CPU and GPU are the same.\n");
	else
		printf("D matrices from CPU and GPU are the different.!\n");

	printf("CPU result: %d\n", D_CPU[(s1Len + 1) * (s2Len + 1) - 1]);
	printf("GPU result: %d\n", D_GPU[(s1Len + 1) * (s2Len + 1) - 1]);

	// Extraction of transformations of string s1 from matrix calculated on the GPU
	auto gpu_path_ts = high_resolution_clock::now();
	Node* GPU_result = RetrievePath(D_GPU, s1Len + 1, s2Len + 1, s1, s2);
	auto gpu_path_te = high_resolution_clock::now();
	gpu_path_time += duration_cast<microseconds> (gpu_path_te - gpu_path_ts).count();

	// Printing the results based on the selected print mode
	switch (print_mode)
	{
	case 0:
	{
		if (mode != 0)
		{
			printf("Cannot save the file woithout path.\n");
			break;
		}
		auto save_to_file_ts = high_resolution_clock::now();
		savePathToFile(GPU_result, result_file_path);
		auto save_to_file_te = high_resolution_clock::now();
		save_to_file_time = duration_cast<microseconds> (save_to_file_te - save_to_file_ts).count();
		break;
	}
	case 1:
	{
		PrintD(D_GPU, s1Len + 1, s2Len + 1, s1, s2);
		break;
	}
	case 2:
	{
		printList(GPU_result);
		break;
	}
	case 3:
	{
		saveDToFile(D_GPU, D_CPU, s1, s2, s1Len, s2Len, CPU_OUTPUTFILEPATH, GPU_OUTPUTFILEPATH);
		break;
	}
	case 4:
	{
		PrintD(D_GPU, s1Len + 1, s2Len + 1, s1, s2);
		printList(GPU_result);
		break;
	}
	case 5:
	{
		printList(GPU_result);
		saveDToFile(D_GPU, D_CPU, s1, s2, s1Len, s2Len, CPU_OUTPUTFILEPATH, GPU_OUTPUTFILEPATH);
		break;
	}
	case 6:
	{
		PrintD(D_GPU, s1Len + 1, s2Len + 1, s1, s2);
		printList(GPU_result);
		saveDToFile(D_GPU, D_CPU, s1, s2, s1Len, s2Len, CPU_OUTPUTFILEPATH, GPU_OUTPUTFILEPATH);
		break;
	}
	}

	std::cout << endl << "CPU time: " << setw(7) << 0.001 * cpu_time << " nsec" << endl;
	std::cout << "Whole GPU time: " << setw(7) << 0.001 * gpu_time << " nsec" << endl;
	std::cout << "GPU memory alloc + copy time: " << setw(7) << 0.001 * gpu_prepare_time << " nsec" << endl;
	std::cout << "CalculateX time: " << setw(7) << 0.001 * gpu_calculateX_time << " nsec" << endl;
	std::cout << "CalculateD time: " << setw(7) << 0.001 * gpu_calculateD_time << " nsec" << endl;
	std::cout << "Algorithm(CalculateD + CalculateX) GPU time: " << setw(7) << 0.001 * gpu_calculateD_time + 0.001 * gpu_calculateX_time << " nsec" << endl;
	std::cout << "Copy to host GPU time: " << setw(7) << 0.001 * gpu_copy_to_h_time << " nsec" << endl;
	std::cout << "RetrievePath CPU time: " << setw(7) << 0.001 * cpu_path_time << " nsec" << endl;
	std::cout << "RetrievePath GPU time: " << setw(7) << 0.001 * gpu_path_time << " nsec" << endl;
	if (print_mode == 0)
		std::cout << "Save results to file time: " << setw(7) << 0.001 * save_to_file_time << " nsec" << endl;

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		goto Error;
	}

Error:
	freeList(CPU_result);
	freeList(GPU_result);
	free(D_CPU);
	free(D_GPU);
	if (mode == 2 || mode == 0)
	{
		free(s1);
		free(s2);
	}
	return 0;
}

/**
 * Checks if the word consists of valid characters (A-Z) and calculates the lenght of it.
 *
 * @param s - pointer to the word string.
 *
 * @return length of the word if valid, 0 if invalid.
 */
__host__ uint32_t checkWord(char* s)
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

/**
 * Conwerts two-dimensional indices (i, j) to a one-dimensional index for the D array.
 *
 * @param i - row index.
 * @param j - column index.
 * @param width - array width.
 *
 * @return index in the one-dimensional array D.
 */
__host__ __device__ uint32_t GetDInd(uint32_t i, uint32_t j, uint32_t width)
{
	return i * width + j;
}

/**
 * Returns the minimum value from three given numbers.
 *
 * @param num1 - first number.
 * @param num2 - second number.
 * @param num3 - thrid number.
 *
 * @return lowest value among the three numbers.
 */
__host__ __device__ uint32_t Min(uint32_t num1, uint32_t num2, uint32_t num3)
{
	if (num1 <= num2 && num1 <= num3)
		return num1;
	else if (num2 <= num1 && num2 <= num3)
		return num2;
	else return num3;
}

/**
 * Basic CPU implementation of the Levenshtein distance algorithm.
 *
 * @param[in] s1 - pointer to the first word (string s1).
 * @param[in] s1Len - length of the first word s1.
 * @param[in] s2 - pointer to the second word (string s2).
 * @param[in] s2Len - length of the second word s2.
 * @param[out] D - array storing the Levenshtein distance matrix.
 */
__host__ void CPULevenshtein(char* s1, uint32_t s1Len, char* s2, uint32_t s2Len, uint32_t* D)
{
	for (uint32_t i = 0; i < s1Len + 1; i++)
	{
		for (uint32_t j = 0; j < s2Len + 1; j++)
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

/**
 * Prints the D matrix on the console.
 *
 * @param D - pointer to the D matrix.
 * @param height - height of the D matrix.
 * @param width - width of the D matrix.
 * @param s1 - pointer to the first word s1.
 * @param s2 - pointer to the second word s2.
 */
__host__ void PrintD(uint32_t* D, uint32_t height, uint32_t width, char* s1, char* s2)
{
	printf("\n");
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

/**
 * Prints the X matrix on the console.
 *
 * @param X - pointer to the X matrix.
 * @param height - height of the X matrix.
 * @param width - width of the X matrix.
 * @param s2 - pointer to the second word s2.
 */
__host__ void PrintX(uint32_t* X, uint32_t height, uint32_t width, char* s2)
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

/**
 * Creates a linked list of operations to transform string s1 into string s2 based on the D matrix.
 *
 * @param D - pointer to the D matrix.
 * @param height - height of the D matrix.
 * @param width - width of the D matrix.
 * @param s1 - pointer to the first word s1.
 * @param s2 - pointer to the second word s2.
 *
 * @return pointer to the head of the linked list containing the operations.
 */
__host__ Node* RetrievePath(uint32_t* D, uint32_t height, uint32_t width, char* s1, char* s2)
{
	int i = height - 1, j = width - 1, added = 0, maxToAdd;
	Node* listHead = NULL, * listTail = NULL;
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
			else if (min == D[GetDInd(i, j - 1, width)])
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

/**
 * Checks if two arrays D are identical.
 *
 * @param hD - pointer to the first array D (host memory).
 * @param dD - pointer to the second array D (device memory).
 * @param height - height of the arrays D.
 * @param width - width of the arrays D.
 *
 * @return true if the arrays are identical, false otherwise.
 */
__host__ bool EasyCheck(uint32_t* hD, uint32_t* dD, uint32_t height, uint32_t width)
{
	int len = height * width;
	for (int i = 0; i < len; i++)
	{
		if (hD[i] != dD[i])
		{
			uint32_t hel1 = hD[i], hel2 = dD[i];
			return false;
		}
	}
	return true;
}

/**
 * Creates a new node for the linked list.
 *
 * @param ind - index of the letter.
 * @param letter - letter to be stored in the node.
 * @param type - type of operation (ADD, DEL, REPLACE).
 *
 * @return pointer to the newly created node.
 */
__host__ Node* createNode(uint32_t ind, char letter, OperationType type)
{
	Node* newNode = (Node*)malloc(sizeof(Node));
	if (!newNode)
	{
		printf("Memory allocation error when creating a node.\n");
		exit(EXIT_FAILURE);
	}
	newNode->ind = ind;
	newNode->letter = letter;
	newNode->type = type;
	newNode->next = NULL;
	return newNode;
}

/**
 * Add a new node to the front of the linked list.
 *
 * @param head - pointer to the head of the linked list.
 * @param ind - index of the letter to be added.
 * @param letter - letter to be added to the node.
 * @param type - type of operation (ADD, DEL, REPLACE).
 */
__host__ void addToFrontList(Node** head, uint32_t ind, char letter, OperationType type)
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

/**
 * Add a new node to the end of the linked list.
 *
 * @param tail - pointer to the tail of the linked list.
 * @param head - pointer to the head of the linked list.
 * @param ind - index of the letter to be added.
 * @param letter - letter to be added to the node.
 * @param type - operation type (ADD, DEL, REPLACE).
 */
__host__ void addToEndList(Node** tail, Node** head, uint32_t ind, char letter, OperationType type)
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

/**
 * Writes the contents of the linked list to the console.
 *
 * @param head - pointer to the head of the linked list.
 */
__host__ void printList(Node* head)
{
	printf("\n");
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

/**
 * Frees the memory allocated for the linked list.
 *
 * @param head - pointer to the head of the linked list.
 */
__host__ void freeList(Node* head)
{
	Node* current = head;
	while (current != NULL)
	{
		Node* temp = current;
		current = current->next;
		free(temp);
	}
}

/**
 * Helper function that aggregates the main CUDA calls in the program (memory allocation on GPU, data transfer to GPU, algorithm execution, results reading from GPU, memory deallocation on GPU).
 * 
 * @param[in] s1 - pointer to the first word (string s1).
 * @param[in] s2 - pointer to the second word (string s2).
 * @param[in] s1Len - length of the first word s1.
 * @param[in] s2Len - length of the second word s2.
 * @param[out] D - pointer to the array storing the Levenshtein distance matrix.
 * @param[out] gpu_prepare_time - pointer to a variable that stores the time taken to prepare the GPU.
 * @param[out] calculateD_time - pointer to a variable that stores the time taken to calculate the D array on the GPU.
 * @param[out] copy_to_h_time - pointer to a variable that stores the time taken to copy the results from GPU to host memory.
 * @param[out] calculateX_time - pointer to a variable that stores the time taken to calculate the X array on the GPU.
 * @return error code indicating the success or failure of the CUDA operations.
 */
cudaError_t LevenshteinGPU(char* s1, char* s2, const uint32_t s1Len, const uint32_t s2Len, uint32_t* D, long long* gpu_prepare_time, long long* calculateD_time, long long* copy_to_h_time, long long* calculateX_time)
{
	uint32_t* d_X;
	uint32_t* d_D;
	char* d_s1, * d_s2;
	cudaError_t cudaStatus;
	uint32_t* d_globalDiagArray;

	auto gpu_preapare_ts = high_resolution_clock::now();

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

	cudaStatus = cudaMalloc(&d_X, (s2Len + 1) * ALPHLEN * sizeof(uint32_t));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "d_X cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_D, (s1Len + 1) * (s2Len + 1) * sizeof(uint32_t));
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "d_D cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc(&d_globalDiagArray, blocks * sizeof(uint32_t));
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

	*gpu_prepare_time = duration_cast<microseconds> (gpu_preapare_te - gpu_preapare_ts).count();

	auto gpu_calculateX_ts = high_resolution_clock::now();
	calculateX << <1, 32 >> > (d_X, d_s2, s2Len);

	cudaStatus = cudaDeviceSynchronize();
	auto gpu_calculateX_te = high_resolution_clock::now();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching calculateX!\n", cudaStatus);
		goto Error;
	}

	*calculateX_time = duration_cast<microseconds> (gpu_calculateX_te - gpu_calculateX_ts).count();

	void* args[] = { (void*)&d_D, (void*)&d_X, (void*)&d_s1, (void*)&d_s2, (void*)&s1Len, (void*)&s2Len, (void*)&d_globalDiagArray };

	// Initialize, then launch
	auto gpu_calculateD_ts = high_resolution_clock::now();
	cudaLaunchCooperativeKernel((void*)calculateD, blocks, threads, args);

	cudaStatus = cudaDeviceSynchronize();
	auto gpu_calculateD_te = high_resolution_clock::now();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	*calculateD_time = duration_cast<microseconds> (gpu_calculateD_te - gpu_calculateD_ts).count();

	auto gpu_memory_back_ts = high_resolution_clock::now();
	cudaStatus = cudaMemcpy(D, d_D, (s1Len + 1) * (s2Len + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost);
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

/**
 * Fills the X matrix with values based on the second word s2.
 *
 * @param X - pointer to the X matrix in device memory.
 * @param s2 - pointer to the second word s2 in device memory.
 * @param s2Len - length of the second word s2.
 */
__global__ void calculateX(uint32_t* X, char* s2, const uint32_t s2Len)
{
	if (threadIdx.x < ALPHLEN)
	{
		uint32_t prev = 0;
		uint32_t num = 0;
		for (int i = 0; i < s2Len + 1; i++)
		{
			if (i == 0)
				num = 0;
			else if (s2[i - 1] == ALPHSTART + threadIdx.x)
				num = i;
			else
				num = prev;

			X[ALPHLEN * i + threadIdx.x] = num;
			prev = num;
		}
	}
}

/**
 * Fills the D matrix with values based on the Levenshtein distance algorithm.
 *
 * @param D - pointer to the D matrix in device memory.
 * @param X - pointer to already filled X matrix in device memory.
 * @param s1 - pointer to the first word s1 in device memory.
 * @param s2 - pointer to the second word s2 in device memory.
 * @param s1Len - length of the first word s1.
 * @param s2Len - length of the second word s2.
 * @param globalDiagArray - pointer to the global diagonal array used for exchanging diagonal values between blocks.
 */
__global__ void calculateD(uint32_t* D, uint32_t* X, char* s1, char* s2, const uint32_t s1Len, const uint32_t s2Len, uint32_t* globalDiagArray)
{
	grid_group grid = this_grid();
	__shared__ uint32_t sharedDiagArray[WARPSINBLOCK - 1]; // A table for exchanging diagonal variables in a given block

	if (threadIdx.x + blockIdx.x * blockDim.x < s2Len + 1)
	{
		uint32_t Xcol[ALPHLEN];
		char s1c, s2c; // s1c - letter of iteration, s2c - letter in the given column
		uint32_t foundVal = threadIdx.x + blockDim.x * blockIdx.x, prevVal = threadIdx.x + blockDim.x * blockIdx.x, diagVal, x;

		// diagVal - D[i-1,j-1] value, for the current iteration (diagonal value in matrix)
		// prevVal - D[i-1,j] value, for the current iteration (upper value in matrix)
		// foundVal - currently found value for D[i,j]

		// Getting the letter (corresponding to the column) that the thread will use in the algorithm
		if (threadIdx.x + blockDim.x * blockIdx.x != 0)
			s2c = s2[threadIdx.x + blockDim.x * blockIdx.x - 1];
		// Fetching the entire column of matrix X for each thread (because one thread will only use the same "own" column in matrix X)
		memcpy(Xcol, X + ALPHLEN * (threadIdx.x + blockDim.x * blockIdx.x), ALPHLEN * sizeof(uint32_t));

		for (int i = 0; i < s1Len + 1; i++)
		{
			// Fetching s1c for the current iteration
			if (i > 0)
				s1c = s1[i - 1];

			x = Xcol[s1c - ALPHSTART];

			// Exchange variables between threads in the warp
			diagVal = __shfl_up_sync(0xffffffff, prevVal, 1);

			// Fetching variables exchanged between warps
			if (threadIdx.x % WARPSIZE == 0 && threadIdx.x != 0)
				diagVal = sharedDiagArray[(threadIdx.x / WARPSIZE) - 1];
			// Fetching variables exchanged between blocks
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

			// Saving the result to D matrix
			D[GetDInd(i, threadIdx.x + blockDim.x * blockIdx.x, s2Len + 1)] = foundVal;

			// Store the value found as the previous one so that it can be passed to the next thread in the next iteration
			prevVal = foundVal;

			// Saving a variable for exchange between warps
			if (threadIdx.x % WARPSIZE == WARPSIZE - 1 && threadIdx.x != WARPSIZE * WARPSINBLOCK - 1)
				sharedDiagArray[(threadIdx.x - (WARPSIZE - 1)) / WARPSIZE] = prevVal;
			// Saving a variable for exchange between blocks
			else if (threadIdx.x == WARPSIZE * WARPSINBLOCK - 1 && blockIdx.x != gridDim.x - 1)
				globalDiagArray[blockIdx.x] = prevVal;

			grid.sync();
		}
	}
}

/**
 * Warmup kernel function that does nothing but is used to warm up the GPU before running the main algorithm.
 *
 * @param i - no purpose integer parameter, used to ensure the kernel is executed.
 */
__global__ void warmup(int i)
{
	int res = threadIdx.x * i;
}

/**
 * Calculates the lengths of the words from a file and saves them in the provided pointers.
 *
 * @param filepath - file path to the text file containing the words.
 * @param strLen1 - pointer to a variable where the length of word 1 will be saved.
 * @param strLen2 - pointer to a variable where the length of word 2 will be saved.
 */
__host__ void wordsLen(char* filepath, int* strLen1, int* strLen2)
{
	FILE* file = fopen(filepath, "r");
	if (file == NULL)
	{
		fprintf(stderr, "Cant open the file with strings.\n");
	}

	int c;
	int lineNum = 0;

	while ((c = fgetc(file)) != EOF)
	{
		if (c == '\n')
		{
			lineNum++;
			if (lineNum == 2)
				break;
		}
		else
		{
			if (lineNum == 0)
				*strLen1 += 1;
			else
				*strLen2 += 1;
		}
	}

	fclose(file);
}

/**
 * Saves a line from a file to a dynamically allocated string.
 *
 * @param file - pointer to the file from which the line will be read.
 * @param strLen - length of the line to be read.
 */
__host__ char* getLineFromFile(FILE* file, int strLen)
{
	char* line = (char*)malloc(sizeof(char) * (strLen + 1));

	char c;

	if (fgets(line, strLen + 1, file) == NULL)
		printf("Error reading strings from file.\n");
	fgetc(file);

	return line;
}

/**
 * Saves the D matrices to a file in a specific format.
 *
 * @param dD - pointer to the first array D (GPU calculated).
 * @param hD - pointer to the second array D (CPU calculated).
 * @param s1 - pointer to the first word s1.
 * @param s2 - pointer to the second word s2.
 * @param s1Len - length of the first word s1.
 * @param s2Len - length of the second word s2.
 * @param cpu_outputfilepath - output file path for the CPU results.
 * @param gpu_outputfilepath - output file path for the GPU results.
 */
__host__ void saveDToFile(uint32_t* dD, uint32_t* hD, char* s1, char* s2, const uint32_t s1Len, const uint32_t s2Len, char* cpu_outputfilepath, char* gpu_outputfilepath)
{
	FILE* cpu_outputfile = fopen(cpu_outputfilepath, "w");
	if (cpu_outputfile == NULL)
	{
		fprintf(stderr, "Cannot open file for saving CPU results\n");
	}

	FILE* gpu_outputfile = fopen(gpu_outputfilepath, "w");
	if (gpu_outputfile == NULL)
	{
		fprintf(stderr, "Cannot open file for saving GPU results\n");
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

/**
 * Saves the operations from the linked list to a file in a specific format.
 *
 * @param head - pointer to the head of the linked list containing the operations.
 * @param result_file_path - file path where the operations will be saved.
 */
__host__ void savePathToFile(Node* head, char* result_file_path)
{
	Node* current = head;
	FILE* result_file = fopen(result_file_path, "w");
	if (result_file == NULL)
	{
		fprintf(stderr, "Cannot open file for saving linked list results.\n");
	}

	while (current != NULL)
	{
		switch (current->type)
		{
		case ADD:
		{
			fprintf(result_file, "I %d %c\n", current->ind, current->letter);
			break;
		}
		case DEL:
		{
			fprintf(result_file, "D %d\n", current->ind);
			break;
		}
		case REPLACE:
		{
			fprintf(result_file, "R %d %c\n", current->ind, current->letter);
			break;
		}
		}
		current = current->next;
	}
	fclose(result_file);
}

/**
 * Writes to the console how to use the program.
 */
__host__ void howToUse()
{
	printf("Bad arguments were entered\n");
	printf("Simple program call: s1_s2_txt_file_path txt_output_file_path\n");
	printf("In case of advance modes please refer to README file\n");
	printf("\n");
}