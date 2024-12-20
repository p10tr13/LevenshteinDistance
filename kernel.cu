
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdint.h>
#include <cstdlib>
#include <string.h>

#define S2 "sadsfa"

#define S1 "dsags"

#define ALPHLEN 26

#define ALPHSTART 'a'

#define ALPHEND 'z'

// Struktura listy jednokierunkowej, dla opisywania operacji zmiany s1 na s2
// Typ operacji
typedef enum {
	ADD,
	DELETE,
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

cudaError_t LevenshteinGPU(char* s1, char* s2, const uint16_t s1Len, const uint16_t s2Len, uint16_t* D, uint16_t* X);

__host__ uint16_t checkWord(char* s);

__host__ __device__ uint32_t GetDInd(uint16_t i, uint16_t j, uint16_t width);

__host__ __device__ uint16_t Min(uint16_t num1, uint16_t num2, uint16_t num3);

__host__ void CPULevenshtein(char* s1, uint16_t s1Len, char* s2, uint16_t s2Len, uint16_t* D);

__host__ void PrintD(uint16_t* D, uint16_t height, uint16_t width, char* s1, char* s2);

__host__ void PrintX(uint16_t* X, uint16_t height, uint16_t width, char* s2);

__host__ Node* RetrievePath(uint16_t* D, uint16_t height, uint16_t width, char* s1, char* s2);

__global__ void addKernel(int* c, const int* a, const int* b)
{

}

__global__ void calculateX(uint16_t* X, char* s2, const uint16_t s2Len);

// Operacje na liście jednokierunkowej, dla opisywania operacji zmiany s1 na s2
Node* createNode(uint16_t ind, char letter, OperationType type);
void addToFrontList(Node** head, uint16_t ind, char letter, OperationType type);
void addToEndList(Node** tail, Node** head, uint16_t ind, char letter, OperationType type);
void printList(Node* head);
void freeList(Node* head);

int main(int argc, char* argv[])
{
	cudaError_t cudaStatus;

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

	CPULevenshtein(s1, s1Len, s2, s2Len, D_CPU);

	//PrintD(D_CPU, s1Len + 1, s2Len + 1, s1, s2);

	Node* result = RetrievePath(D_CPU, s1Len + 1, s2Len + 1, s1, s2);

	printList(result);

	uint16_t* D_GPU = (uint16_t*)malloc(sizeof(uint16_t) * (s1Len + 1) * (s2Len + 1));
	uint16_t* X_GPU = (uint16_t*)malloc(sizeof(uint16_t) * ALPHSTART * (s2Len + 1));
	cudaStatus = LevenshteinGPU(s1, s2, s1Len, s2Len, D_GPU, X_GPU);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "LevenshteinGPU failed! %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}

	PrintX(X_GPU, ALPHLEN, s2Len + 1, s2);

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	freeList(result);
	free(D_CPU);
	free(D_GPU);
	free(X_GPU);
	return 0;
}

__host__ uint16_t checkWord(char* s)
{
	int i = 0;
	while (s[i] != '\0')
	{
		if (s[i] < 97 || s[i] > 122)
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

	while ((i != 0 && j != 0) || added != maxToAdd)
	{
		if (i == 0)
		{
			while (j != 0 || added != maxToAdd)
			{
				addToEndList(&listTail, &listHead, i, s2[j - 1], ADD);
				added++;
				j--;
			}
		}
		else if (j == 0)
		{
			while (i != 0 || added != maxToAdd)
			{
				addToEndList(&listTail, &listHead, i - 1, s1[i - 1], DELETE);
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
				addToEndList(&listTail, &listHead, i - 1, s1[i - 1], DELETE);
				added++;
				i--;
			}
			curCellVal = min;
		}
	}

	return listHead;
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
		case DELETE:
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
cudaError_t LevenshteinGPU(char* s1, char* s2, const uint16_t s1Len, const uint16_t s2Len, uint16_t* D, uint16_t* X)
{
	uint16_t* d_X;
	uint16_t* d_D;
	char* d_s1, * d_s2;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

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
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "s2 cudaMemcpy failed!");
		goto Error;
	}

	calculateX<<<1, 32 >>>(d_X, d_s2, s2Len);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	cudaStatus = cudaMemcpy(X, d_X, (s2Len + 1) * ALPHLEN * sizeof(uint16_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "X cudaMemcpy failed!");
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(D, d_D, (s1Len + 1) * (s2Len + 1) * sizeof(uint16_t), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "D cudaMemcpy failed!");
		goto Error;
	}

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