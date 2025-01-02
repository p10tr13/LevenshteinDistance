
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
__global__ void rozgrzewka(int i);
__host__ void saveToFile(uint16_t* dD, uint16_t* hD, char* s1, char* s2, const uint16_t s1Len, const uint16_t s2Len, char* cpu_outputfilepath, char* gpu_outputfilepath);
__host__ void howToUse();

// Operacje na liście jednokierunkowej, dla opisywania operacji zmiany s1 na s2
__host__ Node* createNode(uint16_t ind, char letter, OperationType type);
__host__ void addToFrontList(Node** head, uint16_t ind, char letter, OperationType type);
__host__ void addToEndList(Node** tail, Node** head, uint16_t ind, char letter, OperationType type);
__host__ void printList(Node* head);
__host__ void freeList(Node* head);

int main(int argc, char* argv[])
{
	cudaError_t cudaStatus;
	long long cpu_time = 0, gpu_time = 0, gpu_calculateD_time = 0, gpu_prepare_time = 0, gpu_copy_to_h_time = 0, gpu_calculateX_time = 0, cpu_path_time = 0, gpu_path_time = 0;
	int mode = 0, print_mode = 0;
	char* s1, * s2;

	srand(time(NULL));

	if (argc > 4)
	{
		print_mode = atoi(argv[4]);
		if (print_mode < 1 || print_mode > 6)
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
	else
	{
		howToUse();
		return 0;
	}

	// Sprawdzenie poprawności słow s1 i s2
	uint16_t s1Len = checkWord(s1), s2Len = checkWord(s2);
	if (s1Len == 0 || s2Len == 0)
	{
		printf("Podane slowa sa niepoprawne!\n");
		return 0;
	}

	uint16_t* D_CPU = (uint16_t*)malloc(sizeof(uint16_t) * (s1Len + 1) * (s2Len + 1));
	if (D_CPU == NULL)
	{
		std::cout << "D_CPU Memory Allocation Failed";
		exit(1);
	}

	// Obliczanie D na CPU
	auto cpu_ts = high_resolution_clock::now();
	CPULevenshtein(s1, s1Len, s2, s2Len, D_CPU);
	auto cpu_te = high_resolution_clock::now();
	cpu_time += duration_cast<microseconds> (cpu_te - cpu_ts).count();

	// Obliczanie przekształceń s1 na s2 z tablicy obliczonej przez CPU
	auto cpu_path_ts = high_resolution_clock::now();
	Node* CPU_result = RetrievePath(D_CPU, s1Len + 1, s2Len + 1, s1, s2);
	auto cpu_path_te = high_resolution_clock::now();
	cpu_path_time += duration_cast<microseconds> (cpu_path_te - cpu_path_ts).count();

	uint16_t* D_GPU = (uint16_t*)malloc(sizeof(uint16_t) * (s1Len + 1) * (s2Len + 1));
	if (D_GPU == NULL)
	{
		std::cout << "D_GPU Memory Allocation Failed";
		free(D_CPU);
		freeList(CPU_result);
		exit(1);
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Funkcja rozgrzewkowa (nic nie robiąca)
	rozgrzewka << <1, 32 >> > (1);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching rozgrzewka!\n", cudaStatus);
		goto Error;
	}

	// Obliczanie D na GPU
	auto gpu_ts = high_resolution_clock::now();
	cudaStatus = LevenshteinGPU(s1, s2, s1Len, s2Len, D_GPU, &gpu_prepare_time, &gpu_calculateD_time, &gpu_copy_to_h_time, &gpu_calculateX_time);
	auto gpu_te = high_resolution_clock::now();
	gpu_time += duration_cast<microseconds> (gpu_te - gpu_ts).count();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "LevenshteinGPU failed! %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// Sprawdzenie czy oba wyniki są identyczne
	if (EasyCheck(D_CPU, D_GPU, s1Len + 1, s2Len + 1))
		printf("\nMacierze D sa takie same :)\n");
	else
		printf("\nMacierze D sa inne!!\n");

	// Obliczanie przekształceń s1 na s2 z tablicy obliczonej przez GPU
	auto gpu_path_ts = high_resolution_clock::now();
	Node* GPU_result = RetrievePath(D_GPU, s1Len + 1, s2Len + 1, s1, s2);
	auto gpu_path_te = high_resolution_clock::now();
	gpu_path_time += duration_cast<microseconds> (gpu_path_te - gpu_path_ts).count();

	// Wypisywanie wyników
	switch (print_mode)
	{
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
			saveToFile(D_GPU, D_CPU, s1, s2, s1Len, s2Len, CPU_OUTPUTFILEPATH, GPU_OUTPUTFILEPATH);
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
			saveToFile(D_GPU, D_CPU, s1, s2, s1Len, s2Len, CPU_OUTPUTFILEPATH, GPU_OUTPUTFILEPATH);
			break;
		}
		case 6:
		{
			PrintD(D_GPU, s1Len + 1, s2Len + 1, s1, s2);
			printList(GPU_result);
			saveToFile(D_GPU, D_CPU, s1, s2, s1Len, s2Len, CPU_OUTPUTFILEPATH, GPU_OUTPUTFILEPATH);
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
	if (mode == 2)
	{
		free(s1);
		free(s2);
	}
	return 0;
}

/**
 * Sprawdzenie, czy dane słowo ma same litery zawierające się w zdefiniowanym alfabecie, oraz przy okazji liczy jego długość.
 *
 * @param s - wskaźnik na sprawdzane słowo
 *
 * @return długość słowa
 */
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

/**
 * Zamienia indeks tablicy dwuwymiarowej na jednowymiarową.
 *
 * @param i - numer wiersza
 * @param j - numer kolumny
 * @param width - szerokość tablicy
 *
 * @return indeks w tablicy jednowymiarowej
 */
__host__ __device__ uint32_t GetDInd(uint16_t i, uint16_t j, uint16_t width)
{
	return i * width + j;
}

/**
 * Oblicza najmniejszą wartość z trzech podanych.
 *
 * @param num1 - pierwsza liczba
 * @param num2 - druga liczba
 * @param num3 - trzecia liczba
 *
 * @return najmniejsza liczba z trzech
 */
__host__ __device__ uint16_t Min(uint16_t num1, uint16_t num2, uint16_t num3)
{
	if (num1 <= num2 && num1 <= num3)
		return num1;
	else if (num2 <= num1 && num2 <= num3)
		return num2;
	else return num3;
}

/**
 * Podstawowa implementacja algorytmu na CPU liczenia odległości Levenshteina.
 *
 * @param[in] s1 - wskaźnik na słowo s1
 * @param[in] s1Len - długość słowa s1
 * @param[in] s2 - wskaźnik na słowo s2
 * @param[in] s2Len - długość słowa s2
 * @param[out] D - tablica, w której zapisywany jest wynik
 */
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

/**
 * Wypisuje tablice D na konsole.
 *
 * @param D - wskaźnik do tablicy D
 * @param height - wysokość tablicy D
 * @param width - szerokość tablicy D
 * @param s1 - wskaźnik na słowo s1
 * @param s2 - wskaźnik na słowo s2
 */
__host__ void PrintD(uint16_t* D, uint16_t height, uint16_t width, char* s1, char* s2)
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
 * Wypisuje tablice X na konsole.
 *
 * @param X - wskaźnik do tablicy X
 * @param height - wysokość tablicy X
 * @param width - szerokość tablicy X
 * @param s2 - wskaźnik na słowo s2
 */
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

/**
 * Tworzy wynikową listę przekształceń zamian liter słowa s1, aby powstało s2 na podstawie tablicy D.
 *
 * @param D - wskaźnik do tablicy D
 * @param height - wysokość tablicy D
 * @param width - szerokość tablicy D
 * @param s1 - wskaźnik na słowo s1
 * @param s2 - wskaźnik na słowo s2
 *
 * @return wskaźnik na początek listy przekształceń
 */
__host__ Node* RetrievePath(uint16_t* D, uint16_t height, uint16_t width, char* s1, char* s2)
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
 * Sprawdza, czy tablice podane są identyczne.
 *
 * @param hD - wskaźnik do pierwszej tablicy D
 * @param dD - wskaźnik do drugiej tablicy D
 * @param height - wysokość tablicy D
 * @param width - szerokość tablicy D
 *
 * @return wynik sprawdzania identyczności tablic
 */
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

/**
 * Tworzenie nowego węzła listy.
 *
 * @param ind - indeks, wpisany w Node
 * @param letter - litera, wpisana w Node
 * @param type - typ operacji, wpisany w Node
 *
 * @return wskaźnik na nowo stworzony węzeł
 */
__host__ Node* createNode(uint16_t ind, char letter, OperationType type)
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

/**
 * Dodanie elementu na początek listy.
 *
 * @param head - wskaźnik na wskaźnik na node z początkiem listy wynikowej
 * @param ind - indeks, wpisany w dodawanego Node
 * @param letter - litera, wpisana w dodawanego Node
 * @param type - typ operacji, wpisany w dodawanego Node
 */
__host__ void addToFrontList(Node** head, uint16_t ind, char letter, OperationType type)
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
 * Dodanie elementu na koniec listy.
 *
 * @param tail - wskaźnik na wskaźnik na node z końcem listy wynikowej
 * @param head - wskaźnik na wskaźnik na node z początkiem listy wynikowej
 * @param ind - indeks, wpisany w dodawanego Node
 * @param letter - litera, wpisana w dodawanego Node
 * @param type - typ operacji, wpisany w dodawanego Node
 */
__host__ void addToEndList(Node** tail, Node** head, uint16_t ind, char letter, OperationType type)
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
 * Wypisanie listy wynikowej na konsole.
 *
 * @param head - wskaźnik na początek listy wynikowej
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
 * Zwolnienie pamięci zajętej przez listę wynikową.
 *
 * @param head - wskaźnik na początek listy wynikowej
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
 * Funckja pomocnicza, agregująca główne wywołania w programie CUDA (alokacja pamięci w GPU, zapis danych do GPU, algorytm, odczyt wyników z GPU, zwolnienie pamięci w GPU)
 *
 * @param[in] s1 - wskaźnik do tablicy char pierwszego ze słów
 * @param[in] s2 - wskaźnik do tablicy char drugiego ze słów
 * @param[in] s1Len - długość słowa s1
 * @param[in] s2Len - długość słowa s2
 * @param[out] D - wskaźnik do tablicy wynikowej, gdzie funkcja zapisze D (wymiar: (s1Len + 1) * (s2Len + 1))
 * @param[out] gpu_prepare_time - wskaźnik na zmienną z czasem, w jakim przygotowywujemy GPU do wywołania kernela (obliczenie parametrów wywołania kernela + alokacja + kopiowanie)
 * @param[out] calculateD_time - wskaźnik na zmienną z czasem, w jakim wykonujemy funckję obliczania samej tablicy D (bez X)
 * @param[out] copy_to_h_time - wskaźnik na zmienną z czasem, w jakim kopiujemy dane z GPU do hosta
 * @param[out] calculateX_time - wskaźnik na zmienną z czasem, w jakim wykonujemy funckję obliczania samej tablicy X (bez D)
 * @return możliwy error, który zaszedł podczas "CUDA-owych" operacji
 */
cudaError_t LevenshteinGPU(char* s1, char* s2, const uint16_t s1Len, const uint16_t s2Len, uint16_t* D, long long* gpu_prepare_time, long long* calculateD_time, long long* copy_to_h_time, long long* calculateX_time)
{
	uint16_t* d_X;
	uint16_t* d_D;
	char* d_s1, * d_s2;
	cudaError_t cudaStatus;
	uint16_t* d_globalDiagArray;

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

	*calculateD_time = duration_cast<microseconds> (gpu_calculateD_te - gpu_calculateD_ts).count();

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

/**
 * Wypełnia tablice X.
 *
 * @param X - wskaźnik do tablicy X w device
 * @param s2 - wskaźnik na słowo s2 w device
 * @param s2Len - długość słowa s2
 */
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

/**
 * Wypełnia tablice D.
 *
 * @param D - wskaźnik do tablicy D w device
 * @param X - wskaźnik do wypełnionej tablicy X w device
 * @param s1 - wskaźnik na słowo s1 w device
 * @param s2 - wskaźnik na słowo s2 w device
 * @param s1Len - długośc słowa s1
 * @param s2Len - długość słowa s2
 * @param globalDiagArray - wskaźnik do tablicy w pamięci globalnej, dzięki której będą przekazywane zmienne podczas działania programu między blokami (długość: ilość bloków)
 */
__global__ void calculateD(uint16_t* D, uint16_t* X, char* s1, char* s2, const uint16_t s1Len, const uint16_t s2Len, uint16_t* globalDiagArray)
{
	grid_group grid = this_grid();
	__shared__ uint16_t sharedDiagArray[WARPSINBLOCK - 1]; // Tablica do wymiany zmiennych diagonalnych w danym bloku

	if (threadIdx.x + blockIdx.x * blockDim.x < s2Len + 1)
	{
		uint16_t Xcol[ALPHLEN];
		char s1c, s2c; // s1c - litera iteracji, s2c - litera w danej kolumnie
		uint16_t foundVal = threadIdx.x + blockDim.x * blockIdx.x, prevVal = threadIdx.x + blockDim.x * blockIdx.x, diagVal, x;

		// diagVal - to wartość D[i-1,j-1], dla aktualnej iteracji (wartość po przekątnej w tabeli)
		// prevVal - to wartość D[i-1,j], dla aktualnej iteracji (wartość o jeden wyżej w tabeli)
		// foundVal - aktualnie znaleziona wartość dla D[i,j]

		// Pobranie litery (odpowiadającej kolumnie), z której będzie korzystał wątek w algorytmie
		if (threadIdx.x + blockDim.x * blockIdx.x != 0)
			memcpy(&s2c, s2 + threadIdx.x + blockDim.x * blockIdx.x - 1, sizeof(char));
		// Pobranie całej kolumny tablicy X dla każdego wątku (bo jeden wątek będzie tylko korzystał z tej samej "swojej" kolumny w tablicy X)
		memcpy(Xcol, X + ALPHLEN * (threadIdx.x + blockDim.x * blockIdx.x), ALPHLEN * sizeof(uint16_t));

		for (int i = 0; i < s1Len + 1; i++)
		{
			// Pobranie s1c dla aktualnej iteracji
			if (i > 0)
				s1c = s1[i - 1];

			x = Xcol[s1c - ALPHSTART];

			// Wymiana zmiennych pomiędzy wątkami w warpie
			diagVal = __shfl_up_sync(0xffffffff, prevVal, 1);

			// Pobranie zmiennych wymieninanych pomiędzy warpami
			if (threadIdx.x % WARPSIZE == 0 && threadIdx.x != 0)
				diagVal = sharedDiagArray[(threadIdx.x / WARPSIZE) - 1];
			// Pobranie zmiennych wymieninanych pomiędzy blokami
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

			// Zapisanie wyniku do tablicy D
			D[GetDInd(i, threadIdx.x + blockDim.x * blockIdx.x, s2Len + 1)] = foundVal;

			// Zapisanie wartości znalezionej jako poprzednią, aby w następnej iteracji można ją było przekazać następnemu wątkowi
			prevVal = foundVal;

			// Zapisanie zmiennej do wymiany między warpami
			if (threadIdx.x % WARPSIZE == WARPSIZE - 1 && threadIdx.x != WARPSIZE * WARPSINBLOCK - 1)
				sharedDiagArray[(threadIdx.x - (WARPSIZE - 1)) / WARPSIZE] = prevVal;
			// Zapisanie zmiennej do wymiany między blokami
			else if (threadIdx.x == WARPSIZE * WARPSINBLOCK - 1 && blockIdx.x != gridDim.x - 1)
				globalDiagArray[blockIdx.x] = prevVal;

			grid.sync();
		}
	}
}

/**
 * Funkcja rozgrzewająca GPU (nie robi nic ciekawego).
 *
 * @param i - parametr ten nie ma znaczenia.
 */
__global__ void rozgrzewka(int i)
{
	int res = threadIdx.x * i;
}

/**
 * Zapisuje tablice dD oraz hD do pliku. Plik jest w formacie .txt, a dane są rozdzielane przecinkami.
 *
 * @param dD - wskaźnik do pierwszej tablicy D
 * @param hD - wskaźnik do drugiej tablicy D
 * @param s1 - wskaźnik na słowo s1
 * @param s2 - wskaźnik na słowo s2
 * @param s1Len - długośc słowa s1
 * @param s2Len - długość słowa s2
 * @param cpu_outputfilepath - ścieżka do pliku, w którym będzie zapisany output tablicy hD
 * @param gpu_outputfilepath - ścieżka do pliku, w którym będzie zapisany output tablicy dD
 */
__host__ void saveToFile(uint16_t* dD, uint16_t* hD, char* s1, char* s2, const uint16_t s1Len, const uint16_t s2Len, char* cpu_outputfilepath, char* gpu_outputfilepath)
{
	FILE* cpu_outputfile = fopen(cpu_outputfilepath, "w");
	if (cpu_outputfile == NULL)
	{
		fprintf(stderr, "Nie mozna otworzyc pliku do zapisu wynikow z cpu\n");
	}

	FILE* gpu_outputfile = fopen(gpu_outputfilepath, "w");
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

/**
 * Wypisuje na konsole jak powinno się uruchamiać program.
 */
__host__ void howToUse()
{
	printf("Podano zle argumenty\n");
	printf("Prawidlowe argumenty: tryb_programu arg2 arg3 (optional) print_mode\n");
	printf("Przyklad: 1 ala lal 3\n");
	printf("Program ma nastepujace tryby pracy:\n");
	printf("1. dwa slowa z liter z przedzialu 'A' - 'Z'\n");
	printf("2. dwie liczby dodatnie wieksze od 2 (program losuje litery do tych dwoch slow o podanej dlugosci)\n");
	printf("Opcjonalnie po argumentach trybu mozna dodac sposob wypisana wyniku:\n");
	printf("1. Wypisanie na konsole tabeli D (GPU)\n");
	printf("2. Wypisanie na konsole listy zamian s1 na s2 (GPU)\n");
	printf("3. Zapisanie do plikow tabel D z CPU i GPU\n");
	printf("4. Tryb wypisywania 1 i 2\n");
	printf("5. Tryb wypisywania 2 i 3\n");
	printf("6. Tryb 1, 2 i 3\n");
	printf("\n");
}