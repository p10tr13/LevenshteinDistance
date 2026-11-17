# CUDA Levenshtein Distance Calculator

[![CI](https://github.com/p10tr13/LevenshteinDistance/actions/workflows/ci.yml/badge.svg)](https://github.com/p10tr13/LevenshteinDistance/actions/workflows/ci.yml)

A high-performance CUDA-based implementation for calculating the **Levenshtein distance** between two strings.  
Supports multiple modes of operation, including reading from files, direct input, and random string generation.

This project was developed based on the research article:
["A parallel approximate string matching under Levenshtein distance on graphics processing units using warp-shuffle operations" (PLOS ONE, 2017)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0186251).

## 🎓 Academic Context

Project was created as part of the academic course **Graphic Processors in Computational Applications** during the **2024/2025 winter semester** at **Warsaw University of Technology**.

## 🚀 Features

- **GPU-accelerated** Levenshtein distance calculation
- Multiple **input modes** (file-based, manual, random)
- Optional **output modes** for detailed debugging and analysis
- **CPU vs GPU** matrix comparison
- Simple **command-line interface**

## 📦 Basic Usage

```bash
./LevenshteinDistance s1_s2_txt_file_path txt_output_file_path
```

**Arguments:**

- **`s1_s2_txt_file_path`** – Path to a `.txt` file where:  
  - Line 1: `s1`  
  - Line 2: `s2`
- **`txt_output_file_path`** – Path to the output file where the transformation steps from `s1` to `s2` will be saved.

**Example:**

```bash
./LevenshteinDistance input.txt output.txt
```

### **Input**
Example `input.txt`:

```
JEONJU
SUWON
```

### **Output**
Transformation steps from `s1` to `s2` will be saved in the following format:
- in case of addition of a letter `A [index] [new letter]`
- in case of deletion of a letter `D [index]`
- in case of replacement of a letter `R [index] [new letter]`

Example `output.txt`:

```
D 5
D 4
R 1 W
R 0 U
A 0 S
```

## ⚙️ Advanced Usage

```bash
./LevenshteinDistance advanced_mode arg2 arg3 [print_mode]
```

**Example:**

```bash
./LevenshteinDistance 1 ALA LAL 2
```

### Advanced Modes

| Mode | Description                                                                                                                  |
| ---- | ---------------------------------------------------------------------------------------------------------------------------- |
| 1    | Two words consisting of letters in the range `A`–`Z`.                                                                        |
| 2    | Two positive integers greater than 2 — program generates two random strings of the given lengths (letters in range `A`–`Z`). |

## 🔈 Output Modes (`print_mode`)

Optional parameter to control the output format:

| Value | Output Description                                              |
| ----- | --------------------------------------------------------------- |
| 1     | Print matrix **D** to console (**GPU only**)                    |
| 2     | Print transformation steps from `s1` to `s2` (**GPU only**)     |
| 3     | Save **D** matrices from both CPU and GPU computations to files |
| 4     | Combine modes **1** and **2**                                   |
| 5     | Combine modes **2** and **3**                                   |
| 6     | Combine modes **1**, **2**, and **3**                           |

## 🧠 How it works

The distance matrix `D` has `|s1| + 1` rows and `|s2| + 1` columns. Every GPU thread owns one column and the rows are computed one after another, following the algorithm from the article above:

1. **`calculateX`** precomputes a table `X[c][j]`: the last position `≤ j` in `s2` where letter `c` occurs. With it a cell `D[i][j]` depends only on the cell above it, the cell diagonally above-left, and one cell of the previous row found through `X`, which removes the left-to-right dependency inside a row.
2. **`calculateD`** sweeps the rows. The diagonal value `D[i-1][j-1]` belongs to the neighbouring thread, so it is passed along with `__shfl_up_sync` inside a warp, through shared memory between warps of a block, and through global memory between blocks.
3. All threads of the grid wait for each other after every row with `grid.sync()` from **cooperative groups**, so the kernel is launched with `cudaLaunchCooperativeKernel` and every block has to be resident on the GPU at the same time.
4. The list of edit operations is then read back from `D` on the CPU.

The program computes the same matrix on the CPU, compares both, and prints the time of each stage in milliseconds.

## 📌 Notes

* Maximum length of `s2` =
  **(Number of Streaming Multiprocessors on GPU) × 1024**,
  because one 1024-thread block per multiprocessor must be resident for the grid-wide synchronization.
* The whole matrix `D` is kept in GPU memory, so it needs `4 × (|s1| + 1) × (|s2| + 1)` bytes.

## 📄 Example Output

If `input.txt` contains:

```
KITTEN
SITTING
```

Running:

```bash
./LevenshteinDistance input.txt output.txt
```

Might produce `output.txt`:

```
A 6 G
R 4 I
R 0 S
```

## 🛠️ Requirements

* CUDA-capable GPU with compute capability 6.0 or newer (cooperative launch)
* NVIDIA CUDA Toolkit
* C++17 compiler

## 🔨 Building

With `nvcc` (Linux or Windows), setting `-arch` to your GPU (for example `sm_61` for a GTX 10xx, `sm_86` for an RTX 30xx):

```bash
nvcc -std=c++17 -O2 -arch=sm_61 -o levenshtein kernel.cu
```

On Windows you can also open `LevenshteinDistance.sln` in Visual Studio with the CUDA Toolkit integration.

The GitHub Actions workflow checks that the code compiles with CUDA 12.6; the runners have no GPU, so it does not run the program.

## 📜 License

This project is released under the MIT License.
