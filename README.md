# CUDA Levenshtein Distance Calculator

A high-performance CUDA-based implementation for calculating the **Levenshtein distance** between two strings.  
Supports multiple modes of operation, including reading from files, direct input, and random string generation.

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
- in case of replecement of a letter `R [index] [new letter]`

Example `input.txt`:

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

## 📌 Notes

* Maximum length of `s2` =
  **(Number of Streaming Multiprocessors on GPU) × 1024**
  (due to CUDA thread block configuration)

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

* CUDA-capable GPU
* NVIDIA CUDA Toolkit installed
* C++17 compiler

## 📜 License

This project is released under the MIT License.
