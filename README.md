# LeNet5-GPU-Inference

This project aims to implement the LeNet-5 architecture on a GPU using CUDA, as part of the Hardware Signal Processing course. The goal is to understand and leverage GPU capabilities for machine learning inference tasks, with a specific focus on image classification using the MNIST dataset. The project is broken down into several components, starting with foundational matrix operations, through convolution layers, subsampling, and finishing with the complete network model, which is trained to recognize handwritten digits from the MNIST dataset.

## Project Structure

```bash
LeNet5-GPU-Inference
│
├── LeNet5_Python_Notebook/        			# LeNet5 in Python
├── Part1-MatrixOperations/        			# Matrix operations implemented in CUDACUDA
├── Part2-Convolution_And_Subsampling/   	        # Convolution layers and pooling implemented in CUDA
├── Part3-LeNet5/                  			# Full LeNet5 Architecture in CUDA
├── LeNet5-GPU-Inference.sln       			# Visual Studio Solution for GPU Inference
```

## How it works

1. **Part 1: Matrix Operations**  
   Implements essential matrix operations (e.g., addition, multiplication) on the GPU.

2. **Part 2: Convolution and Subsampling**  
   Implements convolution and pooling (subsampling) layers on the GPU.

3. **Part 3: Complete LeNet-5 Model**  
   Builds the full LeNet-5 model and imports weights and biases from the Python version for comparison.

## Prerequisites

1. **CUDA**: Make sure you have the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) installed to compile and run GPU-accelerated operations.
   
2. **Compilers**:  
   You’ll need a CUDA-compatible compiler such as [nvcc](https://developer.nvidia.com/cuda-zone) to compile the project. Optionally, you can use Visual Studio for an integrated development environment.

## How to Build and Run the Project

### Step 1: Prepare the Environment

- Install CUDA toolkits and drivers compatible with your GPU.

- Clone the repository or copy the project folder to your local machine.

### Step 2: Compile the Program

- Open the `LeNet5-GPU-Inference.sln` in Visual Studio (or the preferred IDE) and configure the project to target a CUDA-enabled environment.
- Alternatively, you can compile the project using command-line tools if using `nvcc`. For example:

   ```bash
   nvcc -o main main.cu
   ```

### Step 3: Set Up the Weights and Bias Files

- Launch the notebook in the `LeNet5_Python_Notebook/` folder to generate the weight and bias files from the Python implementation.
- Once generated, make sure the weight and bias files are placed in the `data/` folder and are named according to the references in `load_all_weights()` and `load_all_biases()` functions.

### Step 4: Run the Model

- After compiling and ensuring the weight files are set up, execute the program. It will load a test image, preprocess it, load the weights/biases, and perform inference using the GPU.
- You can modify the test image by changing the `imgIndex`. This value determines which image from the MNIST dataset is processed. For example, changing it to `imgIndex = 2;` will process the second test image.

```bash
./main
```

## Known Issues

At the moment, there is an issue with the C/CUDA implementation not producing the same results as the Python implementation in the `LeNet5_Python_Notebook/`. This is believed to be due to precision mismatches when importing the weights and biases from the Python implementation into the C/CUDA version, causing discrepancies in the output. 

For example, a weight value in the Python implementation, such as `0.025248982`, becomes `0.025249` when loaded into the C/CUDA version. This precision mismatch likely affects the output, especially for tasks that require fine-grained computation.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/NightFore/LeNet5-GPU-Inference/blob/main/LICENSE) file for details.