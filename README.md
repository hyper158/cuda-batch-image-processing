# CUDA Batch Image Processing using CIFAR-10

## Project Title

CUDA Batch Image Processing using CIFAR-10

## Project Overview

This project demonstrates GPU-based batch image processing using NVIDIA CUDA.

The program processes 500 CIFAR-10 RGB images of size 32x32. Each image is processed on the GPU using a CUDA kernel.

The CUDA kernel converts RGB pixels to grayscale and then applies a small brightness adjustment.

The main purpose of the project is to demonstrate parallel image processing on a GPU for a batch of images.

## Dataset

The project uses the CIFAR-10 dataset.

- Dataset: CIFAR-10
- Images processed: 500
- Image dimensions: 32x32 pixels
- Image type: RGB
- Total pixels processed: 512,000

CIFAR-10 images were converted to PPM format before being processed by the CUDA program.

## CUDA Implementation

Each GPU thread processes one RGB pixel.

The grayscale value is calculated using:

Gray = 0.299R + 0.587G + 0.114B

After grayscale conversion, a brightness value of 20 is added.

Values greater than 255 are limited to 255.

The program uses:

- CUDA kernel
- GPU global memory
- 256 threads per CUDA block
- 2,000 CUDA blocks for 500 images
- CPU-to-GPU memory transfer
- GPU-to-CPU memory transfer

## Command-Line Arguments

The program accepts three command-line arguments:

1. Input directory
2. Output directory
3. Number of images

Example:

./batch_image_processing input_ppm output_ppm 500

This makes the program configurable instead of using fixed input and output paths.

## Project Structure

cuda-batch-image-processing/

    batch_image_processing.cu
    Makefile
    README.md
    .gitignore
    results/
        execution_results.txt
        comparison_1.png
        comparison_100.png
        comparison_250.png
        comparison_500.png

The input and generated output image folders are excluded from the Git repository because they contain large batches of generated data.

## Building the Project

The project includes a Makefile for compilation.

Run:

make

This compiles the CUDA source using nvcc with C++17 support.

To remove the compiled executable:

make clean

## Running the Project

After building the program, run:

./batch_image_processing input_ppm output_ppm 500

The input directory should contain PPM images named:

image_1.ppm
image_2.ppm
...
image_500.ppm

The processed images are saved in the output directory.

## Execution Environment

The project was executed using:

- GPU: NVIDIA Tesla T4
- CUDA Capability: 7.5
- CUDA Toolkit: 12.8
- Dataset: CIFAR-10
- Image count: 500

## Execution Results

The program successfully processed all 500 images.

- Images requested: 500
- Images loaded: 500
- Images processed: 500
- Images saved: 500
- Total pixels: 512,000
- CUDA blocks: 2,000
- Threads per block: 256
- CPU to GPU transfer: 0.360 ms
- GPU kernel execution: 0.224 ms

The CUDA kernel completed successfully and the processed images were generated.

## Output

The processing operation is:

RGB -> Grayscale -> Brightness Adjustment

Visual comparisons of original and GPU-processed images are included in the results directory.

The execution log is also included in the results directory as evidence of successful GPU execution.

## Learning and Development

This project helped me understand how CUDA can be used to process large amounts of image data in parallel.

The main learning was how GPU threads can divide pixel-level processing among many parallel threads.

I also learned how to allocate GPU memory, transfer data between CPU and GPU, launch CUDA kernels, synchronize GPU execution, and retrieve processed data.

One practical challenge was preparing the CIFAR-10 images in a format that could be read directly by the CUDA program.

The final implementation successfully processed 500 images using the NVIDIA Tesla T4 GPU.
