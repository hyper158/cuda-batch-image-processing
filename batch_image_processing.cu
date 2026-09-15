
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#define IMAGE_WIDTH 32
#define IMAGE_HEIGHT 32
#define CHANNELS 3
#define IMAGE_SIZE (IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS)

#define THREADS_PER_BLOCK 256

// ------------------------------------------------------------
// CUDA Kernel
// Each GPU thread processes one RGB pixel.
// ------------------------------------------------------------
__global__ void processImages(
    const unsigned char* input,
    unsigned char* output,
    int totalPixels)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < totalPixels)
    {
        int pixelIndex = index * 3;

        unsigned char r = input[pixelIndex];
        unsigned char g = input[pixelIndex + 1];
        unsigned char b = input[pixelIndex + 2];

        // Convert RGB pixel to grayscale.
        int gray = static_cast<int>(
            0.299f * r +
            0.587f * g +
            0.114f * b
        );

        // Increase brightness slightly.
        gray += 20;

        if (gray > 255)
        {
            gray = 255;
        }

        output[pixelIndex] = static_cast<unsigned char>(gray);
        output[pixelIndex + 1] = static_cast<unsigned char>(gray);
        output[pixelIndex + 2] = static_cast<unsigned char>(gray);
    }
}

// ------------------------------------------------------------
// Read a 32x32 RGB PPM image.
// ------------------------------------------------------------
bool readPPM(
    const std::string& filename,
    std::vector<unsigned char>& data)
{
    std::ifstream file(filename, std::ios::binary);

    if (!file)
    {
        return false;
    }

    std::string format;
    int width;
    int height;
    int maxValue;

    file >> format;

    if (format != "P6")
    {
        return false;
    }

    file >> width >> height >> maxValue;
    file.get();

    if (width != IMAGE_WIDTH ||
        height != IMAGE_HEIGHT ||
        maxValue != 255)
    {
        return false;
    }

    data.resize(IMAGE_SIZE);

    file.read(
        reinterpret_cast<char*>(data.data()),
        IMAGE_SIZE
    );

    return file.good() || file.eof();
}

// ------------------------------------------------------------
// Write an RGB image as PPM.
// ------------------------------------------------------------
bool writePPM(
    const std::string& filename,
    const std::vector<unsigned char>& data)
{
    std::ofstream file(filename, std::ios::binary);

    if (!file)
    {
        return false;
    }

    file << "P6\n";
    file << IMAGE_WIDTH << " " << IMAGE_HEIGHT << "\n";
    file << "255\n";

    file.write(
        reinterpret_cast<const char*>(data.data()),
        data.size()
    );

    return true;
}

// ------------------------------------------------------------
// Main
// Usage:
// ./batch_image_processing <input_directory>
//                            <output_directory>
//                            <number_of_images>
// ------------------------------------------------------------
int main(int argc, char* argv[])
{
    std::cout
        << "==================================================\n";
    std::cout
        << "     CUDA BATCH IMAGE PROCESSING PROJECT\n";
    std::cout
        << "==================================================\n\n";

    // --------------------------------------------------------
    // Check command-line arguments
    // --------------------------------------------------------
    if (argc != 4)
    {
        std::cerr
            << "Usage: " << argv[0]
            << " <input_directory>"
            << " <output_directory>"
            << " <number_of_images>\n\n";

        std::cerr
            << "Example:\n";

        std::cerr
            << argv[0]
            << " input_ppm output_ppm 500\n";

        return 1;
    }

    const std::string inputDirectory = argv[1];
    const std::string outputDirectory = argv[2];

    int numImages = 0;

    try
    {
        numImages = std::stoi(argv[3]);
    }
    catch (...)
    {
        std::cerr
            << "Error: number_of_images must be an integer.\n";

        return 1;
    }

    if (numImages <= 0)
    {
        std::cerr
            << "Error: number_of_images must be greater than 0.\n";

        return 1;
    }

    // --------------------------------------------------------
    // Detect GPU
    // --------------------------------------------------------
    int deviceCount = 0;

    cudaError_t error =
        cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess || deviceCount == 0)
    {
        std::cerr
            << "CUDA GPU not available.\n";

        return 1;
    }

    cudaDeviceProp deviceProperties;

    cudaGetDeviceProperties(
        &deviceProperties,
        0
    );

    std::cout
        << "GPU Device       : "
        << deviceProperties.name << "\n";

    std::cout
        << "CUDA Capability  : "
        << deviceProperties.major
        << "."
        << deviceProperties.minor
        << "\n";

    std::cout
        << "Dataset          : CIFAR-10\n";

    std::cout
        << "Images Requested : "
        << numImages << "\n";

    std::cout
        << "Image Size       : "
        << IMAGE_WIDTH
        << " x "
        << IMAGE_HEIGHT << "\n";

    std::cout
        << "Image Type       : RGB\n";

    std::cout
        << "CUDA Block Size  : "
        << THREADS_PER_BLOCK << "\n\n";

    // --------------------------------------------------------
    // Create output directory
    // --------------------------------------------------------
    fs::create_directories(outputDirectory);

    // --------------------------------------------------------
    // Prepare host memory
    // --------------------------------------------------------
    const int totalPixels =
        numImages *
        IMAGE_WIDTH *
        IMAGE_HEIGHT;

    const size_t totalBytes =
        static_cast<size_t>(numImages) *
        IMAGE_SIZE;

    std::vector<unsigned char> hostInput(totalBytes);
    std::vector<unsigned char> hostOutput(totalBytes);

    std::cout
        << "Loading input images...\n";

    int loadedImages = 0;

    for (int i = 0; i < numImages; i++)
    {
        std::string filename =
            inputDirectory +
            "/image_" +
            std::to_string(i + 1) +
            ".ppm";

        std::vector<unsigned char> imageData;

        if (!readPPM(filename, imageData))
        {
            std::cerr
                << "Error reading: "
                << filename << "\n";

            return 1;
        }

        std::copy(
            imageData.begin(),
            imageData.end(),
            hostInput.begin() +
            static_cast<size_t>(i) * IMAGE_SIZE
        );

        loadedImages++;
    }

    std::cout
        << "Images loaded successfully: "
        << loadedImages << "\n\n";

    // --------------------------------------------------------
    // Allocate GPU memory
    // --------------------------------------------------------
    unsigned char* deviceInput = nullptr;
    unsigned char* deviceOutput = nullptr;

    error = cudaMalloc(
        &deviceInput,
        totalBytes
    );

    if (error != cudaSuccess)
    {
        std::cerr
            << "GPU input memory allocation failed: "
            << cudaGetErrorString(error) << "\n";

        return 1;
    }

    error = cudaMalloc(
        &deviceOutput,
        totalBytes
    );

    if (error != cudaSuccess)
    {
        std::cerr
            << "GPU output memory allocation failed: "
            << cudaGetErrorString(error) << "\n";

        cudaFree(deviceInput);

        return 1;
    }

    // --------------------------------------------------------
    // Copy data CPU -> GPU
    // --------------------------------------------------------
    auto transferStart =
        std::chrono::high_resolution_clock::now();

    error = cudaMemcpy(
        deviceInput,
        hostInput.data(),
        totalBytes,
        cudaMemcpyHostToDevice
    );

    if (error != cudaSuccess)
    {
        std::cerr
            << "CPU to GPU copy failed: "
            << cudaGetErrorString(error) << "\n";

        cudaFree(deviceInput);
        cudaFree(deviceOutput);

        return 1;
    }

    auto transferEnd =
        std::chrono::high_resolution_clock::now();

    double transferTime =
        std::chrono::duration<double, std::milli>(
            transferEnd - transferStart
        ).count();

    // --------------------------------------------------------
    // CUDA execution
    // --------------------------------------------------------
    int blocks =
        (totalPixels +
         THREADS_PER_BLOCK - 1)
        / THREADS_PER_BLOCK;

    std::cout
        << "Launching CUDA kernel...\n";

    std::cout
        << "Total pixels     : "
        << totalPixels << "\n";

    std::cout
        << "Blocks launched  : "
        << blocks << "\n";

    std::cout
        << "Threads/block    : "
        << THREADS_PER_BLOCK << "\n";

    auto gpuStart =
        std::chrono::high_resolution_clock::now();

    processImages<<<blocks, THREADS_PER_BLOCK>>>(
        deviceInput,
        deviceOutput,
        totalPixels
    );

    error = cudaGetLastError();

    if (error != cudaSuccess)
    {
        std::cerr
            << "Kernel launch failed: "
            << cudaGetErrorString(error)
            << "\n";

        cudaFree(deviceInput);
        cudaFree(deviceOutput);

        return 1;
    }

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess)
    {
        std::cerr
            << "CUDA kernel execution failed: "
            << cudaGetErrorString(error)
            << "\n";

        cudaFree(deviceInput);
        cudaFree(deviceOutput);

        return 1;
    }

    auto gpuEnd =
        std::chrono::high_resolution_clock::now();

    double gpuTime =
        std::chrono::duration<double, std::milli>(
            gpuEnd - gpuStart
        ).count();

    std::cout
        << "CUDA kernel completed successfully.\n";

    // --------------------------------------------------------
    // Copy processed data GPU -> CPU
    // --------------------------------------------------------
    error = cudaMemcpy(
        hostOutput.data(),
        deviceOutput,
        totalBytes,
        cudaMemcpyDeviceToHost
    );

    if (error != cudaSuccess)
    {
        std::cerr
            << "GPU to CPU copy failed: "
            << cudaGetErrorString(error) << "\n";

        cudaFree(deviceInput);
        cudaFree(deviceOutput);

        return 1;
    }

    // --------------------------------------------------------
    // Save output images
    // --------------------------------------------------------
    std::cout
        << "\nSaving processed images...\n";

    int savedImages = 0;

    for (int i = 0; i < numImages; i++)
    {
        std::vector<unsigned char> imageData(
            IMAGE_SIZE
        );

        std::copy(
            hostOutput.begin() +
            static_cast<size_t>(i) * IMAGE_SIZE,

            hostOutput.begin() +
            static_cast<size_t>(i + 1) * IMAGE_SIZE,

            imageData.begin()
        );

        std::string filename =
            outputDirectory +
            "/processed_" +
            std::to_string(i + 1) +
            ".ppm";

        if (writePPM(filename, imageData))
        {
            savedImages++;
        }
    }

    // --------------------------------------------------------
    // Free GPU memory
    // --------------------------------------------------------
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    // --------------------------------------------------------
    // Display results
    // --------------------------------------------------------
    std::cout
        << "\n==================================================\n";

    std::cout
        << "              PROCESSING RESULTS\n";

    std::cout
        << "==================================================\n";

    std::cout
        << std::fixed
        << std::setprecision(3);

    std::cout
        << "Images processed      : "
        << loadedImages << "\n";

    std::cout
        << "Images saved           : "
        << savedImages << "\n";

    std::cout
        << "CPU -> GPU transfer   : "
        << transferTime << " ms\n";

    std::cout
        << "GPU kernel execution  : "
        << gpuTime << " ms\n";

    std::cout
        << "CUDA blocks           : "
        << blocks << "\n";

    std::cout
        << "CUDA threads/block    : "
        << THREADS_PER_BLOCK << "\n";

    std::cout
        << "\nImage processing operation:\n";

    std::cout
        << "RGB -> Grayscale -> Brightness Adjustment\n";

    std::cout
        << "\nGPU processing completed successfully.\n";

    std::cout
        << "CUDA memory released successfully.\n";

    std::cout
        << "\n==================================================\n";

    std::cout
        << "       CUDA IMAGE PROCESSING COMPLETED\n";

    std::cout
        << "==================================================\n";

    return 0;
}
