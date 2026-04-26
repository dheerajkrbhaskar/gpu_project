#include "gpu_pipeline.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

namespace crack {

namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;
constexpr float kSobelMagnitudeScale = 255.0f / (4.0f * 255.0f * 1.41421356237f);

__device__ __forceinline__ int clampIndex(int value, int lower, int upper)
{
    return value < lower ? lower : (value > upper ? upper : value);
}

struct DeviceBuffer {
    unsigned char* ptr = nullptr;

    ~DeviceBuffer()
    {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};

struct EventHandle {
    cudaEvent_t handle = nullptr;

    ~EventHandle()
    {
        if (handle != nullptr) {
            cudaEventDestroy(handle);
        }
    }
};

inline void cudaCheck(cudaError_t error, const char* message)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(error));
    }
}

__global__ void grayscaleKernel(const unsigned char* inputBgr, unsigned char* gray, int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int pixelIndex = y * width + x;
    const int base = pixelIndex * 3;
    const unsigned char blue = inputBgr[base + 0];
    const unsigned char green = inputBgr[base + 1];
    const unsigned char red = inputBgr[base + 2];
    const float grayValue = 0.114f * static_cast<float>(blue) + 0.587f * static_cast<float>(green) + 0.299f * static_cast<float>(red);
    gray[pixelIndex] = static_cast<unsigned char>(grayValue + 0.5f);
}

__global__ void sobelKernel(const unsigned char* gray, unsigned char* magnitude, int width, int height)
{
    extern __shared__ unsigned char tile[];

    const int tileWidth = blockDim.x + 2;
    const int tileHeight = blockDim.y + 2;

    // Each thread cooperatively loads a 2D tile with a one-pixel halo.
    for (int localY = threadIdx.y; localY < tileHeight; localY += blockDim.y) {
        const int globalY = clampIndex(blockIdx.y * blockDim.y + localY - 1, 0, height - 1);
        for (int localX = threadIdx.x; localX < tileWidth; localX += blockDim.x) {
            const int globalX = clampIndex(blockIdx.x * blockDim.x + localX - 1, 0, width - 1);
            tile[localY * tileWidth + localX] = gray[globalY * width + globalX];
        }
    }

    __syncthreads();

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int pixelIndex = y * width + x;
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        magnitude[pixelIndex] = 0;
        return;
    }

    const int centerX = threadIdx.x + 1;
    const int centerY = threadIdx.y + 1;

    const unsigned char topLeft = tile[(centerY - 1) * tileWidth + (centerX - 1)];
    const unsigned char top = tile[(centerY - 1) * tileWidth + centerX];
    const unsigned char topRight = tile[(centerY - 1) * tileWidth + (centerX + 1)];
    const unsigned char left = tile[centerY * tileWidth + (centerX - 1)];
    const unsigned char right = tile[centerY * tileWidth + (centerX + 1)];
    const unsigned char bottomLeft = tile[(centerY + 1) * tileWidth + (centerX - 1)];
    const unsigned char bottom = tile[(centerY + 1) * tileWidth + centerX];
    const unsigned char bottomRight = tile[(centerY + 1) * tileWidth + (centerX + 1)];

    const int gx = -static_cast<int>(topLeft) + static_cast<int>(topRight)
        - 2 * static_cast<int>(left) + 2 * static_cast<int>(right)
        - static_cast<int>(bottomLeft) + static_cast<int>(bottomRight);

    const int gy = -static_cast<int>(topLeft) - 2 * static_cast<int>(top) - static_cast<int>(topRight)
        + static_cast<int>(bottomLeft) + 2 * static_cast<int>(bottom) + static_cast<int>(bottomRight);

    const float magnitudeValue = sqrtf(static_cast<float>(gx * gx + gy * gy));
    const float normalized = fminf(255.0f, magnitudeValue * kSobelMagnitudeScale);
    magnitude[pixelIndex] = static_cast<unsigned char>(normalized + 0.5f);
}

__global__ void thresholdKernel(const unsigned char* magnitude, unsigned char* binary, int width, int height, unsigned char threshold)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int pixelIndex = y * width + x;
    binary[pixelIndex] = magnitude[pixelIndex] >= threshold ? 255 : 0;
}

GpuResult runGpuPipeline(const Image& inputBgr, const GpuOptions& options)
{
    GpuResult result;

    int deviceCount = 0;
    const cudaError_t countStatus = cudaGetDeviceCount(&deviceCount);
    if (countStatus != cudaSuccess || deviceCount <= 0) {
        result.usedCpuFallback = true;
        result.note = "No CUDA device detected; using the CPU baseline as fallback";
        const CpuResult fallback = runCpuPipeline(inputBgr, CpuOptions{false, options.threshold});
        result.images = fallback.images;
        result.timing.totalMs = fallback.timing.totalMs;
        return result;
    }

    try {
        if (inputBgr.empty()) {
            throw std::runtime_error("Input image is empty");
        }

        const int width = inputBgr.width;
        const int height = inputBgr.height;
        const size_t colorBytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
        const size_t grayBytes = static_cast<size_t>(width) * static_cast<size_t>(height);

        DeviceBuffer dInput;
        DeviceBuffer dGray;
        DeviceBuffer dMagnitude;
        DeviceBuffer dBinary;
        EventHandle kernelStart;
        EventHandle kernelStop;

        cudaCheck(cudaEventCreate(&kernelStart.handle), "Failed to create CUDA start event");
        cudaCheck(cudaEventCreate(&kernelStop.handle), "Failed to create CUDA stop event");

        const auto totalStart = Clock::now();

        const auto h2dStart = Clock::now();
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dInput.ptr), colorBytes), "Failed to allocate input buffer");
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dGray.ptr), grayBytes), "Failed to allocate gray buffer");
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dMagnitude.ptr), grayBytes), "Failed to allocate magnitude buffer");
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dBinary.ptr), grayBytes), "Failed to allocate binary buffer");
        cudaCheck(cudaMemcpy(dInput.ptr, inputBgr.data.data(), colorBytes, cudaMemcpyHostToDevice), "Failed to copy input image to device");
        result.timing.h2dMs = elapsedMilliseconds(h2dStart, Clock::now());

        const dim3 block(kBlockX, kBlockY);
        const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
        const size_t sharedBytes = static_cast<size_t>(block.x + 2) * static_cast<size_t>(block.y + 2) * sizeof(unsigned char);

        cudaCheck(cudaEventRecord(kernelStart.handle), "Failed to record kernel start event");
        grayscaleKernel<<<grid, block>>>(dInput.ptr, dGray.ptr, width, height);
        sobelKernel<<<grid, block, sharedBytes>>>(dGray.ptr, dMagnitude.ptr, width, height);
        thresholdKernel<<<grid, block>>>(dMagnitude.ptr, dBinary.ptr, width, height, options.threshold);
        cudaCheck(cudaGetLastError(), "CUDA kernel launch failed");
        cudaCheck(cudaEventRecord(kernelStop.handle), "Failed to record kernel stop event");
        cudaCheck(cudaEventSynchronize(kernelStop.handle), "Failed to synchronize CUDA kernel events");

        float kernelMs = 0.0f;
        cudaCheck(cudaEventElapsedTime(&kernelMs, kernelStart.handle, kernelStop.handle), "Failed to measure CUDA kernel time");
        result.timing.kernelMs = static_cast<double>(kernelMs);

        const auto d2hStart = Clock::now();
        result.images.grayscale.resize(width, height, 1);
        result.images.magnitude.resize(width, height, 1);
        result.images.binary.resize(width, height, 1);
        cudaCheck(cudaMemcpy(result.images.grayscale.data.data(), dGray.ptr, grayBytes, cudaMemcpyDeviceToHost), "Failed to copy grayscale output to host");
        cudaCheck(cudaMemcpy(result.images.magnitude.data.data(), dMagnitude.ptr, grayBytes, cudaMemcpyDeviceToHost), "Failed to copy magnitude output to host");
        cudaCheck(cudaMemcpy(result.images.binary.data.data(), dBinary.ptr, grayBytes, cudaMemcpyDeviceToHost), "Failed to copy binary output to host");
        result.timing.d2hMs = elapsedMilliseconds(d2hStart, Clock::now());

        result.timing.totalMs = elapsedMilliseconds(totalStart, Clock::now());
        return result;
    } catch (const std::exception& error) {
        result.usedCpuFallback = true;
        result.note = std::string("CUDA execution failed; using CPU fallback: ") + error.what();
        const CpuResult fallback = runCpuPipeline(inputBgr, CpuOptions{false, options.threshold});
        result.images = fallback.images;
        result.timing.totalMs = fallback.timing.totalMs;
        return result;
    }
}

} // namespace crack
