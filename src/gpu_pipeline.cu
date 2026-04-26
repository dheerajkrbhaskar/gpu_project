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
        if (ptr != nullptr) cudaFree(ptr);
    }
};

struct EventHandle {
    cudaEvent_t handle = nullptr;
    ~EventHandle()
    {
        if (handle != nullptr) cudaEventDestroy(handle);
    }
};

inline void applyCpuFallback(GpuResult& result,
                             const Image& inputBgr,
                             unsigned char threshold,
                             const std::string& reason)
{
    const CpuResult fallback = runCpuPipeline(inputBgr, CpuOptions{false, threshold});
    result.images = fallback.images;
    result.timing.h2dMs = -1.0;
    result.timing.kernelMs = -1.0;
    result.timing.d2hMs = -1.0;
    result.timing.totalMs = fallback.timing.totalMs;
    result.usedCpuFallback = true;
    result.note = reason;
}

inline void cudaCheck(cudaError_t error, const char* message)
{
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(message) + ": " + cudaGetErrorString(error));
    }
}

__global__ void grayscaleKernel(const unsigned char* inputBgr, unsigned char* gray, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    int base = idx * 3;

    float b = inputBgr[base + 0];
    float g = inputBgr[base + 1];
    float r = inputBgr[base + 2];

    gray[idx] = static_cast<unsigned char>(0.114f*b + 0.587f*g + 0.299f*r + 0.5f);
}

__global__ void sobelKernel(const unsigned char* gray, unsigned char* mag, int width, int height)
{
    extern __shared__ unsigned char tile[];

    int tileW = blockDim.x + 2;
    int tileH = blockDim.y + 2;

    for (int ly = threadIdx.y; ly < tileH; ly += blockDim.y) {
        int gy = clampIndex(blockIdx.y * blockDim.y + ly - 1, 0, height - 1);
        for (int lx = threadIdx.x; lx < tileW; lx += blockDim.x) {
            int gx = clampIndex(blockIdx.x * blockDim.x + lx - 1, 0, width - 1);
            tile[ly * tileW + lx] = gray[gy * width + gx];
        }
    }

    __syncthreads();

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    if (x == 0 || y == 0 || x == width-1 || y == height-1) {
        mag[idx] = 0;
        return;
    }

    int cx = threadIdx.x + 1;
    int cy = threadIdx.y + 1;

    int tl = tile[(cy-1)*tileW + (cx-1)];
    int tr = tile[(cy-1)*tileW + (cx+1)];
    int l  = tile[(cy)*tileW + (cx-1)];
    int r  = tile[(cy)*tileW + (cx+1)];
    int bl = tile[(cy+1)*tileW + (cx-1)];
    int br = tile[(cy+1)*tileW + (cx+1)];

    int t  = tile[(cy-1)*tileW + cx];
    int b  = tile[(cy+1)*tileW + cx];

    int gx = -tl + tr - 2*l + 2*r - bl + br;
    int gy = -tl - 2*t - tr + bl + 2*b + br;

    float m = sqrtf(gx*gx + gy*gy);
    m = fminf(255.0f, m * kSobelMagnitudeScale);

    mag[idx] = static_cast<unsigned char>(m + 0.5f);
}

__global__ void thresholdKernel(const unsigned char* mag, unsigned char* bin, int width, int height, unsigned char th)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    bin[idx] = (mag[idx] >= th) ? 255 : 0;
}

} // anonymous namespace

// ✅ PUBLIC FUNCTION (must be outside anonymous namespace)
GpuResult runGpuPipeline(const Image& inputBgr, const GpuOptions& options)
{
    GpuResult result;

    if (inputBgr.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        applyCpuFallback(result,
                         inputBgr,
                         options.threshold,
                         "No CUDA device found; GPU mode used the CPU fallback.");
        return result;
    }

    try {
        const int w = inputBgr.width;
        const int h = inputBgr.height;

        const size_t colorBytes = static_cast<size_t>(w) * static_cast<size_t>(h) * 3;
        const size_t grayBytes = static_cast<size_t>(w) * static_cast<size_t>(h);

        DeviceBuffer dIn, dGray, dMag, dBin;

        cudaCheck(cudaMalloc(&dIn.ptr, colorBytes), "malloc input");
        cudaCheck(cudaMalloc(&dGray.ptr, grayBytes), "malloc gray");
        cudaCheck(cudaMalloc(&dMag.ptr, grayBytes), "malloc mag");
        cudaCheck(cudaMalloc(&dBin.ptr, grayBytes), "malloc bin");

        EventHandle start, afterH2d, afterKernel, end;
        cudaCheck(cudaEventCreate(&start.handle), "create start event");
        cudaCheck(cudaEventCreate(&afterH2d.handle), "create h2d event");
        cudaCheck(cudaEventCreate(&afterKernel.handle), "create kernel event");
        cudaCheck(cudaEventCreate(&end.handle), "create end event");

        cudaCheck(cudaEventRecord(start.handle), "record start event");
        cudaCheck(cudaMemcpy(dIn.ptr, inputBgr.data.data(), colorBytes, cudaMemcpyHostToDevice), "copy input to device");
        cudaCheck(cudaEventRecord(afterH2d.handle), "record h2d event");

        const dim3 block(kBlockX, kBlockY);
        const dim3 grid((w + kBlockX - 1) / kBlockX, (h + kBlockY - 1) / kBlockY);
        const size_t shared = static_cast<size_t>(kBlockX + 2) * static_cast<size_t>(kBlockY + 2) * sizeof(unsigned char);

        grayscaleKernel<<<grid, block>>>(dIn.ptr, dGray.ptr, w, h);
        cudaCheck(cudaGetLastError(), "launch grayscale kernel");

        sobelKernel<<<grid, block, shared>>>(dGray.ptr, dMag.ptr, w, h);
        cudaCheck(cudaGetLastError(), "launch sobel kernel");

        thresholdKernel<<<grid, block>>>(dMag.ptr, dBin.ptr, w, h, options.threshold);
        cudaCheck(cudaGetLastError(), "launch threshold kernel");
        cudaCheck(cudaEventRecord(afterKernel.handle), "record kernel event");

        result.images.grayscale.resize(w, h, 1);
        result.images.magnitude.resize(w, h, 1);
        result.images.binary.resize(w, h, 1);

        cudaCheck(cudaMemcpy(result.images.grayscale.data.data(), dGray.ptr, grayBytes, cudaMemcpyDeviceToHost), "copy gray to host");
        cudaCheck(cudaMemcpy(result.images.magnitude.data.data(), dMag.ptr, grayBytes, cudaMemcpyDeviceToHost), "copy magnitude to host");
        cudaCheck(cudaMemcpy(result.images.binary.data.data(), dBin.ptr, grayBytes, cudaMemcpyDeviceToHost), "copy binary to host");
        cudaCheck(cudaEventRecord(end.handle), "record end event");
        cudaCheck(cudaEventSynchronize(end.handle), "synchronize end event");

        float h2dMs = 0.0f;
        float kernelMs = 0.0f;
        float d2hMs = 0.0f;
        float totalMs = 0.0f;
        cudaCheck(cudaEventElapsedTime(&h2dMs, start.handle, afterH2d.handle), "elapsed h2d time");
        cudaCheck(cudaEventElapsedTime(&kernelMs, afterH2d.handle, afterKernel.handle), "elapsed kernel time");
        cudaCheck(cudaEventElapsedTime(&d2hMs, afterKernel.handle, end.handle), "elapsed d2h time");
        cudaCheck(cudaEventElapsedTime(&totalMs, start.handle, end.handle), "elapsed total time");

        result.timing.h2dMs = static_cast<double>(h2dMs);
        result.timing.kernelMs = static_cast<double>(kernelMs);
        result.timing.d2hMs = static_cast<double>(d2hMs);
        result.timing.totalMs = static_cast<double>(totalMs);
    } catch (const std::exception& e) {
        applyCpuFallback(result,
                         inputBgr,
                         options.threshold,
                         std::string("CUDA runtime failure; GPU mode used CPU fallback: ") + e.what());
    }

    return result;
}

} // namespace crack