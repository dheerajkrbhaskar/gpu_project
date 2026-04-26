#include "cpu_pipeline.h"

#include <cmath>

using namespace std;

namespace crack {

namespace {

constexpr float kSobelMagnitudeScale = 255.0f / (4.0f * 255.0f * 1.41421356237f);

} // namespace

CpuResult runCpuPipeline(const Image& inputBgr, const CpuOptions& options)
{
    if (inputBgr.empty()) {
        throw std::runtime_error("Input image is empty");
    }

    CpuResult result;
    const auto totalStart = Clock::now();

    const int width = inputBgr.width;
    const int height = inputBgr.height;

    result.images.grayscale.resize(width, height, 1);
    result.images.magnitude.resize(width, height, 1);
    result.images.binary.resize(width, height, 1);

    result.timing.grayscaleMs = measureMilliseconds([&] {
#ifdef _OPENMP
#pragma omp parallel for if(options.useOpenMP) schedule(static)
#endif
        for (int y = 0; y < height; ++y) {
            const unsigned char* srcRow = inputBgr.rowPtr(y);
            unsigned char* grayRow = result.images.grayscale.rowPtr(y);

            for (int x = 0; x < width; ++x) {
                const int base = x * 3;
                const float blue = static_cast<float>(srcRow[base + 0]);
                const float green = static_cast<float>(srcRow[base + 1]);
                const float red = static_cast<float>(srcRow[base + 2]);
                const float grayValue = 0.114f * blue + 0.587f * green + 0.299f * red;
                grayRow[x] = static_cast<unsigned char>(grayValue + 0.5f);
            }
        }
    });

    result.timing.sobelMs = measureMilliseconds([&] {
#ifdef _OPENMP
#pragma omp parallel for if(options.useOpenMP) schedule(static)
#endif
        for (int y = 1; y < height - 1; ++y) {
            const unsigned char* prevRow = result.images.grayscale.rowPtr(y - 1);
            const unsigned char* currRow = result.images.grayscale.rowPtr(y);
            const unsigned char* nextRow = result.images.grayscale.rowPtr(y + 1);
            unsigned char* magnitudeRow = result.images.magnitude.rowPtr(y);

            for (int x = 1; x < width - 1; ++x) {
                const int gx = -static_cast<int>(prevRow[x - 1]) + static_cast<int>(prevRow[x + 1])
                    - 2 * static_cast<int>(currRow[x - 1]) + 2 * static_cast<int>(currRow[x + 1])
                    - static_cast<int>(nextRow[x - 1]) + static_cast<int>(nextRow[x + 1]);

                const int gy = -static_cast<int>(prevRow[x - 1]) - 2 * static_cast<int>(prevRow[x]) - static_cast<int>(prevRow[x + 1])
                    + static_cast<int>(nextRow[x - 1]) + 2 * static_cast<int>(nextRow[x]) + static_cast<int>(nextRow[x + 1]);

                const float magnitude = std::sqrt(static_cast<float>(gx * gx + gy * gy));
                const float normalized = std::min(255.0f, magnitude * kSobelMagnitudeScale);
                magnitudeRow[x] = static_cast<unsigned char>(normalized + 0.5f);
            }
        }
    });

    result.timing.thresholdMs = measureMilliseconds([&] {
#ifdef _OPENMP
#pragma omp parallel for if(options.useOpenMP) schedule(static)
#endif
        for (int y = 0; y < height; ++y) {
            const unsigned char* magnitudeRow = result.images.magnitude.rowPtr(y);
            unsigned char* binaryRow = result.images.binary.rowPtr(y);

            for (int x = 0; x < width; ++x) {
                binaryRow[x] = magnitudeRow[x] >= options.threshold ? 255 : 0;
            }
        }
    });

    result.timing.totalMs = elapsedMilliseconds(totalStart, Clock::now());
    return result;
}

} // namespace crack
