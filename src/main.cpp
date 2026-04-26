#include "gpu_pipeline.h"

#include <array>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

namespace crack {
namespace {

struct BenchSummary {
    std::string label;
    ProcessedImages images;
    CpuTiming cpuTiming;
    GpuTiming gpuTiming;
    bool usedCpuFallback = false;
    std::string note;
};

struct CpuBenchResult {
    BenchSummary summary;
    double averageTotalMs = 0.0;
};

struct GpuBenchResult {
    BenchSummary summary;
    double averageTotalMs = 0.0;
};

struct SavedArtifactPaths {
    std::filesystem::path input;
    std::filesystem::path grayscale;
    std::filesystem::path magnitude;
    std::filesystem::path binary;
};

struct DetectionSummary {
    bool present = false;
    double percent = 0.0;
    unsigned char edgeCutoff = 20;
    double thresholdPercent = 0.20;
};

void printRunBanner(const AppConfig& config, const std::filesystem::path& inputPath)
{
    std::cout << "\n============================================================\n";
    std::cout << "GPU-Accelerated Crack Detection\n";
    std::cout << "  Input     : " << inputPath.string() << "\n";
    std::cout << "  Mode      : " << modeLabel(config.mode) << "\n";
    std::cout << "  Repeat    : " << config.repeat << "\n";
    std::cout << "  Threshold : " << static_cast<int>(config.threshold) << "\n";
    std::cout << "============================================================\n";
}

void printDivider()
{
    std::cout << "+----------------------+------------+------------+---------------+---------+-----------+---------+-----------+----------+\n";
}

void printTableHeader()
{
    printDivider();
    std::cout << "| Label                | Gray ms    | Sobel ms   | Threshold ms  | H2D ms  | Kernel ms | D2H ms  | Total ms  | Speedup  |\n";
    printDivider();
}

std::string formatCell(double value)
{
    if (value < 0.0) {
        return "-";
    }

    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << value;
    return stream.str();
}

std::string formatSpeedup(double value)
{
    if (value < 0.0) {
        return "-";
    }

    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << value << 'x';
    return stream.str();
}

void printTableRow(const std::string& label,
                   double grayMs,
                   double sobelMs,
                   double thresholdMs,
                   double h2dMs,
                   double kernelMs,
                   double d2hMs,
                   double totalMs,
                   double speedup)
{
    std::cout << "| " << std::left << std::setw(20) << label << " | "
              << std::right << std::setw(10) << formatCell(grayMs) << " | "
              << std::setw(10) << formatCell(sobelMs) << " | "
              << std::setw(13) << formatCell(thresholdMs) << " | "
              << std::setw(7) << formatCell(h2dMs) << " | "
              << std::setw(9) << formatCell(kernelMs) << " | "
              << std::setw(7) << formatCell(d2hMs) << " | "
              << std::setw(9) << formatCell(totalMs) << " | "
              << std::setw(8) << formatSpeedup(speedup) << " |\n";
}

double computeSpeedup(double baselineMs, double candidateMs)
{
    if (baselineMs <= 0.0 || candidateMs <= 0.0) {
        return -1.0;
    }
    return baselineMs / candidateMs;
}

void printDetectionSummary(const DetectionSummary& summary)
{
    std::cout << "\nCrack decision : " << (summary.present ? "DETECTED" : "NOT DETECTED") << "\n";
    std::cout << "Decision metric: largest connected edge area = "
              << formatMilliseconds(summary.percent) << "%\n";
    std::cout << "Rule           : area >= " << formatMilliseconds(summary.thresholdPercent)
              << "% using edge cutoff " << static_cast<int>(summary.edgeCutoff) << "\n";
}

template <typename Timing>
void accumulateCpuTiming(CpuTiming& total, const Timing& sample)
{
    total.grayscaleMs += sample.grayscaleMs;
    total.sobelMs += sample.sobelMs;
    total.thresholdMs += sample.thresholdMs;
    total.totalMs += sample.totalMs;
}

void accumulateGpuTiming(GpuTiming& total, const GpuTiming& sample)
{
    total.h2dMs += sample.h2dMs;
    total.kernelMs += sample.kernelMs;
    total.d2hMs += sample.d2hMs;
    total.totalMs += sample.totalMs;
}

CpuBenchResult benchmarkCpu(const Image& input, bool useOpenMP, int repeat, unsigned char threshold)
{
    CpuTiming totalTiming;
    CpuResult lastResult;

    for (int index = 0; index < repeat; ++index) {
        lastResult = runCpuPipeline(input, CpuOptions{useOpenMP, threshold});
        accumulateCpuTiming(totalTiming, lastResult.timing);
    }

    lastResult.timing.grayscaleMs = totalTiming.grayscaleMs / repeat;
    lastResult.timing.sobelMs = totalTiming.sobelMs / repeat;
    lastResult.timing.thresholdMs = totalTiming.thresholdMs / repeat;
    lastResult.timing.totalMs = totalTiming.totalMs / repeat;

    CpuBenchResult output;
    output.summary.label = useOpenMP ? "CPU (OpenMP)" : "CPU baseline";
    output.summary.images = std::move(lastResult.images);
    output.summary.cpuTiming = lastResult.timing;
    output.averageTotalMs = lastResult.timing.totalMs;
    return output;
}

GpuBenchResult benchmarkGpu(const Image& input, int repeat, unsigned char threshold)
{
    GpuTiming totalTiming;
    GpuResult lastResult;

    for (int index = 0; index < repeat; ++index) {
        lastResult = runGpuPipeline(input, GpuOptions{threshold});
        accumulateGpuTiming(totalTiming, lastResult.timing);
    }

    lastResult.timing.h2dMs = totalTiming.h2dMs / repeat;
    lastResult.timing.kernelMs = totalTiming.kernelMs / repeat;
    lastResult.timing.d2hMs = totalTiming.d2hMs / repeat;
    lastResult.timing.totalMs = totalTiming.totalMs / repeat;

    GpuBenchResult output;
    output.summary.label = lastResult.usedCpuFallback ? "GPU (fallback)" : "GPU";
    output.summary.images = std::move(lastResult.images);
    output.summary.gpuTiming = lastResult.timing;
    output.summary.usedCpuFallback = lastResult.usedCpuFallback;
    output.summary.note = lastResult.note;
    output.averageTotalMs = lastResult.timing.totalMs;
    return output;
}

SavedArtifactPaths saveArtifacts(const ProcessedImages& images,
                                const Image& inputImage,
                                const std::filesystem::path& inputPath,
                                const std::string& label,
                                const std::string& timestamp)
{
    const std::string stem = makeOutputStem(inputPath, label, timestamp);
    const std::filesystem::path outputDir = std::filesystem::path("outputs");

    SavedArtifactPaths paths{
        outputDir / (stem + "_input.png"),
        outputDir / (stem + "_grayscale.png"),
        outputDir / (stem + "_magnitude.png"),
        outputDir / (stem + "_binary.png")
    };

    struct ArtifactSpec {
        const std::filesystem::path* path;
        const Image* image;
        bool asBgr;
    };

    const std::array<ArtifactSpec, 4> artifacts{
        ArtifactSpec{&paths.input, &inputImage, true},
        ArtifactSpec{&paths.grayscale, &images.grayscale, false},
        ArtifactSpec{&paths.magnitude, &images.magnitude, false},
        ArtifactSpec{&paths.binary, &images.binary, false},
    };

    for (const auto& artifact : artifacts) {
        saveImagePng(*artifact.path, *artifact.image, artifact.asBgr);
    }

    return paths;
}

void printSavedPaths(const std::string& label, const SavedArtifactPaths& paths)
{
    std::cout << "\nSaved images [" << label << "]:\n";
    const std::array<std::pair<const char*, const std::filesystem::path*>, 4> items{
        std::make_pair("input", &paths.input),
        std::make_pair("grayscale", &paths.grayscale),
        std::make_pair("magnitude", &paths.magnitude),
        std::make_pair("binary", &paths.binary),
    };

    for (const auto& item : items) {
        std::cout << "  " << std::left << std::setw(10) << item.first
                  << " " << item.second->string() << "\n";
    }
}

} // namespace
} // namespace crack

int main(int argc, char** argv)
{
    using namespace crack;

    AppConfig config;
    try {
        config = parseArguments(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n\n";
        printUsage(std::cerr, argv[0]);
        return 1;
    }

    if (config.showHelp) {
        printUsage(std::cout, argv[0]);
        return 0;
    }

    if (config.inputPath.empty()) {
        std::cerr << "Missing required --input argument\n\n";
        printUsage(std::cerr, argv[0]);
        return 1;
    }

    try {
        ensureOutputDirectory();

        const std::filesystem::path inputPath(config.inputPath);
        const Image inputImage = loadImageAsBgr(config.inputPath);
        const std::string timestamp = timestampString();

        printRunBanner(config, inputPath);

        std::cout << "Averaging timings over " << config.repeat << " runs\n";

        const CpuBenchResult cpuBaseline = benchmarkCpu(inputImage, false, config.repeat, config.threshold);
        CpuBenchResult selectedCpu = cpuBaseline;
        GpuBenchResult selectedGpu;
        bool selectedIsGpu = false;

        if (config.mode == Mode::Cpu) {
            selectedCpu = cpuBaseline;
        } else if (config.mode == Mode::Omp) {
            selectedCpu = benchmarkCpu(inputImage, true, config.repeat, config.threshold);
        } else {
            selectedGpu = benchmarkGpu(inputImage, config.repeat, config.threshold);
            selectedIsGpu = true;
        }

        printTableHeader();
        printTableRow(cpuBaseline.summary.label,
            cpuBaseline.summary.cpuTiming.grayscaleMs,
            cpuBaseline.summary.cpuTiming.sobelMs,
            cpuBaseline.summary.cpuTiming.thresholdMs,
            -1.0,
            -1.0,
            -1.0,
            cpuBaseline.summary.cpuTiming.totalMs,
            1.0);

        if (config.mode == Mode::Omp) {
            const double speedup = computeSpeedup(cpuBaseline.averageTotalMs, selectedCpu.averageTotalMs);
            printTableRow(selectedCpu.summary.label,
                selectedCpu.summary.cpuTiming.grayscaleMs,
                selectedCpu.summary.cpuTiming.sobelMs,
                selectedCpu.summary.cpuTiming.thresholdMs,
                -1.0,
                -1.0,
                -1.0,
                selectedCpu.summary.cpuTiming.totalMs,
                speedup);
        } else if (selectedIsGpu) {
            const double speedup = computeSpeedup(cpuBaseline.averageTotalMs, selectedGpu.averageTotalMs);
            const double h2dMs = selectedGpu.summary.usedCpuFallback ? -1.0 : selectedGpu.summary.gpuTiming.h2dMs;
            const double kernelMs = selectedGpu.summary.usedCpuFallback ? -1.0 : selectedGpu.summary.gpuTiming.kernelMs;
            const double d2hMs = selectedGpu.summary.usedCpuFallback ? -1.0 : selectedGpu.summary.gpuTiming.d2hMs;
            printTableRow(selectedGpu.summary.label,
                -1.0,
                -1.0,
                -1.0,
                h2dMs,
                kernelMs,
                d2hMs,
                selectedGpu.summary.gpuTiming.totalMs,
                speedup);
            if (selectedGpu.summary.usedCpuFallback && !selectedGpu.summary.note.empty()) {
                std::cout << "\n" << selectedGpu.summary.note << "\n";
            }
        }

        if (config.mode != Mode::Cpu) {
            printDivider();
        }

        const SavedArtifactPaths baselinePaths = saveArtifacts(cpuBaseline.summary.images, inputImage, inputPath, "cpu_baseline", timestamp);
        printSavedPaths("cpu_baseline", baselinePaths);
        if (config.mode == Mode::Omp) {
            const SavedArtifactPaths ompPaths = saveArtifacts(selectedCpu.summary.images, inputImage, inputPath, "omp", timestamp);
            printSavedPaths("omp", ompPaths);
        } else if (selectedIsGpu) {
            const std::string label = selectedGpu.summary.usedCpuFallback ? "gpu_fallback" : "gpu";
            const SavedArtifactPaths gpuPaths = saveArtifacts(selectedGpu.summary.images, inputImage, inputPath, label, timestamp);
            printSavedPaths(label, gpuPaths);
        }

        if (config.mode == Mode::Omp) {
            const double mismatch = binaryMismatchPercent(cpuBaseline.summary.images.binary, selectedCpu.summary.images.binary);
            const double magnitudeDiff = meanAbsoluteDifference(cpuBaseline.summary.images.magnitude, selectedCpu.summary.images.magnitude);
            std::cout << "\nBinary mismatch vs CPU baseline: " << formatMilliseconds(mismatch) << "%\n";
            std::cout << "Mean absolute magnitude difference: " << formatMilliseconds(magnitudeDiff) << "\n";
        } else if (selectedIsGpu) {
            const double mismatch = binaryMismatchPercent(cpuBaseline.summary.images.binary, selectedGpu.summary.images.binary);
            const double magnitudeDiff = meanAbsoluteDifference(cpuBaseline.summary.images.magnitude, selectedGpu.summary.images.magnitude);
            std::cout << "\nBinary mismatch vs CPU baseline: " << formatMilliseconds(mismatch) << "%\n";
            std::cout << "Mean absolute magnitude difference: " << formatMilliseconds(magnitudeDiff) << "\n";
        }

        const Image& selectedMagnitude = selectedIsGpu ? selectedGpu.summary.images.magnitude : selectedCpu.summary.images.magnitude;
        DetectionSummary detection{
            false,
            largestComponentAreaPercent(selectedMagnitude, 20),
            20,
            0.20,
        };
        detection.present = detection.percent >= detection.thresholdPercent;
        printDetectionSummary(detection);

        std::cout << "\nSaved outputs under outputs/ with timestamp " << timestamp << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
