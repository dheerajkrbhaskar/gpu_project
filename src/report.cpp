#include "report.h"

#include <iomanip>
#include <iostream>
#include <sstream>

using namespace std;

namespace crack {
namespace {

void printDivider()
{
     cout << "------------------------------------------------------------\n";
}

 string formatValue(double value)
{
    if (value < 0.0) {
        return "-";
    }

     ostringstream stream;
    stream <<  fixed <<  setprecision(2) << value;
    return stream.str();
}

 string formatSpeedup(double value)
{
    if (value < 0.0) {
        return "-";
    }

     ostringstream stream;
    stream <<  fixed <<  setprecision(2) << value << 'x';
    return stream.str();
}

} // namespace

 string modeDisplayLabel(Mode mode)
{
    switch (mode) {
    case Mode::Cpu:
        return "CPU";
    case Mode::Omp:
        return "OpenMP";
    case Mode::Gpu:
        return "GPU";
    }

    return "CPU";
}

DetectionSummary makeDetectionSummary(const Image& magnitude)
{
    DetectionSummary summary{
        false,
        largestComponentAreaPercent(magnitude, 20),
        20,
        0.20,
    };
    summary.present = summary.percent >= summary.thresholdPercent;
    return summary;
}

ModeReportRow makeModeReportRow(const  string& label, const CpuTiming& timing, double speedup)
{
    return ModeReportRow{label, timing.grayscaleMs, timing.sobelMs, timing.thresholdMs, timing.totalMs, speedup};
}

ModeReportRow makeModeReportRow(const  string& label, const GpuTiming& timing, double speedup)
{
    return ModeReportRow{label, timing.grayscaleMs, timing.sobelMs, timing.thresholdMs, timing.totalMs, speedup};
}

SavedArtifactPaths saveArtifacts(const ProcessedImages& images,
                                 const  filesystem::path& inputPath,
                                 const  string& label,
                                 const  string& timestamp)
{
    const  string stem = makeOutputStem(inputPath, label, timestamp);
    const  filesystem::path outputDir =  filesystem::path("outputs");

    SavedArtifactPaths paths{
        outputDir / (stem + "_grayscale.png"),
        outputDir / (stem + "_magnitude.png"),
        outputDir / (stem + "_binary.png")
    };

    saveImagePng(paths.grayscale, images.grayscale, false);
    saveImagePng(paths.magnitude, images.magnitude, false);
    saveImagePng(paths.binary, images.binary, false);

    return paths;
}

void printReportHeader(const  filesystem::path& inputPath, const Image& inputImage, const AppConfig& config)
{
     cout << "============================================================\n";
     cout << "CRACK DETECTION REPORT\n";
     cout << "============================================================\n";
     cout << "Input Image   : " << inputPath.filename().string() << "\n";
     cout << "Resolution    : " << inputImage.width << " x " << inputImage.height << "\n";
     cout << "Mode          : " << modeDisplayLabel(config.mode) << "\n";
     cout << "Runs Averaged : " << config.repeat << "\n";
     cout << "Threshold     : " << static_cast<int>(config.threshold) << "\n";
    printDivider();
}

void printResultSection(const DetectionSummary& summary)
{
     cout << "RESULT        : " << (summary.present ? "DETECTED" : "NOT DETECTED") << "\n";
     cout << "Metric        : " << formatMilliseconds(summary.percent)
              << "% edge area (threshold: " << formatMilliseconds(summary.thresholdPercent) << "%)\n";
    printDivider();
}

void printPerformanceSummary(const  array<ModeReportRow, 3>& rows)
{
     cout << "PERFORMANCE SUMMARY\n";
    printDivider();
     cout << "Mode        Gray   Sobel   Thresh   Total    Speedup\n";
    printDivider();
    for (const auto& row : rows) {
         cout <<  left <<  setw(11) << row.label << ' '
                  <<  right <<  setw(6) << formatValue(row.grayMs) << ' '
                  <<  setw(7) << formatValue(row.sobelMs) << ' '
                  <<  setw(8) << formatValue(row.thresholdMs) << ' '
                  <<  setw(8) << formatValue(row.totalMs) << ' '
                  <<  setw(8) << formatSpeedup(row.speedup) << '\n';
    }
    printDivider();
}

void printOutputFiles(const SavedArtifactPaths& paths)
{
     cout << "OUTPUT FILES\n";
    printDivider();
     cout << "Grayscale : " << paths.grayscale.string() << "\n";
     cout << "Magnitude : " << paths.magnitude.string() << "\n";
     cout << "Binary    : " << paths.binary.string() << "\n";
    printDivider();
}

} // namespace crack