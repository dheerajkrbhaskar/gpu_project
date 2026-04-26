#pragma once

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "timer.h"

#ifndef GPU_CRACK_PYTHON_EXECUTABLE
#define GPU_CRACK_PYTHON_EXECUTABLE "python3"
#endif

namespace crack {

struct Image {
    int width = 0;
    int height = 0;
    int channels = 0;
    std::vector<unsigned char> data;

    bool empty() const;

    size_t pixelCount() const;

    void resize(int newWidth, int newHeight, int newChannels);

    unsigned char* rowPtr(int y);

    const unsigned char* rowPtr(int y) const;
};

enum class Mode {
    Cpu,
    Omp,
    Gpu
};

struct ProcessedImages {
    Image grayscale;
    Image magnitude;
    Image binary;
};

struct AppConfig {
    std::string inputPath;
    Mode mode = Mode::Cpu;
    int repeat = 3;
    unsigned char threshold = 60;
    bool showHelp = false;
};

std::string modeLabel(Mode mode);

Mode parseMode(const std::string& value);

std::string shellQuote(const std::string& value);

std::string runCommandCapture(const std::string& command);

AppConfig parseArguments(int argc, char** argv);

void printUsage(std::ostream& stream, const char* programName);

Image loadImageAsBgr(const std::string& path);

void ensureOutputDirectory();

std::string sanitizeStem(const std::filesystem::path& path);

std::string makeOutputStem(const std::filesystem::path& inputPath, const std::string& label, const std::string& timestamp);

void saveImagePgm(const std::filesystem::path& path, const Image& image);

void saveImagePng(const std::filesystem::path& path, const Image& image, bool treatInputAsBgr);

double binaryMismatchPercent(const Image& lhs, const Image& rhs);

double meanAbsoluteDifference(const Image& lhs, const Image& rhs);

size_t largestConnectedComponentArea(const Image& binary, unsigned char edgeCutoff);

double largestComponentAreaPercent(const Image& magnitude, unsigned char edgeCutoff = 20);

} // namespace crack