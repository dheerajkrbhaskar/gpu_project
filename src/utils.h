#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
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

    bool empty() const
    {
        return width <= 0 || height <= 0 || channels <= 0 || data.empty();
    }

    size_t pixelCount() const
    {
        return static_cast<size_t>(width) * static_cast<size_t>(height);
    }

    void resize(int newWidth, int newHeight, int newChannels)
    {
        width = newWidth;
        height = newHeight;
        channels = newChannels;
        data.assign(static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(channels), 0);
    }

    unsigned char* rowPtr(int y)
    {
        return data.data() + static_cast<size_t>(y) * static_cast<size_t>(width) * static_cast<size_t>(channels);
    }

    const unsigned char* rowPtr(int y) const
    {
        return data.data() + static_cast<size_t>(y) * static_cast<size_t>(width) * static_cast<size_t>(channels);
    }
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

inline std::string modeLabel(Mode mode)
{
    switch (mode) {
    case Mode::Cpu:
        return "cpu";
    case Mode::Omp:
        return "omp";
    case Mode::Gpu:
        return "gpu";
    }
    return "cpu";
}

inline Mode parseMode(const std::string& value)
{
    std::string lowered = value;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    if (lowered == "cpu") {
        return Mode::Cpu;
    }
    if (lowered == "omp") {
        return Mode::Omp;
    }
    if (lowered == "gpu") {
        return Mode::Gpu;
    }

    throw std::invalid_argument("Unknown mode: " + value);
}

inline std::string shellQuote(const std::string& value)
{
    std::string quoted = "'";
    for (char ch : value) {
        if (ch == '\'') {
            quoted += "'\\''";
        } else {
            quoted.push_back(ch);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

inline std::string runCommandCapture(const std::string& command)
{
    std::array<char, 4096> buffer{};
    std::string output;
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        throw std::runtime_error("Failed to start helper process");
    }

    while (true) {
        const size_t bytesRead = std::fread(buffer.data(), 1, buffer.size(), pipe);
        if (bytesRead > 0) {
            output.append(buffer.data(), bytesRead);
        }
        if (bytesRead < buffer.size()) {
            if (std::feof(pipe)) {
                break;
            }
            if (std::ferror(pipe)) {
                pclose(pipe);
                throw std::runtime_error("Failed to read helper process output");
            }
        }
    }

    const int exitCode = pclose(pipe);
    if (exitCode != 0) {
        throw std::runtime_error("Helper process failed while reading image data");
    }

    return output;
}

inline AppConfig parseArguments(int argc, char** argv)
{
    AppConfig config;

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];

        if (argument == "--help" || argument == "-h") {
            config.showHelp = true;
            return config;
        }

        if (argument == "--input" && index + 1 < argc) {
            config.inputPath = argv[++index];
            continue;
        }

        if (argument == "--mode" && index + 1 < argc) {
            config.mode = parseMode(argv[++index]);
            continue;
        }

        if (argument == "--repeat" && index + 1 < argc) {
            config.repeat = std::max(1, std::stoi(argv[++index]));
            continue;
        }

        if (argument == "--threshold" && index + 1 < argc) {
            const int threshold = std::clamp(std::stoi(argv[++index]), 0, 255);
            config.threshold = static_cast<unsigned char>(threshold);
            continue;
        }

        throw std::invalid_argument("Unknown or incomplete argument: " + argument);
    }

    return config;
}

inline void printUsage(std::ostream& stream, const char* programName)
{
    stream << "Usage: " << programName
           << " --input <path> [--mode cpu|omp|gpu] [--repeat N] [--threshold V]\n"
           << "\n"
           << "Examples:\n"
           << "  " << programName << " --input data/CrackNJ156_Up/000.png --mode cpu --repeat 5\n"
           << "  " << programName << " --input data/CrackNJ156_Up/000.png --mode omp --repeat 5\n";
}

inline Image loadImageAsBgr(const std::string& path)
{
    const std::string script =
        "import sys\n"
        "from PIL import Image\n"
        "img = Image.open(sys.argv[1]).convert('RGB')\n"
        "w, h = img.size\n"
        "sys.stdout.buffer.write(f'{w} {h}\\n'.encode())\n"
        "sys.stdout.buffer.write(img.tobytes())\n";

    const std::string command = std::string(GPU_CRACK_PYTHON_EXECUTABLE) + " -c " + shellQuote(script) + " " + shellQuote(path);
    const std::string output = runCommandCapture(command);

    const size_t newline = output.find('\n');
    if (newline == std::string::npos) {
        throw std::runtime_error("Image loader returned malformed output");
    }

    std::istringstream header(output.substr(0, newline));
    int width = 0;
    int height = 0;
    header >> width >> height;
    if (width <= 0 || height <= 0) {
        throw std::runtime_error("Invalid image dimensions returned by loader");
    }

    const std::string raw = output.substr(newline + 1);
    const size_t expectedBytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
    if (raw.size() != expectedBytes) {
        throw std::runtime_error("Image loader returned unexpected byte count");
    }

    Image image;
    image.resize(width, height, 3);

    for (size_t index = 0; index < static_cast<size_t>(width) * static_cast<size_t>(height); ++index) {
        const size_t rgb = index * 3u;
        const size_t bgr = rgb;
        image.data[bgr + 0] = static_cast<unsigned char>(raw[rgb + 2]);
        image.data[bgr + 1] = static_cast<unsigned char>(raw[rgb + 1]);
        image.data[bgr + 2] = static_cast<unsigned char>(raw[rgb + 0]);
    }

    return image;
}

inline void ensureOutputDirectory()
{
    std::filesystem::create_directories("outputs");
}

inline std::string sanitizeStem(const std::filesystem::path& path)
{
    std::string stem = path.stem().string();
    if (stem.empty()) {
        stem = "image";
    }
    return stem;
}

inline std::string makeOutputStem(const std::filesystem::path& inputPath, const std::string& label, const std::string& timestamp)
{
    return sanitizeStem(inputPath) + "_" + label + "_" + timestamp;
}

inline void saveImagePgm(const std::filesystem::path& path, const Image& image)
{
    if (image.channels != 1) {
        throw std::runtime_error("PGM output requires a single-channel image");
    }

    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open output file: " + path.string());
    }

    file << "P5\n" << image.width << ' ' << image.height << "\n255\n";
    file.write(reinterpret_cast<const char*>(image.data.data()), static_cast<std::streamsize>(image.data.size()));
}

inline void saveImagePng(const std::filesystem::path& path, const Image& image, bool treatInputAsBgr)
{
    if (image.channels != 1 && image.channels != 3) {
        throw std::runtime_error("PNG output requires 1 or 3 channels");
    }

    const std::filesystem::path rawPath = path.string() + ".rawtmp";
    {
        std::ofstream raw(rawPath, std::ios::binary);
        if (!raw) {
            throw std::runtime_error("Failed to open temporary raw file: " + rawPath.string());
        }
        raw.write(reinterpret_cast<const char*>(image.data.data()), static_cast<std::streamsize>(image.data.size()));
    }

    const std::string script =
        "import pathlib, sys\n"
        "from PIL import Image\n"
        "w = int(sys.argv[1])\n"
        "h = int(sys.argv[2])\n"
        "c = int(sys.argv[3])\n"
        "raw_path = pathlib.Path(sys.argv[4])\n"
        "out_path = pathlib.Path(sys.argv[5])\n"
        "is_bgr = int(sys.argv[6])\n"
        "data = raw_path.read_bytes()\n"
        "if c == 1:\n"
        "    img = Image.frombytes('L', (w, h), data)\n"
        "elif c == 3:\n"
        "    if is_bgr:\n"
        "        converted = bytearray(len(data))\n"
        "        converted[0::3] = data[2::3]\n"
        "        converted[1::3] = data[1::3]\n"
        "        converted[2::3] = data[0::3]\n"
        "        data = bytes(converted)\n"
        "    img = Image.frombytes('RGB', (w, h), data)\n"
        "else:\n"
        "    raise ValueError('Unsupported channel count')\n"
        "img.save(out_path, format='PNG')\n";

    const std::string command = std::string(GPU_CRACK_PYTHON_EXECUTABLE)
        + " -c " + shellQuote(script)
        + " " + std::to_string(image.width)
        + " " + std::to_string(image.height)
        + " " + std::to_string(image.channels)
        + " " + shellQuote(rawPath.string())
        + " " + shellQuote(path.string())
        + " " + std::to_string(treatInputAsBgr ? 1 : 0);

    const int status = std::system(command.c_str());
    std::filesystem::remove(rawPath);
    if (status != 0) {
        throw std::runtime_error("Failed to save PNG output: " + path.string());
    }
}

inline double binaryMismatchPercent(const Image& lhs, const Image& rhs)
{
    if (lhs.width != rhs.width || lhs.height != rhs.height || lhs.channels != 1 || rhs.channels != 1) {
        throw std::runtime_error("Cannot compare binary images with different sizes or channel counts");
    }

    size_t mismatchedPixels = 0;
    for (size_t index = 0; index < lhs.pixelCount(); ++index) {
        if (lhs.data[index] != rhs.data[index]) {
            ++mismatchedPixels;
        }
    }

    if (lhs.pixelCount() == 0) {
        return 0.0;
    }

    return 100.0 * static_cast<double>(mismatchedPixels) / static_cast<double>(lhs.pixelCount());
}

inline double meanAbsoluteDifference(const Image& lhs, const Image& rhs)
{
    if (lhs.width != rhs.width || lhs.height != rhs.height || lhs.channels != rhs.channels) {
        throw std::runtime_error("Cannot compare images with different sizes or channel counts");
    }

    double totalDifference = 0.0;
    for (size_t index = 0; index < lhs.data.size(); ++index) {
        totalDifference += std::abs(static_cast<int>(lhs.data[index]) - static_cast<int>(rhs.data[index]));
    }

    return lhs.data.empty() ? 0.0 : totalDifference / static_cast<double>(lhs.data.size());
}

inline size_t largestConnectedComponentArea(const Image& binary, unsigned char edgeCutoff)
{
    if (binary.channels != 1) {
        throw std::runtime_error("Connected-component metric expects a single-channel image");
    }

    if (binary.pixelCount() == 0) {
        return 0;
    }

    const int width = binary.width;
    const int height = binary.height;
    std::vector<unsigned char> visited(binary.pixelCount(), 0);

    auto indexOf = [width](int x, int y) {
        return static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
    };

    size_t largestArea = 0;
    std::vector<size_t> stack;
    stack.reserve(binary.pixelCount());

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t startIndex = indexOf(x, y);
            if (visited[startIndex] || binary.data[startIndex] < edgeCutoff) {
                continue;
            }

            visited[startIndex] = 1;
            stack.clear();
            stack.push_back(startIndex);

            size_t area = 0;
            while (!stack.empty()) {
                const size_t current = stack.back();
                stack.pop_back();
                ++area;

                const int cx = static_cast<int>(current % static_cast<size_t>(width));
                const int cy = static_cast<int>(current / static_cast<size_t>(width));

                for (int ny = cy - 1; ny <= cy + 1; ++ny) {
                    if (ny < 0 || ny >= height) {
                        continue;
                    }
                    for (int nx = cx - 1; nx <= cx + 1; ++nx) {
                        if (nx < 0 || nx >= width) {
                            continue;
                        }
                        const size_t neighborIndex = indexOf(nx, ny);
                        if (!visited[neighborIndex] && binary.data[neighborIndex] >= edgeCutoff) {
                            visited[neighborIndex] = 1;
                            stack.push_back(neighborIndex);
                        }
                    }
                }
            }

            largestArea = std::max(largestArea, area);
        }
    }

    return largestArea;
}

inline double largestComponentAreaPercent(const Image& magnitude, unsigned char edgeCutoff = 20)
{
    if (magnitude.pixelCount() == 0) {
        return 0.0;
    }

    return 100.0 * static_cast<double>(largestConnectedComponentArea(magnitude, edgeCutoff))
        / static_cast<double>(magnitude.pixelCount());
}

} // namespace crack