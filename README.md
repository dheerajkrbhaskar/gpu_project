# GPU-Accelerated Crack Detection with Performance Profiling

This project detects road cracks using a classical image-processing pipeline and accelerates it with CUDA. It includes:

- a clean CPU baseline with optional OpenMP parallelism
- a CUDA implementation with shared-memory tiling for the Sobel stage
- timing breakdowns for CPU and GPU paths
- output images for grayscale, edge magnitude, and crack mask
- a simple CPU-vs-GPU correctness comparison metric


## Problem Statement

Road-crack inspection is a good fit for classical vision pipelines because the target structure is thin, high-contrast, and edge-dominant. The challenge is throughput: real inspections can involve many moderate-to-large images, and the Sobel/threshold stages are easy to parallelize.

The project demonstrates how the same pipeline can run on the CPU or GPU while keeping the output visually comparable and the performance impact measurable.

## Approach

Pipeline used in both implementations:

1. Load an RGB image from disk.
2. Convert it to grayscale.
3. Apply a 3x3 Sobel filter to compute gradient magnitude.
4. Normalize the magnitude to 8-bit range.
5. Threshold the magnitude to produce a binary crack map.
6. Save grayscale, magnitude, and binary outputs.

Parallelization strategy:

- CPU baseline uses straightforward nested loops.
- Optional OpenMP parallelizes the row loops.
- CUDA uses a 2D thread grid with 16x16 blocks.
- Sobel uses shared-memory tiling with a one-pixel halo to reduce global-memory traffic.
- GPU timings are split into H2D copy, kernel execution, and D2H copy.

## Repository Layout

- `src/main.cpp` command-line entry point and benchmarking harness
- `src/cpu_pipeline.cpp` CPU/OpenMP pipeline
- `src/gpu_pipeline.cu` CUDA pipeline and kernels
- `src/timer.h` high-resolution timing helpers
- `src/utils.h` argument parsing, image helpers, output naming, and metrics
- `data/D/UD/` clean or mostly clean road images
- `data/D/CD/` crack-damaged road images
- `outputs/` generated PNG results

## Build

Requirements:

- CMake 3.20+
- Python 3 with `Pillow` for image loading and PNG writing
- a CUDA-capable compiler toolchain with `nvcc` for GPU mode only
- OpenMP support is optional

Example build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build -j
```

If CUDA is not installed, the project still builds the CPU/OpenMP path automatically and the `gpu` mode falls back to the CPU pipeline with a clear message.

If your GPU requires a different architecture, override `CMAKE_CUDA_ARCHITECTURES` accordingly.

## Run

Examples:

```bash
./build/gpu_crack_detection --input data/D/UD/7001-12.jpg --mode cpu --repeat 5
./build/gpu_crack_detection --input data/D/UD/7001-12.jpg --mode omp --repeat 5
./build/gpu_crack_detection --input data/D/CD/7001-41.jpg --mode gpu --repeat 5
```

Flags:

- `--input <path>` input image path
- `--mode cpu|omp|gpu` execution mode
- `--repeat N` average over `N` runs
- `--threshold V` binary threshold in 8-bit magnitude space

If no CUDA device is available, GPU mode automatically falls back to the CPU pipeline and reports that in the output.

## Outputs

For each run, the project writes four PNG images into `outputs/`:

- input image
- grayscale image
- edge-magnitude image
- binary crack map

Output filenames include the execution label and a timestamp.

## Results

Use this section to record measured numbers from your own GPU and dataset.

Example table template:

| Mode | Gray ms | Sobel ms | Threshold ms | H2D ms | Kernel ms | D2H ms | Total ms | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CPU |  |  |  | - | - | - |  | 1.00x |
| OMP |  |  |  | - | - | - |  |  |
| GPU | - | - | - |  |  |  |  |  |

## Discussion

### Bottlenecks

- Grayscale conversion is memory-bandwidth friendly and usually not the dominant cost.
- Sobel is the heaviest stage on the CPU because it touches a 3x3 neighborhood per pixel.
- GPU speedups are limited by host/device transfers for small images.

### Optimizations

- Shared-memory tiling reduces repeated global-memory fetches in the Sobel stage.
- Coalesced reads and writes are preserved by mapping one thread per pixel.
- Using a fixed 16x16 block size keeps the kernel simple and predictable.

### Trade-offs

- GPU acceleration adds transfer overhead.
- Python/Pillow keeps image I/O simple without adding a large C++ image library dependency.
- The CUDA path is fast for moderate and large images, while the CPU path is portable and easier to debug.

## Expected Speedups

On moderate image sizes, a well-behaved GPU configuration should often beat the single-threaded CPU baseline by several times, with typical gains around 3x or more depending on the image size, GPU model, and whether OpenMP is enabled on the CPU side.

## Author
- Dheeraj Kumar Bhaskar
- dheerajbhaskar.dev
