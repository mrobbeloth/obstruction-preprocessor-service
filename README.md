# obstruction-preprocessor-service

This service prepares images for characterization by obstruction algorithms. It reads PNG images from the [COIL-100](https://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php) dataset, converts them to grayscale, applies sharpening and filtering (with optional CUDA GPU acceleration), performs k-means segmentation, and writes results to an `output/` directory.

## Prerequisites

| Dependency | Required | Notes |
|------------|----------|-------|
| CMake ≥ 3.10 | Yes | Build system |
| C++17 compiler | Yes | GCC, Clang, or MSVC |
| OpenCV 4.x | Yes | Core, imgcodecs, imgproc modules |
| NVIDIA CUDA Toolkit | No | Optional — enables GPU acceleration |
| OpenCV CUDA modules | No | `cudaimgproc`, `cudafilters`, `cudaarithm` — only needed with CUDA |

### Installing OpenCV

- **Linux (Debian/Ubuntu):** `sudo apt install libopencv-dev` (CPU-only). For CUDA support, [build from source](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html) with `-DWITH_CUDA=ON`.
- **Windows:** Download from [opencv.org/releases](https://opencv.org/releases/) or use [vcpkg](https://vcpkg.io/): `vcpkg install opencv4`

## Building

### Linux (GCC / Clang)

```bash
mkdir -p build && cd build

# CPU-only (no CUDA required)
cmake -DENABLE_CUDA=OFF ..
make -j$(nproc)

# With CUDA GPU acceleration (requires CUDA toolkit + OpenCV CUDA modules)
cmake -DENABLE_CUDA=ON ..
make -j$(nproc)
```

### Windows (MSVC — Visual Studio)

```powershell
mkdir build && cd build

# CPU-only
cmake -G "Visual Studio 17 2022" -DENABLE_CUDA=OFF -DOpenCV_DIR=C:\path\to\opencv\build ..
cmake --build . --config Release

# With CUDA
cmake -G "Visual Studio 17 2022" -DENABLE_CUDA=ON -DOpenCV_DIR=C:\path\to\opencv\build ..
cmake --build . --config Release
```

### Windows (MinGW)

```bash
mkdir build && cd build
cmake -G "MinGW Makefiles" -DENABLE_CUDA=OFF -DOpenCV_DIR=C:\path\to\opencv\build ..
mingw32-make
```

> **Note:** Set `OpenCV_DIR` to the directory containing `OpenCVConfig.cmake`. On Linux this is typically auto-detected; on Windows you may need to specify it explicitly.

## Running

The executable must be run from the `build/` directory so it can locate input images at `../data/coil-100/` and write results to `../output/`.

```bash
cd build

# Run with defaults (k=4 clusters, 16 k-means iterations)
./preprocessor

# Enable debug output
./preprocessor --debug

# Specify number of clusters (k) and max k-means iterations
./preprocessor --debug <k> <kMeansIterations>
```

**Arguments:**

| Position | Value | Default | Description |
|----------|-------|---------|-------------|
| 1 | `--debug` | *(debug on)* | Enable verbose/debug output |
| 2 | `<k>` | `4` | Number of k-means clusters |
| 3 | `<kMeansIterations>` | `16` | Maximum k-means iterations |

**Output** is written to `output/` (sibling of `build/`). Any existing `output/` directory is removed and recreated on each run.

When CUDA is available and the build was configured with `-DENABLE_CUDA=ON`, GPU acceleration is used automatically. Otherwise, all operations run on the CPU.

## Project Structure

```
├── CMakeLists.txt          # Build configuration (CUDA optional)
├── README.md
├── LICENSE                 # Apache 2.0
├── data/coil-100/          # Input images (COIL-100 dataset)
├── design/                 # UML diagrams, PlantUML sources
└── src/
    ├── preprocessor.cpp    # Main entry point and image pipeline
    ├── utility.cpp/.h      # Helper functions (sharpen, file I/O)
    └── CompositeMat.cpp/.h # Composite matrix data structure
```

## Development Environment

Visual Studio Code is the recommended IDE. See the project Wiki for workspace configuration details.
