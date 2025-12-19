# hmat-omp: hierarhical matrices with weak admissibility

A lightweight, high-performance C++ header-only library for Hierarchical OFF-Diagonal (HODLR) matrices and Hierarchically Semiseparable (HSS) Matrices. This library implements OpenMP-based parallelization and Randomized SVD to achieve orders-of-magnitude speedups over standard dense linear algebra (BLAS/LAPACK) for data-sparse problems (e.g., Cauchy, Gaussian, Coulomb kernels).

 Key FeaturesFast
 
 * Construction: Utilizes Randomized SVD (RSVD) to reduce construction complexity from $O(N^3)$ to approx $O(N \log N)$ or $O(N)$.
 * Fast Direct Solver: Implements an exact Woodbury Matrix Identity solver (not iterative), achieving $O(N)$ complexity for solving linear systems.
 * Task-Based Parallelism: Uses OpenMP Tasks to recursively parallelize tree traversals.
 * LAPACK/BLAS Integration: Lightweight wrapper around standard LAPACK routines for dense block operations.


## Compilation

Linux / Windows:

```bash
mkdir build && cd build
cmake ..
make
./hodlr_test
```

macOS (Homebrew): Since macOS clang does not ship with OpenMP by default:
```bash
brew install libomp
mkdir build && cd build
# CMake will automatically find libomp if installed via brew
cmake .. 
make
./hodlr_test # ./hss_test
```



## Prerequisites

C++ Compiler (C++14 or later), CMake (>= 3.10), BLAS & LAPACK (OpenBLAS, MKL, or Accelerate), OpenMP (libomp)


## Structure

```text
project/
├── CMakeLists.txt       # Build configuration (Auto-detects OpenMP/BLAS)
├── src/
│   └── main.cpp         # Benchmark driver & correctness tests
└── include/
    ├── matrix.h         # Lightweight wrapper for LAPACK/BLAS (LU, Inverse, SVD)
    ├── compression.h    # Algorithms: Standard SVD & Randomized SVD
    ├── utils.h          # HSS Tree Node Data Structures
    ├── kernel.h         # Data generators (Cauchy, Gaussian, Coulomb)
    ├── hodlr_routines.h   # Serial Implementation (Reference)
    └── hodlr_routines_omp.h # Parallel Implementation (The Core Code)
    ├── hss_routines.h   # Serial Implementation (Reference)
    └── hss_routines_omp.h # Parallel Implementation (The Core Code)
```


## License

Under MIT License. Feel free to contribute and use for research or educational purposes.