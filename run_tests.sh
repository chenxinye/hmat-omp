#!/bin/bash

# ==========================================
#  HSS & HODLR 
# ==========================================

export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

export OMP_NUM_THREADS=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)

echo "========================================"
echo " Environmental settings"
echo "========================================"
echo " OMP_NUM_THREADS      : $OMP_NUM_THREADS"
echo " OPENBLAS_NUM_THREADS : $OPENBLAS_NUM_THREADS"
echo "========================================"

rm -f hss_test hodlr_test

echo -e "\n[Compiling] HSS Matrix Test..."
g++ ./src/main_hss.cpp -o hss_test \
    -O3 -fopenmp -std=c++17 \
    -lblas -llapack \
    -Wall

if [ $? -ne 0 ]; then
    echo "HSS 编译失败！"
    exit 1
fi

echo -e "[Compiling] HODLR Matrix Test..."
g++ ./src/main_hodlr.cpp -o hodlr_test \
    -O3 -fopenmp -std=c++17 \
    -lblas -llapack \
    -Wall

if [ $? -ne 0 ]; then
    echo "HODLR compilation fail！"
    exit 1
fi

echo "compilation works! run tests..."

echo -e "\n\n########################################"
echo " Running HSS Benchmark"
echo "########################################"
./hss_test

echo -e "\n\n########################################"
echo " Running HODLR Benchmark"
echo "########################################"
./hodlr_test

echo -e "\nTesting Workflow Completed."