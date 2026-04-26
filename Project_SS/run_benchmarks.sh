#!/usr/bin/env bash
set -euo pipefail

# HPC benchmark orchestrator:
# 1) Builds CPU + CUDA binaries with aggressive optimization flags
# 2) Generates binary datasets (.bin) for reproducible experiments
# 3) Runs solver binaries sequentially + unified benchmark driver

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

N="${N:-1000000}"
Q="${Q:-100000}"
VALUE_RANGE="${VALUE_RANGE:-1000000}"
SEED="${SEED:-42}"
CUDA_ARCH="${CUDA_ARCH:-sm_86}"   # Override per GPU, e.g. sm_80/sm_89/sm_90
BLOCK_SIZE="${BLOCK_SIZE:-256}"
THREADS="${THREADS:-$(nproc)}"
MODE="${1:-full}"                  # full | quick

echo "[1/5] Building CPU binaries (O3, march=native, OpenMP)"
g++ -O3 -std=c++17 -march=native mos.cpp -o mos
g++ -O3 -std=c++17 -march=native -fopenmp omp_bruteforce.cpp -o omp_bruteforce
g++ -O3 -std=c++17 -march=native input_generator.cpp -o input_generator
g++ -O3 -std=c++17 -march=native -fopenmp benchmark.cpp -o benchmark

echo "[2/5] Building CUDA binary (O3, ${CUDA_ARCH})"
if command -v nvcc >/dev/null 2>&1; then
  nvcc -O3 -std=c++17 -arch="${CUDA_ARCH}" cuda_kernel.cu -o cuda_kernel
else
  echo "WARN: nvcc not found; skipping CUDA build and CUDA runs."
fi

echo "[3/5] Generating binary datasets (.bin) in results/"
./input_generator \
  --n "${N}" \
  --q "${Q}" \
  --value-range "${VALUE_RANGE}" \
  --seed "${SEED}" \
  --out-dir results

echo "[4/5] Running per-method sanity benchmarks"
./mos --n "${N}" --q "${Q}" --value-range "${VALUE_RANGE}" --max-len 2048 --seed "${SEED}" | tee results/mos.log
./omp_bruteforce --n "${N}" --q "${Q}" --value-range "${VALUE_RANGE}" --max-len 2048 --threads "${THREADS}" --seed "${SEED}" | tee results/omp.log
if [[ -x ./cuda_kernel ]]; then
  ./cuda_kernel --n "${N}" --q "${Q}" --value-range "${VALUE_RANGE}" --max-len 2048 --block-size "${BLOCK_SIZE}" --seed "${SEED}" | tee results/cuda.log
fi

echo "[5/5] Running aggregate benchmark suite"
if [[ "${MODE}" == "quick" ]]; then
  ./benchmark --n "${N}" --value-range "${VALUE_RANGE}" --seed "${SEED}" --quick | tee results/benchmark.log
else
  ./benchmark --n "${N}" --value-range "${VALUE_RANGE}" --seed "${SEED}" | tee results/benchmark.log
fi

echo "Done. Primary output: results/data.csv"
echo "Note: Current solver binaries are seed-driven (in-memory generation)."
echo "      Binary datasets are generated for reproducibility and external I/O benchmarking workflows."
