#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-e2e}"
readonly DATASET_PATH="${1:-/data/jingyue.zjl/ob_data/hamming/sift-256-hamming.hdf5}"
readonly BUILD_JOBS="${COMPILE_JOBS:-6}"

cd "${REPO_ROOT}"

cmake -S . -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_ASAN=OFF \
    -DENABLE_CCACHE=ON \
    -DENABLE_TESTS=ON \
    -DENABLE_EXAMPLES=ON \
    -DENABLE_BINARY_HAMMING_E2E=ON \
    -DENABLE_TOOLS=OFF \
    -DENABLE_MOCKIMPL=OFF
cmake --build "${BUILD_DIR}" \
    --target unittests 113_index_binary_hamming_hnsw \
    --parallel "${BUILD_JOBS}"

"${BUILD_DIR}/tests/unittests" -d yes '[binary],[hamming]'
"${BUILD_DIR}/examples/cpp/113_index_binary_hamming_hnsw" "${DATASET_PATH}"
