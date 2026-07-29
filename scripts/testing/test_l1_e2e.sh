#!/usr/bin/env bash

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build-e2e}"
readonly DEFAULT_DATASET_DIR="/data/jingyue.zjl/ob_data/L1/mnist-784"
readonly DATASET_PATH="${1:-${DEFAULT_DATASET_DIR}/mnist-784-l1-1k.hdf5}"
readonly BUILD_JOBS="${COMPILE_JOBS:-36}"

cd "${REPO_ROOT}"

cmake -S . -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_ASAN=OFF \
    -DENABLE_CCACHE=ON \
    -DENABLE_TESTS=ON \
    -DENABLE_EXAMPLES=ON \
    -DENABLE_L1_E2E=ON \
    -DENABLE_TOOLS=OFF \
    -DENABLE_MOCKIMPL=OFF
cmake --build "${BUILD_DIR}" \
    --target unittests 114_index_float32_l1_hnsw \
    --parallel "${BUILD_JOBS}"

"${BUILD_DIR}/tests/unittests" -d yes '[l1]'
"${BUILD_DIR}/examples/cpp/114_index_float32_l1_hnsw" "${DATASET_PATH}"
