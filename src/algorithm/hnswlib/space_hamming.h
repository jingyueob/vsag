// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "simd/basic_func.h"
#include "space_interface.h"

namespace hnswlib {

class HammingSpace : public SpaceInterface {
public:
    explicit HammingSpace(uint64_t byte_dim)
        : fstdistfunc_(vsag::Hamming), data_size_(byte_dim), byte_dim_(byte_dim) {
    }

    uint64_t
    get_data_size() override {
        return data_size_;
    }

    DISTFUNC
    get_dist_func() override {
        return fstdistfunc_;
    }

    void*
    get_dist_func_param() override {
        return &byte_dim_;
    }

private:
    DISTFUNC fstdistfunc_;
    uint64_t data_size_;
    uint64_t byte_dim_;
};

}  // namespace hnswlib
