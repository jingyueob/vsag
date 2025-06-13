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

#include "fht_kac_rotator.h"
#include <iostream>
namespace vsag {

static inline void 
flip_array(const uint8_t* flip, float* data, size_t dim) {  //generic
    for (size_t i = 0; i < dim; i ++) {
        bool mask = (flip[i/8] & (1 << (i%8))) != 0;
        if (mask) {
          data[i] = -data[i];
        }
    }
}

inline size_t 
floor_log2(size_t x) {  //smaller or equal
    size_t ret = 0;
    while (x > 1) {
        ret++;
        x >>= 1;
    }
    return ret;
}
inline void 
vec_rescale(float* data, size_t dim, float val) {
    for(int i = 0; i < dim; i++){
        data[i] *= val;
    }
}

void
FhtKacRotator::CopyFlip(uint8_t* out_flip) const {
    std::copy(flip_.data(), flip_.data() + flip_.size(), out_flip);
}

FhtKacRotator::FhtKacRotator(uint64_t dim, Allocator* allocator)
    :dim_(dim),  allocator_(allocator){
    flip_offset_ = (dim_ + 7) / kByteLen_;
    flip_.resize(round_ * flip_offset_);    // round = 4
    size_t bottom_log_dim = floor_log2(dim);
    trunc_dim_ = 1 << bottom_log_dim;
    fac_ = 1.0F / std::sqrt(static_cast<float>(trunc_dim_));
}
FhtKacRotator::~FhtKacRotator() = default;

bool 
FhtKacRotator::Build()
{
    std::random_device rd;   // Seed
    std::mt19937 gen(rd());  // Mersenne Twister RNG
    std::uniform_int_distribution<int> dist(0, 255);
    for (auto& i : flip_) {
        i = static_cast<uint8_t>(dist(gen));
    }
    return true;
}

static void 
kacs_walk_generic(float* data, size_t len) {
    // ! len % 32 == 0;
    size_t base = len % 2;
    size_t offset = base + (len / 2); // for odd dim
    for (size_t i = 0; i < len / 2; i ++) {
        float add = data[i] + data[i + offset];
        float sub = data[i] - data[i + offset];
        data[i] = add;
        data[i + offset] = sub;
    }
    if(base != 0){
        data[len / 2] *= sqrt(2.0f); 
        //In odd condition, we operate the prev len/2 items and the post len/2 items, the No.len/2 item stay still,
        //As we need to resize the while sequence in the next step, so we increase the val of No.len/2 item to eliminate the impact of the following resize.
    }
}

void 
FhtKacRotator::fht_float_(float* data) const {
    size_t n = trunc_dim_;
    size_t step = 1;
    while (step < n) {
        for (size_t i = 0; i < n; i += step * 2) {
            for (int j = 0; j < step; j++) {
                float even = data[i + j];
                float odd = data[i + j + step];
                data[i + j] = even + odd;
                data[i + j + step] = even - odd;
            }
        }
        step *= 2;
    }
}

void 
FhtKacRotator::Transform(const float* data, float* rotated_vec) const{
    std::memcpy(rotated_vec, data, sizeof(float) * dim_);
    std::fill(rotated_vec + dim_, rotated_vec + dim_, 0);
    if (trunc_dim_ == dim_) {
        flip_array(flip_.data() + 0 * flip_offset_, rotated_vec, dim_);
        fht_float_(rotated_vec);
        vec_rescale(rotated_vec, trunc_dim_, fac_);

        flip_array(flip_.data()+ 1 * flip_offset_, rotated_vec, dim_);
        fht_float_(rotated_vec);
        vec_rescale(rotated_vec, trunc_dim_, fac_);

        flip_array(flip_.data()+ 2 * flip_offset_, rotated_vec, dim_);
        fht_float_(rotated_vec);
        vec_rescale(rotated_vec, trunc_dim_, fac_);

        flip_array(flip_.data()+ 3 * flip_offset_, rotated_vec, dim_);
        fht_float_(rotated_vec);
        vec_rescale(rotated_vec, trunc_dim_, fac_);

        return;
    }

    size_t start = dim_ - trunc_dim_;

    flip_array(flip_.data() + 0 * flip_offset_, rotated_vec, dim_);
    fht_float_(rotated_vec);
    vec_rescale(rotated_vec, trunc_dim_, fac_);
    kacs_walk_generic(rotated_vec, dim_);
    vec_rescale(rotated_vec, dim_, sqrt(0.5f));  
    //origin vec(x,y), after kacs_walk_generic() -> (x+y, x-y),should be resize by sqrt(0.5) to make the len of vector consistency

    flip_array(flip_.data() + flip_offset_, rotated_vec, dim_);
    fht_float_(rotated_vec + start);
    vec_rescale(rotated_vec + start, trunc_dim_, fac_);
    kacs_walk_generic(rotated_vec, dim_);
    vec_rescale(rotated_vec, dim_, sqrt(0.5f));

    flip_array(flip_.data() + 2 * flip_offset_, rotated_vec, dim_);
    fht_float_(rotated_vec);
    vec_rescale(rotated_vec, trunc_dim_, fac_);
    kacs_walk_generic(rotated_vec, dim_);
    vec_rescale(rotated_vec, dim_, sqrt(0.5f));

    flip_array(flip_.data() + 3 * flip_offset_, rotated_vec, dim_);
    fht_float_(rotated_vec + start);
    vec_rescale(rotated_vec + start, trunc_dim_, fac_);
    kacs_walk_generic(rotated_vec, dim_);
    vec_rescale(rotated_vec, dim_, sqrt(0.5f));

}
void 
FhtKacRotator::InverseTransform(float const*data, float*rotated_vec) const {
    std::memcpy(rotated_vec, data, sizeof(float) * dim_);
    std::fill(rotated_vec + dim_, rotated_vec + dim_, 0);
    if (trunc_dim_ == dim_) {
        fht_float_(rotated_vec);
        vec_rescale(rotated_vec, trunc_dim_, fac_);
        flip_array(flip_.data() + 3 * flip_offset_, rotated_vec, dim_);
        
        fht_float_(rotated_vec);
        vec_rescale(rotated_vec, trunc_dim_, fac_);
        flip_array(flip_.data()+ 2 * flip_offset_, rotated_vec, dim_);

        fht_float_(rotated_vec);
        vec_rescale(rotated_vec, trunc_dim_, fac_);
        flip_array(flip_.data()+ 1 * flip_offset_, rotated_vec, dim_);

        fht_float_(rotated_vec);
        vec_rescale(rotated_vec, trunc_dim_, fac_);
        flip_array(flip_.data() + 0 * flip_offset_, rotated_vec, dim_);

        return;
    }

    size_t start = dim_ - trunc_dim_;

    kacs_walk_generic(rotated_vec,dim_);
    vec_rescale(rotated_vec, dim_, sqrt(0.5f));
    fht_float_(rotated_vec + start);
    vec_rescale(rotated_vec + start, trunc_dim_, fac_);
    flip_array(flip_.data() + 3 * flip_offset_,rotated_vec, dim_);

    kacs_walk_generic(rotated_vec, dim_);
    vec_rescale(rotated_vec, dim_, sqrt(0.5f));
    fht_float_(rotated_vec);
    vec_rescale(rotated_vec, trunc_dim_, fac_);
    flip_array(flip_.data() + 2 * flip_offset_,rotated_vec, dim_);

    kacs_walk_generic(rotated_vec,dim_);
    vec_rescale(rotated_vec, dim_, sqrt(0.5f));
    fht_float_(rotated_vec + start);
    vec_rescale(rotated_vec + start, trunc_dim_, fac_);
    flip_array(flip_.data() + 1 * flip_offset_,rotated_vec, dim_);

    kacs_walk_generic(rotated_vec, dim_);
    vec_rescale(rotated_vec, dim_, sqrt(0.5f));
    fht_float_(rotated_vec);
    vec_rescale(rotated_vec, trunc_dim_, fac_);
    flip_array(flip_.data() + 0 * flip_offset_,rotated_vec, dim_);
}

void 
FhtKacRotator::Serialize(StreamWriter& writer) {
    StreamWriter::WriteVector(writer, this->flip_);
}

void
FhtKacRotator::Deserialize(StreamReader& reader) {
    StreamReader::ReadVector(reader, this->flip_);
}


}
