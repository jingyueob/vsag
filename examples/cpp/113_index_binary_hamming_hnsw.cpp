// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <vsag/vsag.h>

#include <H5Cpp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr const char* kDefaultDatasetPath =
    "/data/jingyue.zjl/ob_data/hamming/sift-256-hamming.hdf5";
constexpr int64_t kQueryCount = 100;
constexpr int64_t kTopK = 10;
constexpr float kMinimumRecall = 0.90F;
constexpr uint64_t kReadChunkRows = 4096;
constexpr float kFloatTolerance = 1e-6F;

struct HammingDataset {
    int64_t dim;
    int64_t base_count;
    int64_t query_count;
    int64_t truth_k;
    std::vector<int64_t> ids;
    std::vector<uint8_t> base;
    std::vector<uint8_t> queries;
    std::vector<int64_t> neighbors;
    std::vector<float> distances;
};

struct SearchResults {
    std::vector<int64_t> ids;
    std::vector<float> distances;
};

class TemporaryFile {
public:
    explicit TemporaryFile(std::filesystem::path path) : path_(std::move(path)) {
    }

    TemporaryFile(const TemporaryFile&) = delete;
    TemporaryFile&
    operator=(const TemporaryFile&) = delete;

    ~TemporaryFile() {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    [[nodiscard]] const std::filesystem::path&
    Path() const {
        return path_;
    }

private:
    std::filesystem::path path_;
};

void
Require(bool condition, const std::string& message) {
    if (not condition) {
        throw std::runtime_error(message);
    }
}

template <typename Expected>
void
RequireExpected(const Expected& result, const std::string& stage) {
    if (not result.has_value()) {
        throw std::runtime_error(stage + " failed: " + result.error().message);
    }
}

std::pair<uint64_t, uint64_t>
GetMatrixShape(const H5::DataSet& dataset, const std::string& name) {
    auto data_space = dataset.getSpace();
    Require(data_space.getSimpleExtentNdims() == 2, name + " must be a two-dimensional matrix");

    hsize_t dimensions[2]{};
    data_space.getSimpleExtentDims(dimensions);
    return {dimensions[0], dimensions[1]};
}

std::string
ReadStringAttribute(const H5::H5File& file, const std::string& name) {
    Require(file.attrExists(name), "missing HDF5 attribute: " + name);
    auto attribute = file.openAttribute(name);
    auto string_type = attribute.getStrType();
    std::string value;
    attribute.read(string_type, value);
    return value;
}

std::vector<uint8_t>
ReadAndPackBits(const H5::H5File& file, const std::string& name, uint64_t rows) {
    auto input = file.openDataSet(name);
    const auto [available_rows, dim] = GetMatrixShape(input, name);
    Require(rows <= available_rows, name + " does not contain enough rows");
    Require(dim > 0 and dim % 8 == 0, name + " dimension must be a positive multiple of 8");

    const uint64_t bytes_per_vector = dim / 8;
    std::vector<uint8_t> packed(rows * bytes_per_vector, 0);

    for (uint64_t row_offset = 0; row_offset < rows; row_offset += kReadChunkRows) {
        const uint64_t chunk_rows = std::min(kReadChunkRows, rows - row_offset);
        hsize_t start[2]{row_offset, 0};
        hsize_t count[2]{chunk_rows, dim};
        auto file_space = input.getSpace();
        file_space.selectHyperslab(H5S_SELECT_SET, count, start);
        H5::DataSpace memory_space(2, count);
        std::vector<uint8_t> unpacked(chunk_rows * dim);
        input.read(
            unpacked.data(), H5::PredType::NATIVE_UINT8, memory_space, file_space);

        for (uint64_t row = 0; row < chunk_rows; ++row) {
            for (uint64_t bit = 0; bit < dim; ++bit) {
                if (unpacked[row * dim + bit] != 0) {
                    packed[(row_offset + row) * bytes_per_vector + bit / 8] |=
                        static_cast<uint8_t>(1U << (bit % 8));
                }
            }
        }
    }
    return packed;
}

template <typename T>
std::vector<T>
ReadFirstRows(const H5::H5File& file,
              const std::string& name,
              uint64_t rows,
              const H5::PredType& memory_type) {
    auto input = file.openDataSet(name);
    const auto [available_rows, columns] = GetMatrixShape(input, name);
    Require(rows <= available_rows, name + " does not contain enough rows");

    hsize_t start[2]{0, 0};
    hsize_t count[2]{rows, columns};
    auto file_space = input.getSpace();
    file_space.selectHyperslab(H5S_SELECT_SET, count, start);
    H5::DataSpace memory_space(2, count);
    std::vector<T> result(rows * columns);
    input.read(result.data(), memory_type, memory_space, file_space);
    return result;
}

HammingDataset
LoadHammingDataset(const std::string& path) {
    Require(std::filesystem::exists(path), "HDF5 file does not exist: " + path);
    H5::H5File file(path, H5F_ACC_RDONLY);
    Require(ReadStringAttribute(file, "distance") == "hamming",
            "HDF5 distance attribute must be hamming");
    Require(ReadStringAttribute(file, "point_type") == "bit",
            "HDF5 point_type attribute must be bit");

    const auto train_shape = GetMatrixShape(file.openDataSet("train"), "train");
    const auto test_shape = GetMatrixShape(file.openDataSet("test"), "test");
    const auto neighbors_shape = GetMatrixShape(file.openDataSet("neighbors"), "neighbors");
    const auto distances_shape = GetMatrixShape(file.openDataSet("distances"), "distances");

    Require(train_shape.first >= 2, "train must contain at least two rows");
    Require(test_shape.first >= static_cast<uint64_t>(kQueryCount),
            "test must contain at least 100 rows");
    Require(train_shape.second == test_shape.second, "train and test dimensions must match");
    Require(train_shape.second > 0 and train_shape.second % 8 == 0,
            "binary vector dimension must be a positive multiple of 8");
    Require(neighbors_shape == distances_shape, "neighbors and distances shapes must match");
    Require(neighbors_shape.first >= static_cast<uint64_t>(kQueryCount),
            "neighbors must contain at least 100 rows");
    Require(neighbors_shape.second >= static_cast<uint64_t>(kTopK),
            "neighbors must contain at least 10 columns");
    Require(train_shape.first <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
            "train row count exceeds int64_t");
    Require(train_shape.second <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
            "train dimension exceeds int64_t");
    Require(neighbors_shape.second <=
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
            "ground-truth width exceeds int64_t");

    HammingDataset result{
        .dim = static_cast<int64_t>(train_shape.second),
        .base_count = static_cast<int64_t>(train_shape.first),
        .query_count = kQueryCount,
        .truth_k = static_cast<int64_t>(neighbors_shape.second),
    };
    result.ids.resize(train_shape.first);
    std::iota(result.ids.begin(), result.ids.end(), int64_t{0});
    result.base = ReadAndPackBits(file, "train", train_shape.first);
    result.queries = ReadAndPackBits(file, "test", kQueryCount);
    result.neighbors =
        ReadFirstRows<int64_t>(file, "neighbors", kQueryCount, H5::PredType::NATIVE_INT64);
    result.distances =
        ReadFirstRows<float>(file, "distances", kQueryCount, H5::PredType::NATIVE_FLOAT);
    return result;
}

uint64_t
HammingDistance(const uint8_t* lhs, const uint8_t* rhs, uint64_t byte_count) {
    uint64_t distance = 0;
    for (uint64_t i = 0; i < byte_count; ++i) {
        uint8_t value = lhs[i] ^ rhs[i];
        while (value != 0) {
            value &= static_cast<uint8_t>(value - 1);
            ++distance;
        }
    }
    return distance;
}

std::string
BuildParameters(int64_t dim) {
    std::ostringstream stream;
    stream << R"({
        "dtype": "binary",
        "metric_type": "hamming",
        "dim": )"
           << dim << R"(,
        "hnsw": {
            "max_degree": 32,
            "ef_construction": 200
        }
    })";
    return stream.str();
}

const std::string&
SearchParameters() {
    static const std::string parameters = R"({
        "hnsw": {
            "ef_search": 200
        }
    })";
    return parameters;
}

vsag::IndexPtr
CreateIndex(const std::string& parameters) {
    auto result = vsag::Factory::CreateIndex("hnsw", parameters);
    RequireExpected(result, "CreateIndex");
    return result.value();
}

vsag::DatasetPtr
MakeQuery(const HammingDataset& dataset, int64_t query_id) {
    const uint64_t bytes_per_vector = static_cast<uint64_t>(dataset.dim) / 8;
    return vsag::Dataset::Make()
        ->NumElements(1)
        ->Dim(dataset.dim)
        ->BinaryVectors(dataset.queries.data() +
                        static_cast<uint64_t>(query_id) * bytes_per_vector)
        ->Owner(false);
}

SearchResults
RunKnnSearch(const vsag::IndexPtr& index, const HammingDataset& dataset) {
    SearchResults results;
    results.ids.reserve(static_cast<uint64_t>(dataset.query_count) * kTopK);
    results.distances.reserve(static_cast<uint64_t>(dataset.query_count) * kTopK);

    for (int64_t query_id = 0; query_id < dataset.query_count; ++query_id) {
        auto search = index->KnnSearch(
            MakeQuery(dataset, query_id), kTopK, SearchParameters());
        RequireExpected(search, "KnnSearch query " + std::to_string(query_id));
        Require(search.value()->GetDim() == kTopK,
                "KnnSearch returned an unexpected result count");
        results.ids.insert(results.ids.end(),
                           search.value()->GetIds(),
                           search.value()->GetIds() + kTopK);
        results.distances.insert(results.distances.end(),
                                 search.value()->GetDistances(),
                                 search.value()->GetDistances() + kTopK);
    }
    return results;
}

float
ComputeRecall(const SearchResults& results, const HammingDataset& dataset) {
    uint64_t matches = 0;
    for (int64_t query_id = 0; query_id < dataset.query_count; ++query_id) {
        std::unordered_set<int64_t> truth;
        const uint64_t truth_offset = static_cast<uint64_t>(query_id) * dataset.truth_k;
        for (int64_t i = 0; i < kTopK; ++i) {
            truth.insert(dataset.neighbors[truth_offset + i]);
        }
        const uint64_t result_offset = static_cast<uint64_t>(query_id) * kTopK;
        for (int64_t i = 0; i < kTopK; ++i) {
            matches += truth.count(results.ids[result_offset + i]);
        }
    }
    return static_cast<float>(matches) /
           static_cast<float>(dataset.query_count * kTopK);
}

void
ValidateSearchDistances(const SearchResults& results, const HammingDataset& dataset) {
    const uint64_t bytes_per_vector = static_cast<uint64_t>(dataset.dim) / 8;
    for (int64_t query_id = 0; query_id < dataset.query_count; ++query_id) {
        const auto* query =
            dataset.queries.data() + static_cast<uint64_t>(query_id) * bytes_per_vector;
        for (int64_t i = 0; i < kTopK; ++i) {
            const uint64_t offset = static_cast<uint64_t>(query_id) * kTopK + i;
            const int64_t id = results.ids[offset];
            Require(id >= 0 and id < dataset.base_count,
                    "KnnSearch returned an out-of-range ID");
            const auto* base =
                dataset.base.data() + static_cast<uint64_t>(id) * bytes_per_vector;
            const float expected =
                static_cast<float>(HammingDistance(query, base, bytes_per_vector));
            Require(std::fabs(results.distances[offset] - expected) <= kFloatTolerance,
                    "KnnSearch returned an incorrect Hamming distance");
        }
    }
}

void
ValidateRangeAndDistance(const vsag::IndexPtr& index, const HammingDataset& dataset) {
    auto query = MakeQuery(dataset, 0);
    const float radius =
        std::round(dataset.distances[kTopK - 1] * static_cast<float>(dataset.dim));
    auto range = index->RangeSearch(query, radius, SearchParameters());
    RequireExpected(range, "RangeSearch");
    Require(range.value()->GetDim() > 0, "RangeSearch returned no results");

    const uint64_t bytes_per_vector = static_cast<uint64_t>(dataset.dim) / 8;
    const auto* query_data = dataset.queries.data();
    for (int64_t i = 0; i < range.value()->GetDim(); ++i) {
        const int64_t id = range.value()->GetIds()[i];
        Require(id >= 0 and id < dataset.base_count,
                "RangeSearch returned an out-of-range ID");
        const auto* base =
            dataset.base.data() + static_cast<uint64_t>(id) * bytes_per_vector;
        const float expected =
            static_cast<float>(HammingDistance(query_data, base, bytes_per_vector));
        Require(range.value()->GetDistances()[i] <= radius,
                "RangeSearch returned a distance above radius");
        Require(std::fabs(range.value()->GetDistances()[i] - expected) <= kFloatTolerance,
                "RangeSearch returned an incorrect Hamming distance");
    }

    const int64_t nearest_id = dataset.neighbors[0];
    auto distance = index->CalcDistanceById(query, nearest_id);
    RequireExpected(distance, "CalcDistanceById");
    const auto* nearest =
        dataset.base.data() + static_cast<uint64_t>(nearest_id) * bytes_per_vector;
    const float expected =
        static_cast<float>(HammingDistance(query_data, nearest, bytes_per_vector));
    Require(std::fabs(distance.value() - expected) <= kFloatTolerance,
            "CalcDistanceById returned an incorrect Hamming distance");
    Require(std::fabs(distance.value() / static_cast<float>(dataset.dim) -
                      dataset.distances[0]) <= kFloatTolerance,
            "CalcDistanceById disagrees with HDF5 ground truth");
}

void
RequireSameResults(const SearchResults& before, const SearchResults& after) {
    Require(before.ids == after.ids, "restored index returned different IDs");
    Require(before.distances.size() == after.distances.size(),
            "restored index returned a different distance count");
    for (uint64_t i = 0; i < before.distances.size(); ++i) {
        Require(std::fabs(before.distances[i] - after.distances[i]) <= kFloatTolerance,
                "restored index returned different distances");
    }
}

float
RunEndToEnd(const std::string& dataset_path) {
    std::cout << "[1/6] Loading and packing HDF5 dataset" << std::endl;
    auto dataset = LoadHammingDataset(dataset_path);
    const uint64_t bytes_per_vector = static_cast<uint64_t>(dataset.dim) / 8;
    const auto build_parameters = BuildParameters(dataset.dim);

    std::cout << "[2/6] Building Binary/Hamming HNSW with " << dataset.base_count
              << " vectors" << std::endl;
    auto index = CreateIndex(build_parameters);
    auto base = vsag::Dataset::Make()
                    ->NumElements(dataset.base_count - 1)
                    ->Dim(dataset.dim)
                    ->Ids(dataset.ids.data())
                    ->BinaryVectors(dataset.base.data())
                    ->Owner(false);
    auto build = index->Build(base);
    RequireExpected(build, "Build");
    Require(build.value().empty(), "Build reported failed IDs");

    const uint64_t last = static_cast<uint64_t>(dataset.base_count - 1);
    auto added = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(dataset.dim)
                     ->Ids(dataset.ids.data() + last)
                     ->BinaryVectors(dataset.base.data() + last * bytes_per_vector)
                     ->Owner(false);
    auto add = index->Add(added);
    RequireExpected(add, "Add");
    Require(add.value().empty(), "Add reported failed IDs");
    Require(index->GetNumElements() == dataset.base_count,
            "index element count does not match HDF5 train rows");

    std::cout << "[3/6] Running KNN recall validation" << std::endl;
    const auto original_results = RunKnnSearch(index, dataset);
    ValidateSearchDistances(original_results, dataset);
    const float recall = ComputeRecall(original_results, dataset);
    Require(recall >= kMinimumRecall,
            "recall@10 is below " + std::to_string(kMinimumRecall) + ": " +
                std::to_string(recall));

    std::cout << "[4/6] Running range and distance validation" << std::endl;
    ValidateRangeAndDistance(index, dataset);

    std::cout << "[5/6] Serializing and restoring index" << std::endl;
    const auto unique_suffix =
        std::chrono::steady_clock::now().time_since_epoch().count();
    TemporaryFile index_file(std::filesystem::temp_directory_path() /
                             ("vsag-binary-hamming-" + std::to_string(unique_suffix) + ".index"));
    {
        std::ofstream output(index_file.Path(), std::ios::binary);
        Require(output.is_open(), "cannot open temporary index file for writing");
        auto serialize = index->Serialize(output);
        RequireExpected(serialize, "Serialize");
    }
    index.reset();

    auto restored = CreateIndex(build_parameters);
    {
        std::ifstream input(index_file.Path(), std::ios::binary);
        Require(input.is_open(), "cannot open temporary index file for reading");
        auto deserialize = restored->Deserialize(input);
        RequireExpected(deserialize, "Deserialize");
    }

    std::cout << "[6/6] Verifying restored search results" << std::endl;
    const auto restored_results = RunKnnSearch(restored, dataset);
    RequireSameResults(original_results, restored_results);
    return recall;
}

}  // namespace

int
main(int argc, char** argv) {
    H5::Exception::dontPrint();
    try {
        vsag::init();
        const std::string dataset_path = argc > 1 ? argv[1] : kDefaultDatasetPath;
        const float recall = RunEndToEnd(dataset_path);
        std::cout << "PASS: Binary/Hamming HNSW e2e, recall@10=" << recall << std::endl;
        return 0;
    } catch (const H5::Exception& error) {
        std::cerr << "FAIL: HDF5 error: " << error.getDetailMsg() << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
    }
    return 1;
}
