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
    "/data/jingyue.zjl/ob_data/L1/mnist-784/mnist-784-l1-1k.hdf5";
constexpr int64_t kQueryCount = 100;
constexpr int64_t kTopK = 100;
constexpr float kMinimumRecall = 0.90F;
constexpr float kFloatTolerance = 1e-4F;

struct L1Dataset {
    int64_t dim;
    int64_t base_count;
    int64_t query_count;
    int64_t truth_k;
    std::vector<int64_t> ids;
    std::vector<float> base;
    std::vector<float> queries;
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

L1Dataset
LoadL1Dataset(const std::string& path) {
    Require(std::filesystem::exists(path), "HDF5 file does not exist: " + path);
    H5::H5File file(path, H5F_ACC_RDONLY);
    Require(ReadStringAttribute(file, "distance") == "manhattan",
            "HDF5 distance attribute must be manhattan");
    Require(ReadStringAttribute(file, "metric") == "l1",
            "HDF5 metric attribute must be l1");

    const auto train_shape = GetMatrixShape(file.openDataSet("train"), "train");
    const auto test_shape = GetMatrixShape(file.openDataSet("test"), "test");
    const auto neighbors_shape = GetMatrixShape(file.openDataSet("neighbors"), "neighbors");
    const auto distances_shape = GetMatrixShape(file.openDataSet("distances"), "distances");

    Require(train_shape.first >= 2, "train must contain at least two rows");
    Require(test_shape.first >= static_cast<uint64_t>(kQueryCount),
            "test must contain at least 100 rows");
    Require(train_shape.second == test_shape.second, "train and test dimensions must match");
    Require(neighbors_shape == distances_shape, "neighbors and distances shapes must match");
    Require(neighbors_shape.first >= static_cast<uint64_t>(kQueryCount),
            "neighbors must contain at least 100 rows");
    Require(neighbors_shape.second >= static_cast<uint64_t>(kTopK),
            "neighbors must contain at least 100 columns");
    Require(train_shape.first <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
            "train row count exceeds int64_t");
    Require(train_shape.second <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
            "train dimension exceeds int64_t");
    Require(neighbors_shape.second <=
                static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
            "ground-truth width exceeds int64_t");

    L1Dataset result{};
    result.dim = static_cast<int64_t>(train_shape.second);
    result.base_count = static_cast<int64_t>(train_shape.first);
    result.query_count = kQueryCount;
    result.truth_k = static_cast<int64_t>(neighbors_shape.second);
    result.ids.resize(train_shape.first);
    std::iota(result.ids.begin(), result.ids.end(), int64_t{0});
    result.base =
        ReadFirstRows<float>(file, "train", train_shape.first, H5::PredType::NATIVE_FLOAT);
    result.queries =
        ReadFirstRows<float>(file, "test", kQueryCount, H5::PredType::NATIVE_FLOAT);
    result.neighbors =
        ReadFirstRows<int64_t>(file, "neighbors", kQueryCount, H5::PredType::NATIVE_INT64);
    result.distances =
        ReadFirstRows<float>(file, "distances", kQueryCount, H5::PredType::NATIVE_FLOAT);
    return result;
}

float
L1Distance(const float* lhs, const float* rhs, uint64_t dim) {
    float distance = 0.0F;
    for (uint64_t i = 0; i < dim; ++i) {
        distance += std::fabs(lhs[i] - rhs[i]);
    }
    return distance;
}

std::string
BuildParameters(int64_t dim) {
    std::ostringstream stream;
    stream << R"({
        "dtype": "float32",
        "metric_type": "l1",
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
MakeQuery(const L1Dataset& dataset, int64_t query_id) {
    return vsag::Dataset::Make()
        ->NumElements(1)
        ->Dim(dataset.dim)
        ->Float32Vectors(dataset.queries.data() +
                         static_cast<uint64_t>(query_id) * dataset.dim)
        ->Owner(false);
}

SearchResults
RunKnnSearch(const vsag::IndexPtr& index, const L1Dataset& dataset) {
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
ComputeRecall(const SearchResults& results, const L1Dataset& dataset) {
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
ValidateSearchDistances(const SearchResults& results, const L1Dataset& dataset) {
    for (int64_t query_id = 0; query_id < dataset.query_count; ++query_id) {
        const auto* query =
            dataset.queries.data() + static_cast<uint64_t>(query_id) * dataset.dim;
        for (int64_t i = 0; i < kTopK; ++i) {
            const uint64_t offset = static_cast<uint64_t>(query_id) * kTopK + i;
            const int64_t id = results.ids[offset];
            Require(id >= 0 and id < dataset.base_count,
                    "KnnSearch returned an out-of-range ID");
            const auto* base =
                dataset.base.data() + static_cast<uint64_t>(id) * dataset.dim;
            const float expected =
                L1Distance(query, base, static_cast<uint64_t>(dataset.dim));
            Require(std::fabs(results.distances[offset] - expected) <= kFloatTolerance,
                    "KnnSearch returned an incorrect L1 distance");
        }
    }
}

void
ValidateRangeAndDistance(const vsag::IndexPtr& index, const L1Dataset& dataset) {
    auto query = MakeQuery(dataset, 0);
    const float radius = dataset.distances[kTopK - 1];
    auto range = index->RangeSearch(query, radius, SearchParameters());
    RequireExpected(range, "RangeSearch");
    Require(range.value()->GetDim() > 0, "RangeSearch returned no results");

    const auto* query_data = dataset.queries.data();
    for (int64_t i = 0; i < range.value()->GetDim(); ++i) {
        const int64_t id = range.value()->GetIds()[i];
        Require(id >= 0 and id < dataset.base_count,
                "RangeSearch returned an out-of-range ID");
        const auto* base =
            dataset.base.data() + static_cast<uint64_t>(id) * dataset.dim;
        const float expected =
            L1Distance(query_data, base, static_cast<uint64_t>(dataset.dim));
        Require(range.value()->GetDistances()[i] <= radius,
                "RangeSearch returned a distance above radius");
        Require(std::fabs(range.value()->GetDistances()[i] - expected) <= kFloatTolerance,
                "RangeSearch returned an incorrect L1 distance");
    }

    const int64_t nearest_id = dataset.neighbors[0];
    auto distance = index->CalcDistanceById(query, nearest_id);
    RequireExpected(distance, "CalcDistanceById");
    const auto* nearest =
        dataset.base.data() + static_cast<uint64_t>(nearest_id) * dataset.dim;
    const float expected =
        L1Distance(query_data, nearest, static_cast<uint64_t>(dataset.dim));
    Require(std::fabs(distance.value() - expected) <= kFloatTolerance,
            "CalcDistanceById returned an incorrect L1 distance");
    Require(std::fabs(distance.value() - dataset.distances[0]) <= kFloatTolerance,
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
    std::cout << "[1/6] Loading HDF5 dataset" << std::endl;
    auto dataset = LoadL1Dataset(dataset_path);
    const auto build_parameters = BuildParameters(dataset.dim);

    std::cout << "[2/6] Building Float32/L1 HNSW with " << dataset.base_count
              << " vectors" << std::endl;
    auto index = CreateIndex(build_parameters);
    auto base = vsag::Dataset::Make()
                    ->NumElements(dataset.base_count - 1)
                    ->Dim(dataset.dim)
                    ->Ids(dataset.ids.data())
                    ->Float32Vectors(dataset.base.data())
                    ->Owner(false);
    auto build = index->Build(base);
    RequireExpected(build, "Build");
    Require(build.value().empty(), "Build reported failed IDs");

    const uint64_t last = static_cast<uint64_t>(dataset.base_count - 1);
    auto added = vsag::Dataset::Make()
                     ->NumElements(1)
                     ->Dim(dataset.dim)
                     ->Ids(dataset.ids.data() + last)
                     ->Float32Vectors(dataset.base.data() + last * dataset.dim)
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
            "recall@100 is below " + std::to_string(kMinimumRecall) + ": " +
                std::to_string(recall));

    std::cout << "[4/6] Running range and distance validation" << std::endl;
    ValidateRangeAndDistance(index, dataset);

    std::cout << "[5/6] Serializing and restoring index" << std::endl;
    const auto unique_suffix =
        std::chrono::steady_clock::now().time_since_epoch().count();
    TemporaryFile index_file(std::filesystem::temp_directory_path() /
                             ("vsag-float32-l1-" + std::to_string(unique_suffix) + ".index"));
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
        std::cout << "PASS: Float32/L1 HNSW e2e, recall@100=" << recall << std::endl;
        return 0;
    } catch (const H5::Exception& error) {
        std::cerr << "FAIL: HDF5 error: " << error.getDetailMsg() << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << std::endl;
    }
    return 1;
}
