
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

#include "factory/resource_owner_wrapper.h"
#include "index_common_param.h"
#include "unittest.h"
TEST_CASE("IndexCommonParam Basic Test", "[ut][IndexCommonParam][binary][hamming]") {
    std::shared_ptr<vsag::Resource> resource =
        std::make_shared<vsag::ResourceOwnerWrapper>(new vsag::Resource(), true);
    SECTION("wrong metric type") {
        auto build_parameter_json = R"(
        {
            "metric_type": "unknown type",
            "dtype": "float32",
            "dim": 12
        }
        )";
        auto parsed_params = vsag::JsonType::Parse(build_parameter_json);
        REQUIRE_THROWS(vsag::IndexCommonParam::CheckAndCreate(parsed_params, resource));
    }

    SECTION("wrong data type") {
        auto build_parameter_json = R"(
        {
            "metric_type": "l2",
            "dtype": "unknown type",
            "dim": 12
        }
        )";
        auto parsed_params = vsag::JsonType::Parse(build_parameter_json);
        REQUIRE_THROWS(vsag::IndexCommonParam::CheckAndCreate(parsed_params, resource));
    }

    SECTION("wrong dim") {
        auto build_parameter_json = R"(
        {
            "metric_type": "l2",
            "dtype": "float32",
            "dim": -1
        }
        )";
        auto parsed_params = vsag::JsonType::Parse(build_parameter_json);
        REQUIRE_THROWS(vsag::IndexCommonParam::CheckAndCreate(parsed_params, resource));
    }

    SECTION("success") {
        auto build_parameter_json = R"(
        {
            "metric_type": "l2",
            "dtype": "float32",
            "dim": 12,
            "extra_info_size": 38
        }
        )";
        auto parsed_params = vsag::JsonType::Parse(build_parameter_json);
        auto param = vsag::IndexCommonParam::CheckAndCreate(parsed_params, resource);
        REQUIRE(param.metric_ == vsag::MetricType::METRIC_TYPE_L2SQR);
        REQUIRE(param.dim_ == 12);
        REQUIRE(param.extra_info_size_ == 38);
        REQUIRE(param.data_type_ == vsag::DataTypes::DATA_TYPE_FLOAT);
    }

    SECTION("binary hamming success") {
        auto build_parameter_json = R"(
        {
            "metric_type": "hamming",
            "dtype": "binary",
            "dim": 16
        }
        )";
        auto parsed_params = vsag::JsonType::Parse(build_parameter_json);
        auto param = vsag::IndexCommonParam::CheckAndCreate(parsed_params, resource);
        REQUIRE(param.metric_ == vsag::MetricType::METRIC_TYPE_HAMMING);
        REQUIRE(param.data_type_ == vsag::DataTypes::DATA_TYPE_BINARY);
        REQUIRE(param.dim_ == 16);
    }

    SECTION("binary dimension must be byte aligned") {
        auto build_parameter_json = R"(
        {
            "metric_type": "hamming",
            "dtype": "binary",
            "dim": 7
        }
        )";
        auto parsed_params = vsag::JsonType::Parse(build_parameter_json);
        REQUIRE_THROWS(vsag::IndexCommonParam::CheckAndCreate(parsed_params, resource));
    }

    SECTION("binary requires hamming") {
        auto build_parameter_json = R"(
        {
            "metric_type": "l2",
            "dtype": "binary",
            "dim": 16
        }
        )";
        auto parsed_params = vsag::JsonType::Parse(build_parameter_json);
        REQUIRE_THROWS(vsag::IndexCommonParam::CheckAndCreate(parsed_params, resource));
    }

    SECTION("hamming requires binary") {
        auto build_parameter_json = R"(
        {
            "metric_type": "hamming",
            "dtype": "float32",
            "dim": 16
        }
        )";
        auto parsed_params = vsag::JsonType::Parse(build_parameter_json);
        REQUIRE_THROWS(vsag::IndexCommonParam::CheckAndCreate(parsed_params, resource));
    }
}
