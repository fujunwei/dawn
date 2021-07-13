// Copyright 2021 The Tint Authors.
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

#include "fuzzers/tint_ast_fuzzer/mt_rng.h"

#include <cassert>

namespace tint {
namespace fuzzers {
namespace ast_fuzzer {
namespace {

template <typename T>
T RandomUInt(std::mt19937* rng, T bound) {
  assert(bound > 0 && "`bound` must be positive");
  return std::uniform_int_distribution<T>(0, bound - 1)(*rng);
}

}  // namespace

MtRng::MtRng(uint32_t seed) : rng_(seed) {}

uint32_t MtRng::RandomUint32(uint32_t bound) {
  return RandomUInt(&rng_, bound);
}

uint64_t MtRng::RandomUint64(uint64_t bound) {
  return RandomUInt(&rng_, bound);
}

}  // namespace ast_fuzzer
}  // namespace fuzzers
}  // namespace tint
