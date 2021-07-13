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

#ifndef FUZZERS_TINT_AST_FUZZER_PROTOBUFS_TINT_AST_FUZZER_H_
#define FUZZERS_TINT_AST_FUZZER_PROTOBUFS_TINT_AST_FUZZER_H_

// Compilation of the protobuf library and its autogenerated code can produce
// warnings. Ignore them since we can't control them.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wreserved-id-macro"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
#pragma clang diagnostic ignored "-Wextra-semi-stmt"
#pragma clang diagnostic ignored "-Winconsistent-missing-destructor-override"
#pragma clang diagnostic ignored "-Wweak-vtables"

#include "fuzzers/tint_ast_fuzzer/protobufs/tint_ast_fuzzer.pb.h"

#pragma clang diagnostic pop

#endif  // FUZZERS_TINT_AST_FUZZER_PROTOBUFS_TINT_AST_FUZZER_H_
