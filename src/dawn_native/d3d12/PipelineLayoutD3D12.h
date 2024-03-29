// Copyright 2017 The Dawn Authors
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

#ifndef DAWNNATIVE_D3D12_PIPELINELAYOUTD3D12_H_
#define DAWNNATIVE_D3D12_PIPELINELAYOUTD3D12_H_

#include "dawn_native/PipelineLayout.h"

#include "dawn_native/d3d12/d3d12_platform.h"

namespace dawn_native { namespace d3d12 {

    class Device;

    class PipelineLayout : public PipelineLayoutBase {
      public:
        PipelineLayout(Device* device, const PipelineLayoutDescriptor* descriptor);

        uint32_t GetCbvUavSrvRootParameterIndex(uint32_t group) const;
        uint32_t GetSamplerRootParameterIndex(uint32_t group) const;

        // Returns the index of the root parameter reserved for a dynamic buffer binding
        uint32_t GetDynamicRootParameterIndex(uint32_t group, uint32_t binding) const;

        ComPtr<ID3D12RootSignature> GetRootSignature() const;

      private:
        std::array<uint32_t, kMaxBindGroups> mCbvUavSrvRootParameterInfo;
        std::array<uint32_t, kMaxBindGroups> mSamplerRootParameterInfo;
        std::array<std::array<uint32_t, kMaxBindingsPerGroup>, kMaxBindGroups>
            mDynamicRootParameterIndices;
        ComPtr<ID3D12RootSignature> mRootSignature;
    };

}}  // namespace dawn_native::d3d12

#endif  // DAWNNATIVE_D3D12_PIPELINELAYOUTD3D12_H_
