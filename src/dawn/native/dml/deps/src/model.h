//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#include <dawn_native/dawn_native_export.h>
#include <webgpu/webgpu.h>

#pragma once

namespace pydml
{
    struct DAWN_NATIVE_EXPORT CompiledModel
    {
        CompiledModel(
            dml::Graph& graph, 
            DML_EXECUTION_FLAGS flags,
            std::vector<dml::Expression>& outputs
            ) : 
            op(graph.Compile(flags, outputs))
        {}

        Microsoft::WRL::ComPtr<IDMLCompiledOperator> op;
    };

    struct DAWN_NATIVE_EXPORT TensorData
    {
        TensorData(WGPUBuffer buffer,
                   size_t size,
                   size_t offset = 0) :
            buffer(buffer),
            size(size),
            offset(offset) {}

        WGPUBuffer Get() const { return buffer; }

        size_t Size() const { return size; }

        size_t Offset() const { return offset; }

        WGPUBuffer buffer;
        size_t size;
        size_t offset;
    };

    struct DAWN_NATIVE_EXPORT Binding
    {
        explicit Binding(dml::Expression& expression, 
                         WGPUBuffer buffer,
                         size_t size,
                         size_t offset = 0)
            :   exp(expression),
                desc(expression.GetOutputDesc()),
                data(buffer, size, offset)
        {}

        dml::Expression exp;
        dml::TensorDesc desc;
        TensorData data;
    };
}
