//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#include "precomp.h"

#pragma warning(push)
#pragma warning(disable:4238)

using namespace pydml;
using Microsoft::WRL::ComPtr;
using dawn_native::ErrorData;

#define DAWN_TRY_WITH_HRESULT(EXPR)                                                          \
    {                                                                                        \
        auto DAWN_LOCAL_VAR = EXPR;                                                          \
        if (DAWN_UNLIKELY(DAWN_LOCAL_VAR.IsError())) {                                       \
            return (E_FAIL);                                                                 \
        }                                                                                    \
    }                                                                                        \
    for (;;)                                                                                 \
    break

Device::Device(WGPUDevice wgpuDevice, bool useDebugLayer)
    : m_wgpuDevice(wgpuDevice), m_useGpu(true), m_useDebugLayer(useDebugLayer) {
}

HRESULT Device::Init()
{
    // 
    // Create D3D12 resources
    //

    D3D12_DESCRIPTOR_HEAP_DESC descriptorHeapDesc = {};
    descriptorHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    descriptorHeapDesc.NumDescriptors = 4; // One each for the input, output, persistent, and temporary resources
    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    dawn_native::d3d12::Device* dawnDevice = reinterpret_cast<dawn_native::d3d12::Device*>(m_wgpuDevice);
    ReturnIfFailed(dawnDevice->GetD3D12Device()->CreateDescriptorHeap(&descriptorHeapDesc, IID_GRAPHICS_PPV_ARGS(m_clearUavDescriptorHeapCpu.GetAddressOf())));

    descriptorHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ReturnIfFailed(dawnDevice->GetD3D12Device()->CreateDescriptorHeap(&descriptorHeapDesc, IID_GRAPHICS_PPV_ARGS(m_clearUavDescriptorHeapGpu.GetAddressOf())));

    // 
    // Create DML resources
    //

    // DMLCreateDevice1 is supported since DML 1.1.0 and dml_resource_version specified by download_dml.py must exceed this version.
    // TODO: Consider relaxing DML_FEATURE_LEVEL_3_0 requirement to support more hardware.
    if (    !m_useDebugLayer 
        ||  FAILED(DMLCreateDevice1(dawnDevice->GetD3D12Device(), DML_CREATE_DEVICE_FLAG_DEBUG, DML_FEATURE_LEVEL_3_0, IID_PPV_ARGS(&m_dmlDevice))))
    {
        ReturnIfFailed(DMLCreateDevice1(dawnDevice->GetD3D12Device(), DML_CREATE_DEVICE_FLAG_NONE, DML_FEATURE_LEVEL_3_0, IID_PPV_ARGS(&m_dmlDevice)));
    }

    ReturnIfFailed(m_dmlDevice->CreateCommandRecorder(IID_PPV_ARGS(&m_commandRecorder)));
    ReturnIfFailed(m_dmlDevice->CreateOperatorInitializer(0, nullptr, IID_PPV_ARGS(&m_operatorInitializer)));
    ReturnIfFailed(m_dmlDevice->CreateBindingTable(nullptr, IID_PPV_ARGS(&m_bindingTable)));
    return S_OK;
}

HRESULT Device::DispatchOperator(
    IDMLCompiledOperator* op,
    const std::vector<pydml::Binding*>& inputs,
    const std::vector<pydml::Binding*>& outputs
    )
{
    std::vector<DmlBufferBinding> inputBindings(inputs.size());
    uint64_t inputsResourceSize = 0;

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        auto input = inputs[i];

        if (!input)
        {
            continue; // null optional tensor
        }

        DmlBufferTensorDesc desc = *input->desc.AsPtr<DML_BUFFER_TENSOR_DESC>();

        // If OWNED_BY_DML is *not* set, this input must be bound at execution
        // fix clang error: logical not is only applied to the left hand side of this bitwise operator [-Werror,-Wlogical-not-parentheses]
        if (!(desc.flags & DML_TENSOR_FLAG_OWNED_BY_DML))
        {
            uint32_t requiredAlignment = std::max(desc.guaranteedBaseOffsetAlignment, DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);

            // Bind to the end of the inputs resource (with appropriate alignment)
            inputBindings[i].offset = RoundUpToMultiple(inputsResourceSize, (uint64_t)requiredAlignment);
            inputBindings[i].sizeInBytes = desc.totalTensorSizeInBytes;

            inputsResourceSize = inputBindings[i].offset + desc.totalTensorSizeInBytes;
        }
    }

    std::vector<DmlBufferBinding> outputBindings(outputs.size());
    uint64_t outputsResourceSize = 0;

    for (size_t i = 0; i < outputs.size(); ++i)
    {
        auto output = outputs[i];

        if (!output)
        {
            continue; // null optional tensor
        }

        dml::TensorDesc desc = *output->desc.AsPtr<DML_BUFFER_TENSOR_DESC>();
        DmlBufferTensorDesc bufferDesc = *desc.AsPtr<DML_BUFFER_TENSOR_DESC>();

        uint32_t requiredAlignment = std::max(bufferDesc.guaranteedBaseOffsetAlignment, DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);

        // Bind to the end of the outputs resource (with appropriate alignment)
        outputBindings[i].offset = RoundUpToMultiple(outputsResourceSize, (uint64_t)requiredAlignment);
        outputBindings[i].sizeInBytes = bufferDesc.totalTensorSizeInBytes;

        outputsResourceSize = outputBindings[i].offset + outputBindings[i].sizeInBytes;
    }

    DML_BINDING_PROPERTIES bindingProps = op->GetBindingProperties();

    ReturnIfFailed(EnsureCpuOrDefaultBufferSize(inputsResourceSize, m_inputsResource));
    ReturnIfFailed(EnsureCpuOrDefaultBufferSize(outputsResourceSize, m_outputsResource));
    ReturnIfFailed(EnsureDefaultBufferSize(bindingProps.TemporaryResourceSize, m_temporaryResource));
    ReturnIfFailed(EnsureDescriptorHeapSize(bindingProps.RequiredDescriptorCount));

    // Set up input and output bindings to point to their respective buffers
    for (auto& binding : inputBindings)
    {
        if (binding.sizeInBytes != 0)
        {
            binding.buffer = m_inputsResource.Get();
        }
    }

    for (auto& binding : outputBindings)
    {
        if (binding.sizeInBytes != 0)
        {
            binding.buffer = m_outputsResource.Get();
        }
    }

    // The persistent resource should have already been initialized when the operator was initialized
    assert(m_persistentResource.Get()->GetDesc().Width >= bindingProps.PersistentResourceSize);

    // Upload inputs for execution
    std::vector<ID3D12Resource*> buffersToClear =
    {
        m_inputsResource.Get(),
        m_temporaryResource.Get(),
        m_outputsResource.Get()
    };

    dawn_native::d3d12::Device* dawnDevice = reinterpret_cast<dawn_native::d3d12::Device*>(m_wgpuDevice);
    dawn_native::d3d12::CommandRecordingContext* commandRecordingContext;
    DAWN_TRY_ASSIGN_WITH_CLEANUP(commandRecordingContext, dawnDevice->GetPendingCommandContext(), {}, E_FAIL);
    ID3D12GraphicsCommandList* commandList = commandRecordingContext->GetCommandList();

    ReturnIfFailed(ClearGpuBuffers(commandList, buffersToClear));

    if (inputsResourceSize)
    {
        commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_inputsResource.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_DEST)
            );

        for (size_t i = 0; i < inputs.size(); ++i)
        {
            if (!inputBindings[i].buffer)
            {
                // This input tensor doesn't need to be bound for initialize
                continue;
            }

            DmlBufferTensorDesc bufferDesc = *inputs[i]->desc.AsPtr<DML_BUFFER_TENSOR_DESC>();
            assert(inputs[i]->data.Size() == bufferDesc.totalTensorSizeInBytes);

            WGPUBuffer wgpuBuffer = inputs[i]->data.Get();
            dawn_native::d3d12::Buffer* inputBuffer = reinterpret_cast<dawn_native::d3d12::Buffer*>(wgpuBuffer);
            DAWN_TRY_WITH_HRESULT(inputBuffer->EnsureDataInitialized(commandRecordingContext));
            inputBuffer->TrackUsageAndTransitionNow(commandRecordingContext, wgpu::BufferUsage::CopySrc);

            commandList->CopyBufferRegion(
                m_inputsResource.Get(), inputBindings[i].offset,
                inputBuffer->GetD3D12Resource(), inputs[i]->data.Offset(),
                inputs[i]->data.Size());
        }

        commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_inputsResource.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            );
    }

    // Bind for execution
    DmlTypeConverter<1024> converter;

    DML_BINDING_TABLE_DESC bindingTableDesc = {};
    bindingTableDesc.Dispatchable = op;
    bindingTableDesc.CPUDescriptorHandle = m_descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    bindingTableDesc.GPUDescriptorHandle = m_descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    bindingTableDesc.SizeInDescriptors = bindingProps.RequiredDescriptorCount;

    ReturnIfFailed(m_bindingTable->Reset(&bindingTableDesc));

    // Bind inputs
    std::vector<DML_BINDING_DESC> inputBindingDescs(inputBindings.size());
    for (size_t i = 0; i < inputBindings.size(); ++i)
    {
        inputBindingDescs[i] = converter.ToBindingDesc(inputBindings[i]);
    }

    m_bindingTable->BindInputs(static_cast<uint32_t>(inputBindingDescs.size()), inputBindingDescs.data());

    // Bind outputs
    std::vector<DML_BINDING_DESC> outputBindingDescs(outputBindings.size());
    for (size_t i = 0; i < outputBindings.size(); ++i)
    {
        outputBindingDescs[i] = converter.ToBindingDesc(outputBindings[i]);
    }

    m_bindingTable->BindOutputs(static_cast<uint32_t>(outputBindingDescs.size()), outputBindingDescs.data());

    // Bind persistent/temporary resources
    if (bindingProps.PersistentResourceSize != 0)
    {
        DML_BUFFER_BINDING persistentBinding = { m_persistentResource.Get(), 0, bindingProps.PersistentResourceSize };
        auto bindingDesc = DML_BINDING_DESC { DML_BINDING_TYPE_BUFFER, &persistentBinding };
        m_bindingTable->BindPersistentResource(&bindingDesc);
    }

    if (bindingProps.TemporaryResourceSize != 0)
    {
        DML_BUFFER_BINDING temporaryBinding = { m_temporaryResource.Get(), 0, bindingProps.TemporaryResourceSize };
        auto bindingDesc = DML_BINDING_DESC { DML_BINDING_TYPE_BUFFER, &temporaryBinding };
        m_bindingTable->BindTemporaryResource(&bindingDesc);
    }

    // Record and execute commands, and wait for completion
    commandList->SetDescriptorHeaps(1, m_descriptorHeap.GetAddressOf());
    m_commandRecorder->RecordDispatch(commandList, op, m_bindingTable.Get());

    if (outputsResourceSize != 0)
    {
        commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_outputsResource.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE)
            );

        for (size_t i = 0; i < outputs.size(); ++i)
        {
            auto output = outputs[i];

            if (!output)
            {
                // This output tensor is optional (and null)
                continue;
            }

            dml::TensorDesc desc = *output->desc.AsPtr<DML_BUFFER_TENSOR_DESC>();
            DmlBufferTensorDesc bufferDesc = *desc.AsPtr<DML_BUFFER_TENSOR_DESC>();
            assert(output->data.Size() == bufferDesc.totalTensorSizeInBytes);

            WGPUBuffer wgpuBuffer = output->data.Get();
            dawn_native::d3d12::Buffer* outputBuffer = reinterpret_cast<dawn_native::d3d12::Buffer*>(wgpuBuffer);
            bool cleared;
            DAWN_TRY_ASSIGN_WITH_CLEANUP(cleared,
                            outputBuffer->EnsureDataInitializedAsDestination(
                                commandRecordingContext, output->data.Offset(), outputBindings[i].sizeInBytes), {}, E_FAIL);
            DAWN_UNUSED(cleared);
            outputBuffer->TrackUsageAndTransitionNow(commandRecordingContext, wgpu::BufferUsage::CopyDst);

            commandList->CopyBufferRegion(
                outputBuffer->GetD3D12Resource(), output->data.Offset(),
                m_outputsResource.Get(), outputBindings[i].offset, outputBindings[i].sizeInBytes);
        }

        commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_outputsResource.Get(),
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
            );
    }

    DAWN_TRY_WITH_HRESULT(commandRecordingContext->ExecuteCommandList(dawnDevice));
    DAWN_TRY_WITH_HRESULT(dawnDevice->NextSerial());
    return S_OK;
}

HRESULT Device::InitializeOperator(
    IDMLCompiledOperator* op,
    const std::vector<pydml::Binding*>& inputs
    )
{
    // Allocate resources for initialization
    ReturnIfFailed(m_operatorInitializer->Reset(1, &op));

    DmlBufferArrayBinding inputBinding;
    inputBinding.bindings.resize(inputs.size());

    // Fill in the offsets and sizes for each binding, which will also tell us how big we need to make our buffer
    uint64_t inputsResourceSize = 0;

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        auto input = inputs[i];

        if (!input)
        {
            continue; // null optional tensor
        }

        DmlBufferTensorDesc bufferDesc = *input->desc.AsPtr<DML_BUFFER_TENSOR_DESC>();

        // If OWNED_BY_DML is set, this input must be bound at initialize
        if (bufferDesc.flags & DML_TENSOR_FLAG_OWNED_BY_DML)
        {
            uint32_t requiredAlignment = std::max(bufferDesc.guaranteedBaseOffsetAlignment, DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT);

            // Bind to the end of the inputs resource (with appropriate alignment)
            inputBinding.bindings[i].offset = RoundUpToMultiple(inputsResourceSize, (uint64_t)requiredAlignment);
            inputBinding.bindings[i].sizeInBytes = bufferDesc.totalTensorSizeInBytes;

            inputsResourceSize = inputBinding.bindings[i].offset + bufferDesc.totalTensorSizeInBytes;
        }
    }

    uint64_t temporaryResourceSize = m_operatorInitializer->GetBindingProperties().TemporaryResourceSize;
    uint64_t persistentResourceSize = op->GetBindingProperties().PersistentResourceSize;
    uint32_t descriptorHeapSize = m_operatorInitializer->GetBindingProperties().RequiredDescriptorCount;

    ReturnIfFailed(EnsureCpuOrDefaultBufferSize(inputsResourceSize, m_inputsResource));
    ReturnIfFailed(EnsureDefaultBufferSize(temporaryResourceSize, m_temporaryResource));
    ReturnIfFailed(EnsureDefaultBufferSize(persistentResourceSize, m_persistentResource));
    ReturnIfFailed(EnsureDescriptorHeapSize(descriptorHeapSize));

    // Set up the bindings to point to our input resource
    for (auto& binding : inputBinding.bindings)
    {
        if (binding.sizeInBytes != 0)
        {
            binding.buffer = m_inputsResource.Get();
        }
    }

    // Upload inputs for initialization
    std::vector<ID3D12Resource*> buffersToClear =
    {
        m_inputsResource.Get(),
        m_temporaryResource.Get(),
        m_persistentResource.Get()
    };

    dawn_native::d3d12::Device* dawnDevice = reinterpret_cast<dawn_native::d3d12::Device*>(m_wgpuDevice);
    dawn_native::d3d12::CommandRecordingContext* commandRecordingContext;
    DAWN_TRY_ASSIGN_WITH_CLEANUP(commandRecordingContext, dawnDevice->GetPendingCommandContext(), {}, E_FAIL);
    ID3D12GraphicsCommandList* commandList = commandRecordingContext->GetCommandList();

    ReturnIfFailed(ClearGpuBuffers(commandList, buffersToClear));

    if (inputsResourceSize)
    {
        commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_inputsResource.Get(),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_DEST)
            );

        for (size_t i = 0; i < inputs.size(); ++i)
        {
            if (!inputBinding.bindings[i].buffer)
            {
                // This input tensor doesn't need to be bound for initialize
                continue;
            }

            DmlBufferTensorDesc bufferDesc = *inputs[i]->desc.AsPtr<DML_BUFFER_TENSOR_DESC>();
            assert(inputs[i]->data.Size() == bufferDesc.totalTensorSizeInBytes);

            WGPUBuffer wgpuBuffer = inputs[i]->data.Get();
            dawn_native::d3d12::Buffer* inputBuffer = reinterpret_cast<dawn_native::d3d12::Buffer*>(wgpuBuffer);
            DAWN_TRY_WITH_HRESULT(inputBuffer->EnsureDataInitialized(commandRecordingContext));
            inputBuffer->TrackUsageAndTransitionNow(commandRecordingContext, wgpu::BufferUsage::CopySrc);

            commandList->CopyBufferRegion(
                m_inputsResource.Get(), inputBinding.bindings[i].offset,
                inputBuffer->GetD3D12Resource(), inputs[i]->data.Offset(),
                inputs[i]->data.Size());
        }

        commandList->ResourceBarrier(
            1,
            &CD3DX12_RESOURCE_BARRIER::Transition(
                m_inputsResource.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
                );
    }

    // Bind for initialization
    DmlTypeConverter<1024> converter;

    DML_BINDING_TABLE_DESC bindingTableDesc = {};
    bindingTableDesc.Dispatchable = m_operatorInitializer.Get();
    bindingTableDesc.CPUDescriptorHandle = m_descriptorHeap->GetCPUDescriptorHandleForHeapStart();
    bindingTableDesc.GPUDescriptorHandle = m_descriptorHeap->GetGPUDescriptorHandleForHeapStart();
    bindingTableDesc.SizeInDescriptors = descriptorHeapSize;

    ReturnIfFailed(m_bindingTable->Reset(&bindingTableDesc));

    DML_BINDING_DESC inputBindingDesc = converter.ToBindingDesc(inputBinding);
    m_bindingTable->BindInputs(1, &inputBindingDesc);

    if (persistentResourceSize != 0)
    {
        DML_BUFFER_BINDING outputBinding = { m_persistentResource.Get(), 0, persistentResourceSize };
        auto desc = DML_BINDING_DESC { DML_BINDING_TYPE_BUFFER, &outputBinding };
        m_bindingTable->BindOutputs(1, &desc);
    }

    if (temporaryResourceSize != 0)
    {
        DML_BUFFER_BINDING temporaryBinding = { m_temporaryResource.Get(), 0, temporaryResourceSize };
        auto desc = DML_BINDING_DESC { DML_BINDING_TYPE_BUFFER, &temporaryBinding };
        m_bindingTable->BindTemporaryResource(&desc);
    }

    // Record and execute commands, and wait for completion
    commandList->SetDescriptorHeaps(1, m_descriptorHeap.GetAddressOf());
    m_commandRecorder->RecordDispatch(commandList, m_operatorInitializer.Get(), m_bindingTable.Get());
    
    DAWN_TRY_WITH_HRESULT(commandRecordingContext->ExecuteCommandList(dawnDevice));
    DAWN_TRY_WITH_HRESULT(dawnDevice->NextSerial());
    return S_OK;
}

HRESULT Device::EnsureCpuOrDefaultBufferSize(uint64_t requestedSizeInBytes, _Inout_ ComPtr<ID3D12Resource>& buffer)
{
    if (m_useCpuCustomHeapResources)
    {
        ReturnIfFailed(EnsureCpuBufferSize(requestedSizeInBytes, buffer));
    }
    else
    {
        ReturnIfFailed(EnsureDefaultBufferSize(requestedSizeInBytes, buffer));
    }
    return S_OK;
}

HRESULT Device::EnsureCpuBufferSize(uint64_t requestedSizeInBytes, _Inout_ ComPtr<ID3D12Resource>& buffer)
{
    uint64_t existingSize = buffer ? buffer.Get()->GetDesc().Width : 0;
    uint64_t newSize = RoundUpToPow2(requestedSizeInBytes);     // ensures geometric growth
    newSize = std::max(newSize, static_cast<uint64_t>(65536));  // Minimum size of 64k

    if (newSize != existingSize)
    {
        buffer = nullptr;
        dawn_native::d3d12::Device* dawnDevice = reinterpret_cast<dawn_native::d3d12::Device*>(m_wgpuDevice);
        ReturnIfFailed(CreateCpuCustomBuffer(dawnDevice->GetD3D12Device(), newSize, buffer));
    }
    
    return S_OK;
}

HRESULT Device::EnsureDefaultBufferSize(uint64_t requestedSizeInBytes, _Inout_ ComPtr<ID3D12Resource>& buffer)
{
    uint64_t existingSize = buffer ? buffer.Get()->GetDesc().Width : 0;
    uint64_t newSize = RoundUpToPow2(requestedSizeInBytes);     // ensures geometric growth
    newSize = std::max(newSize, static_cast<uint64_t>(65536));  // Minimum size of 64k

    if (newSize != existingSize)
    {
        buffer = nullptr;
        dawn_native::d3d12::Device* dawnDevice = reinterpret_cast<dawn_native::d3d12::Device*>(m_wgpuDevice);
        ReturnIfFailed(CreateDefaultBuffer(dawnDevice->GetD3D12Device(), newSize, buffer));
    }

    return S_OK;
}

HRESULT Device::EnsureDescriptorHeapSize(uint32_t requestedSizeInDescriptors)
{
    uint32_t existingSize = m_descriptorHeap ? m_descriptorHeap->GetDesc().NumDescriptors : 0;
    uint32_t newSize = RoundUpToPow2(requestedSizeInDescriptors); // ensures geometric growth

    if (newSize != existingSize)
    {
        m_descriptorHeap = nullptr;
        
        D3D12_DESCRIPTOR_HEAP_DESC desc = {};
        desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        desc.NumDescriptors = newSize;
        desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;

        dawn_native::d3d12::Device* dawnDevice = reinterpret_cast<dawn_native::d3d12::Device*>(m_wgpuDevice);
        ReturnIfFailed(dawnDevice->GetD3D12Device()->CreateDescriptorHeap(&desc, IID_GRAPHICS_PPV_ARGS(m_descriptorHeap.GetAddressOf())));
    }
    return S_OK;
}

HRESULT Device::ClearGpuBuffers(ID3D12GraphicsCommandList* commandList, dml::Span<ID3D12Resource*> buffers)
{
    static const uint32_t ClearValue = static_cast<uint32_t>(-1);

    // The number of buffers we can clear at once is limited by the size of our descriptor heap
    assert(static_cast<uint32_t>(buffers.size()) <= m_clearUavDescriptorHeapCpu->GetDesc().NumDescriptors);

    uint32_t descriptorOffset = 0;
    for (ID3D12Resource* buffer : buffers)
    {
        if (!buffer)
        {
            // Nothing to clear; these buffers are lazily-initialized
            continue;
        }

        ReturnIfFailed(FillGpuBuffer(
            commandList,
            m_clearUavDescriptorHeapCpu.Get(),
            m_clearUavDescriptorHeapGpu.Get(),
            descriptorOffset,
            buffer,
            ClearValue));

        ++descriptorOffset;
    }

    commandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(nullptr));
    return S_OK;
}

#pragma warning(pop)