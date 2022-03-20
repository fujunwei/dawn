//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------


#include "model.h"

#include <dawn_native/dawn_native_export.h>
#include <webgpu/webgpu.h>

// #pragma once
namespace pydml
{
    class DAWN_NATIVE_EXPORT Device
    {
    public:
        explicit Device(WGPUDevice wgpuDevice, bool useDebugLayer = false);

        HRESULT Init();

        inline bool UseGpu() const
        {
            return m_useGpu;
        }

        inline IDMLDevice* GetDevice() const
        {
            return m_dmlDevice.Get();
        }

        HRESULT InitializeOperator(
            IDMLCompiledOperator* op,
            const std::vector<pydml::Binding*>& inputs
            );

        HRESULT DispatchOperator(
            IDMLCompiledOperator* op,
            const std::vector<pydml::Binding*>& inputs,
            const std::vector<pydml::Binding*>& outputs
            );

    protected:
        void RecordOutputReadBack(uint64_t outputsResourceSize);

        // HRESULT DownloadFromReadBackHeap(
        //     uint64_t outputsResourceSize, 
        //     const std::vector<dml::Expression*>& outputs,
        //     const std::vector<DmlBufferBinding>& outputBindings,
        //     std::vector<pydml::TensorData*>& outputData
        //     );

        HRESULT EnsureCpuOrDefaultBufferSize(uint64_t requestedSizeInBytes, _Inout_ Microsoft::WRL::ComPtr<ID3D12Resource>& buffer);
        HRESULT EnsureCpuBufferSize(uint64_t requestedSizeInBytes, _Inout_ Microsoft::WRL::ComPtr<ID3D12Resource>& buffer);
        HRESULT EnsureDefaultBufferSize(uint64_t requestedSizeInBytes, _Inout_ Microsoft::WRL::ComPtr<ID3D12Resource>& buffer);
        HRESULT EnsureDescriptorHeapSize(uint32_t requestedSizeInDescriptors);

        HRESULT ClearGpuBuffers(ID3D12GraphicsCommandList* commandList, dml::Span<ID3D12Resource*> buffers);

        WGPUDevice m_wgpuDevice;

        // GPU- and CPU-visible descriptor heaps used for ClearUnorderedAccessView
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_clearUavDescriptorHeapGpu;
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_clearUavDescriptorHeapCpu;

        Microsoft::WRL::ComPtr<IDMLDevice> m_dmlDevice;
        Microsoft::WRL::ComPtr<IDMLCommandRecorder> m_commandRecorder;
        Microsoft::WRL::ComPtr<IDMLOperatorInitializer> m_operatorInitializer;
        Microsoft::WRL::ComPtr<IDMLBindingTable> m_bindingTable;

        // Lazily-initialized resources for operator initialization/execution
        Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_descriptorHeap;

        // DEFAULT heap buffers to hold input tensors, output tensors, and temporary and persistent resources. The input
        // and output resources are suballocated for operators that have multiple inputs or outputs.
        Microsoft::WRL::ComPtr<ID3D12Resource> m_inputsResource;
        Microsoft::WRL::ComPtr<ID3D12Resource> m_outputsResource;
        Microsoft::WRL::ComPtr<ID3D12Resource> m_temporaryResource;
        Microsoft::WRL::ComPtr<ID3D12Resource> m_persistentResource;

        bool m_useCpuCustomHeapResources = false;
        bool m_useGpu = true;
        bool m_useDebugLayer = false;
    };
}