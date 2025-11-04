#pragma once

#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <DirectXMath.h>
#include <wrl/client.h>
#include <vector>
#include <string>
#include "Math4D.h"
#include "HypersphereGenerator.h"

using Microsoft::WRL::ComPtr;

// Constant buffer structure
struct SceneConstants
{
    DirectX::XMFLOAT4X4 viewProj;
    float rotation4D[16];  // 4x4 rotation matrix for 4D space
    float projectionDistance;
    float sphereRadius;
    int projectionType;  // 0=perspective, 1=stereographic, 2=orthographic
    int pointCount;
    float time;
    float padding[3];
};

class D3D12Renderer
{
public:
    D3D12Renderer();
    ~D3D12Renderer();

    bool Initialize(HWND hwnd, uint32_t width, uint32_t height);
    void Shutdown();

    void Update(float deltaTime);
    void Render();

    void OnResize(uint32_t width, uint32_t height);

    // Control parameters
    void SetPointCount(uint32_t count);
    void SetProjectionType(Math4D::ProjectionType type);
    void SetRotationSpeeds(float speedWX, float speedWY, float speedWZ);
    void SetProjectionDistance(float distance);

    // Getters
    uint32_t GetWidth() const { return m_width; }
    uint32_t GetHeight() const { return m_height; }
    ID3D12Device* GetDevice() const { return m_device.Get(); }
    ID3D12GraphicsCommandList* GetCommandList() const { return m_commandList.Get(); }

private:
    // Initialization helpers
    bool CreateDevice();
    bool CreateCommandQueue();
    bool CreateSwapChain(HWND hwnd);
    bool CreateDescriptorHeaps();
    bool CreateRenderTargets();
    bool CreateDepthStencil();
    bool CreateRootSignature();
    bool CreatePipelineState();
    bool LoadShaders();
    bool CreatePointBuffers(uint32_t count);
    bool CreateConstantBuffer();
    void CreateViewportScissor();

    // Rendering helpers
    void WaitForGPU();
    void MoveToNextFrame();
    void PopulateCommandList();
    void UpdateConstantBuffer();

    // Window dimensions
    uint32_t m_width;
    uint32_t m_height;
    float m_aspectRatio;

    // DirectX 12 core objects
    ComPtr<ID3D12Device> m_device;
    ComPtr<IDXGISwapChain3> m_swapChain;
    ComPtr<ID3D12CommandQueue> m_commandQueue;
    ComPtr<ID3D12GraphicsCommandList> m_commandList;
    ComPtr<ID3D12CommandAllocator> m_commandAllocators[3];  // Triple buffering

    // Synchronization objects
    ComPtr<ID3D12Fence> m_fence;
    UINT64 m_fenceValues[3];
    UINT m_frameIndex;
    HANDLE m_fenceEvent;

    // Descriptor heaps
    ComPtr<ID3D12DescriptorHeap> m_rtvHeap;
    ComPtr<ID3D12DescriptorHeap> m_dsvHeap;
    ComPtr<ID3D12DescriptorHeap> m_cbvHeap;
    UINT m_rtvDescriptorSize;
    UINT m_dsvDescriptorSize;
    UINT m_cbvDescriptorSize;

    // Render targets
    static constexpr UINT FRAME_COUNT = 3;
    ComPtr<ID3D12Resource> m_renderTargets[FRAME_COUNT];
    ComPtr<ID3D12Resource> m_depthStencil;

    // Pipeline objects
    ComPtr<ID3D12RootSignature> m_rootSignature;
    ComPtr<ID3D12PipelineState> m_pipelineState;

    // Shaders
    ComPtr<ID3DBlob> m_vertexShader;
    ComPtr<ID3DBlob> m_pixelShader;

    // Geometry buffers
    ComPtr<ID3D12Resource> m_pointBuffer;
    D3D12_VERTEX_BUFFER_VIEW m_pointBufferView;
    uint32_t m_pointCount;
    uint32_t m_maxPointCount;

    // Constant buffer
    ComPtr<ID3D12Resource> m_constantBuffer;
    SceneConstants* m_cbData;
    UINT8* m_cbDataBegin;

    // Viewport and scissor
    D3D12_VIEWPORT m_viewport;
    D3D12_RECT m_scissorRect;

    // Scene parameters
    HypersphereGenerator m_hypersphereGen;
    Math4D::ProjectionType m_projectionType;
    float m_projectionDistance;
    float m_rotationSpeedWX;
    float m_rotationSpeedWY;
    float m_rotationSpeedWZ;
    float m_cameraDistance;
    float m_cameraAngleX;
    float m_cameraAngleY;
    float m_time;
};
