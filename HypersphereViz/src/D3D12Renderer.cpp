#include "../include/D3D12Renderer.h"
#include "../include/d3dx12.h"
#include <stdexcept>
#include <vector>
#include <fstream>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

using namespace DirectX;

// Helper function to check HRESULT
#define ThrowIfFailed(x) \
{ \
    HRESULT hr = (x); \
    if (FAILED(hr)) { \
        throw std::runtime_error("DirectX 12 Error"); \
    } \
}

D3D12Renderer::D3D12Renderer()
    : m_width(1920)
    , m_height(1080)
    , m_aspectRatio(0.0f)
    , m_frameIndex(0)
    , m_fenceEvent(nullptr)
    , m_rtvDescriptorSize(0)
    , m_dsvDescriptorSize(0)
    , m_cbvDescriptorSize(0)
    , m_cbData(nullptr)
    , m_cbDataBegin(nullptr)
    , m_pointCount(100000)
    , m_maxPointCount(10000000)
    , m_projectionType(Math4D::ProjectionType::Perspective)
    , m_projectionDistance(2.5f)
    , m_rotationSpeedWX(0.5f)
    , m_rotationSpeedWY(0.3f)
    , m_rotationSpeedWZ(0.7f)
    , m_cameraDistance(5.0f)
    , m_cameraAngleX(0.5f)  // ~30 degrees azimuth
    , m_cameraAngleY(1.0f)  // ~60 degrees elevation (PI/3)
    , m_time(0.0f)
{
    for (UINT i = 0; i < FRAME_COUNT; i++)
    {
        m_fenceValues[i] = 0;
    }
}

D3D12Renderer::~D3D12Renderer()
{
    Shutdown();
}

bool D3D12Renderer::Initialize(HWND hwnd, uint32_t width, uint32_t height)
{
    m_width = width;
    m_height = height;
    m_aspectRatio = static_cast<float>(width) / static_cast<float>(height);

    try
    {
        if (!CreateDevice()) return false;
        if (!CreateCommandQueue()) return false;
        if (!CreateSwapChain(hwnd)) return false;
        if (!CreateDescriptorHeaps()) return false;
        if (!CreateRenderTargets()) return false;
        if (!CreateDepthStencil()) return false;
        if (!CreateRootSignature()) return false;
        if (!LoadShaders()) return false;
        if (!CreatePipelineState()) return false;
        if (!CreatePointBuffers(m_pointCount)) return false;
        if (!CreateConstantBuffer()) return false;
        CreateViewportScissor();

        // Initialize hypersphere generator
        std::vector<Point4D> points;
        const float shellRadius = 1.0f;
        const float shellThickness = 0.02f;  // 2% of radius
        m_hypersphereGen.GeneratePointsCPU(points, m_pointCount, shellRadius, shellThickness);

        // Upload point data to GPU
        D3D12_SUBRESOURCE_DATA vertexData = {};
        vertexData.pData = points.data();
        vertexData.RowPitch = points.size() * sizeof(Point4D);
        vertexData.SlicePitch = vertexData.RowPitch;

        // We need an upload heap
        ComPtr<ID3D12Resource> uploadBuffer;
        CD3DX12_HEAP_PROPERTIES uploadHeapProps(D3D12_HEAP_TYPE_UPLOAD);
        CD3DX12_RESOURCE_DESC uploadBufferDesc = CD3DX12_RESOURCE_DESC::Buffer(vertexData.SlicePitch);

        ThrowIfFailed(m_device->CreateCommittedResource(
            &uploadHeapProps,
            D3D12_HEAP_FLAG_NONE,
            &uploadBufferDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&uploadBuffer)));

        // Copy data to upload buffer
        UINT8* pVertexDataBegin;
        CD3DX12_RANGE readRange(0, 0);
        ThrowIfFailed(uploadBuffer->Map(0, &readRange, reinterpret_cast<void**>(&pVertexDataBegin)));
        memcpy(pVertexDataBegin, vertexData.pData, vertexData.SlicePitch);
        uploadBuffer->Unmap(0, nullptr);

        // Copy from upload heap to default heap
        m_commandList->CopyBufferRegion(m_pointBuffer.Get(), 0, uploadBuffer.Get(), 0, vertexData.SlicePitch);

        // Transition point buffer to vertex buffer state
        CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
            m_pointBuffer.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
        m_commandList->ResourceBarrier(1, &barrier);

        // Execute command list and wait
        ThrowIfFailed(m_commandList->Close());
        ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
        m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
        WaitForGPU();

        return true;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

void D3D12Renderer::Shutdown()
{
    WaitForGPU();

    if (m_fenceEvent)
    {
        CloseHandle(m_fenceEvent);
        m_fenceEvent = nullptr;
    }
}

bool D3D12Renderer::CreateDevice()
{
    UINT dxgiFactoryFlags = 0;

#if defined(_DEBUG)
    ComPtr<ID3D12Debug> debugController;
    if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController))))
    {
        debugController->EnableDebugLayer();
        dxgiFactoryFlags |= DXGI_CREATE_FACTORY_DEBUG;
    }
#endif

    ComPtr<IDXGIFactory4> factory;
    ThrowIfFailed(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&factory)));

    // Try to create hardware device
    ComPtr<IDXGIAdapter1> hardwareAdapter;
    for (UINT adapterIndex = 0; DXGI_ERROR_NOT_FOUND != factory->EnumAdapters1(adapterIndex, &hardwareAdapter); ++adapterIndex)
    {
        DXGI_ADAPTER_DESC1 desc;
        hardwareAdapter->GetDesc1(&desc);

        if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            continue;

        if (SUCCEEDED(D3D12CreateDevice(hardwareAdapter.Get(), D3D_FEATURE_LEVEL_12_0, _uuidof(ID3D12Device), nullptr)))
            break;
    }

    if (hardwareAdapter)
    {
        ThrowIfFailed(D3D12CreateDevice(hardwareAdapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&m_device)));
    }
    else
    {
        ComPtr<IDXGIAdapter> warpAdapter;
        ThrowIfFailed(factory->EnumWarpAdapter(IID_PPV_ARGS(&warpAdapter)));
        ThrowIfFailed(D3D12CreateDevice(warpAdapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&m_device)));
    }

    return true;
}

bool D3D12Renderer::CreateCommandQueue()
{
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

    ThrowIfFailed(m_device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&m_commandQueue)));

    // Create command allocators
    for (UINT i = 0; i < FRAME_COUNT; i++)
    {
        ThrowIfFailed(m_device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&m_commandAllocators[i])));
    }

    // Create command list
    ThrowIfFailed(m_device->CreateCommandList(
        0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        m_commandAllocators[0].Get(),
        nullptr,
        IID_PPV_ARGS(&m_commandList)));

    // Create fence
    ThrowIfFailed(m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_fence)));
    m_fenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    if (!m_fenceEvent)
        return false;

    return true;
}

bool D3D12Renderer::CreateSwapChain(HWND hwnd)
{
    ComPtr<IDXGIFactory4> factory;
    ThrowIfFailed(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)));

    DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
    swapChainDesc.BufferCount = FRAME_COUNT;
    swapChainDesc.Width = m_width;
    swapChainDesc.Height = m_height;
    swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    swapChainDesc.SampleDesc.Count = 1;

    ComPtr<IDXGISwapChain1> swapChain;
    ThrowIfFailed(factory->CreateSwapChainForHwnd(
        m_commandQueue.Get(),
        hwnd,
        &swapChainDesc,
        nullptr,
        nullptr,
        &swapChain));

    ThrowIfFailed(factory->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER));
    ThrowIfFailed(swapChain.As(&m_swapChain));
    m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

    return true;
}

bool D3D12Renderer::CreateDescriptorHeaps()
{
    // RTV heap
    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc = {};
    rtvHeapDesc.NumDescriptors = FRAME_COUNT;
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    ThrowIfFailed(m_device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&m_rtvHeap)));
    m_rtvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    // DSV heap
    D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc = {};
    dsvHeapDesc.NumDescriptors = 1;
    dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
    dsvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
    ThrowIfFailed(m_device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&m_dsvHeap)));
    m_dsvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);

    // CBV heap
    D3D12_DESCRIPTOR_HEAP_DESC cbvHeapDesc = {};
    cbvHeapDesc.NumDescriptors = FRAME_COUNT;
    cbvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    cbvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ThrowIfFailed(m_device->CreateDescriptorHeap(&cbvHeapDesc, IID_PPV_ARGS(&m_cbvHeap)));
    m_cbvDescriptorSize = m_device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

    return true;
}

bool D3D12Renderer::CreateRenderTargets()
{
    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_rtvHeap->GetCPUDescriptorHandleForHeapStart());

    for (UINT i = 0; i < FRAME_COUNT; i++)
    {
        ThrowIfFailed(m_swapChain->GetBuffer(i, IID_PPV_ARGS(&m_renderTargets[i])));
        m_device->CreateRenderTargetView(m_renderTargets[i].Get(), nullptr, rtvHandle);
        rtvHandle.Offset(1, m_rtvDescriptorSize);
    }

    return true;
}

bool D3D12Renderer::CreateDepthStencil()
{
    D3D12_DEPTH_STENCIL_VIEW_DESC depthStencilDesc = {};
    depthStencilDesc.Format = DXGI_FORMAT_D32_FLOAT;
    depthStencilDesc.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    depthStencilDesc.Flags = D3D12_DSV_FLAG_NONE;

    D3D12_CLEAR_VALUE depthOptimizedClearValue = {};
    depthOptimizedClearValue.Format = DXGI_FORMAT_D32_FLOAT;
    depthOptimizedClearValue.DepthStencil.Depth = 1.0f;
    depthOptimizedClearValue.DepthStencil.Stencil = 0;

    CD3DX12_HEAP_PROPERTIES depthHeapProps(D3D12_HEAP_TYPE_DEFAULT);
    CD3DX12_RESOURCE_DESC depthResourceDesc = CD3DX12_RESOURCE_DESC::Tex2D(
        DXGI_FORMAT_D32_FLOAT,
        m_width,
        m_height,
        1, 1, 1, 0,
        D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);

    ThrowIfFailed(m_device->CreateCommittedResource(
        &depthHeapProps,
        D3D12_HEAP_FLAG_NONE,
        &depthResourceDesc,
        D3D12_RESOURCE_STATE_DEPTH_WRITE,
        &depthOptimizedClearValue,
        IID_PPV_ARGS(&m_depthStencil)));

    m_device->CreateDepthStencilView(
        m_depthStencil.Get(),
        &depthStencilDesc,
        m_dsvHeap->GetCPUDescriptorHandleForHeapStart());

    return true;
}

bool D3D12Renderer::CreateRootSignature()
{
    CD3DX12_DESCRIPTOR_RANGE1 ranges[1];
    ranges[0].Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0, 0, D3D12_DESCRIPTOR_RANGE_FLAG_DATA_STATIC);

    CD3DX12_ROOT_PARAMETER1 rootParameters[1];
    rootParameters[0].InitAsDescriptorTable(1, &ranges[0], D3D12_SHADER_VISIBILITY_ALL);

    D3D12_ROOT_SIGNATURE_FLAGS rootSignatureFlags =
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS |
        D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS;

    CD3DX12_VERSIONED_ROOT_SIGNATURE_DESC rootSignatureDesc;
    rootSignatureDesc.Init_1_1(_countof(rootParameters), rootParameters, 0, nullptr, rootSignatureFlags);

    ComPtr<ID3DBlob> signature;
    ComPtr<ID3DBlob> error;
    ThrowIfFailed(D3DX12SerializeVersionedRootSignature(&rootSignatureDesc, D3D_ROOT_SIGNATURE_VERSION_1_1, &signature, &error));
    ThrowIfFailed(m_device->CreateRootSignature(0, signature->GetBufferPointer(), signature->GetBufferSize(), IID_PPV_ARGS(&m_rootSignature)));

    return true;
}

bool D3D12Renderer::LoadShaders()
{
    // For this demo, we'll compile shaders at runtime
    // In production, you should compile them offline and load .cso files

    UINT compileFlags = D3DCOMPILE_DEBUG | D3DCOMPILE_SKIP_OPTIMIZATION;

    // Read shader files
    auto readFile = [](const char* filename) -> std::string {
        std::ifstream file(filename);
        return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    };

    std::string vsSource = readFile("shaders/VertexShader.hlsl");
    std::string psSource = readFile("shaders/PixelShader.hlsl");

    ComPtr<ID3DBlob> error;

    HRESULT hr = D3DCompile(vsSource.c_str(), vsSource.size(), nullptr, nullptr, nullptr,
        "main", "vs_5_1", compileFlags, 0, &m_vertexShader, &error);
    if (FAILED(hr) && error)
    {
        OutputDebugStringA((char*)error->GetBufferPointer());
        return false;
    }

    hr = D3DCompile(psSource.c_str(), psSource.size(), nullptr, nullptr, nullptr,
        "main", "ps_5_1", compileFlags, 0, &m_pixelShader, &error);
    if (FAILED(hr) && error)
    {
        OutputDebugStringA((char*)error->GetBufferPointer());
        return false;
    }

    return true;
}

bool D3D12Renderer::CreatePipelineState()
{
    // Define input layout
    D3D12_INPUT_ELEMENT_DESC inputElementDescs[] =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32_UINT, 0, 16, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 1, DXGI_FORMAT_R32G32B32_FLOAT, 0, 20, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = {};
    psoDesc.InputLayout = { inputElementDescs, _countof(inputElementDescs) };
    psoDesc.pRootSignature = m_rootSignature.Get();
    psoDesc.VS = CD3DX12_SHADER_BYTECODE(m_vertexShader.Get());
    psoDesc.PS = CD3DX12_SHADER_BYTECODE(m_pixelShader.Get());
    psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    psoDesc.SampleDesc.Count = 1;

    ThrowIfFailed(m_device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&m_pipelineState)));

    return true;
}

bool D3D12Renderer::CreatePointBuffers(uint32_t count)
{
    const UINT64 bufferSize = count * sizeof(Point4D);

    CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_DEFAULT);
    CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(bufferSize);

    ThrowIfFailed(m_device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(&m_pointBuffer)));

    m_pointBufferView.BufferLocation = m_pointBuffer->GetGPUVirtualAddress();
    m_pointBufferView.StrideInBytes = sizeof(Point4D);
    m_pointBufferView.SizeInBytes = static_cast<UINT>(bufferSize);

    return true;
}

bool D3D12Renderer::CreateConstantBuffer()
{
    const UINT64 constantBufferSize = sizeof(SceneConstants) * FRAME_COUNT;

    CD3DX12_HEAP_PROPERTIES heapProps(D3D12_HEAP_TYPE_UPLOAD);
    CD3DX12_RESOURCE_DESC bufferDesc = CD3DX12_RESOURCE_DESC::Buffer(constantBufferSize);

    ThrowIfFailed(m_device->CreateCommittedResource(
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(&m_constantBuffer)));

    // Map the constant buffer
    CD3DX12_RANGE readRange(0, 0);
    ThrowIfFailed(m_constantBuffer->Map(0, &readRange, reinterpret_cast<void**>(&m_cbDataBegin)));
    m_cbData = reinterpret_cast<SceneConstants*>(m_cbDataBegin);

    // Create CBV for each frame
    CD3DX12_CPU_DESCRIPTOR_HANDLE cbvHandle(m_cbvHeap->GetCPUDescriptorHandleForHeapStart());
    for (UINT i = 0; i < FRAME_COUNT; i++)
    {
        D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
        cbvDesc.BufferLocation = m_constantBuffer->GetGPUVirtualAddress() + i * sizeof(SceneConstants);
        cbvDesc.SizeInBytes = sizeof(SceneConstants);
        m_device->CreateConstantBufferView(&cbvDesc, cbvHandle);
        cbvHandle.Offset(1, m_cbvDescriptorSize);
    }

    return true;
}

void D3D12Renderer::CreateViewportScissor()
{
    m_viewport.TopLeftX = 0.0f;
    m_viewport.TopLeftY = 0.0f;
    m_viewport.Width = static_cast<float>(m_width);
    m_viewport.Height = static_cast<float>(m_height);
    m_viewport.MinDepth = 0.0f;
    m_viewport.MaxDepth = 1.0f;

    m_scissorRect.left = 0;
    m_scissorRect.top = 0;
    m_scissorRect.right = static_cast<LONG>(m_width);
    m_scissorRect.bottom = static_cast<LONG>(m_height);
}

void D3D12Renderer::Update(float deltaTime)
{
    m_time += deltaTime;

    // Update 4D rotation
    m_hypersphereGen.UpdateRotation(deltaTime, m_rotationSpeedWX, m_rotationSpeedWY, m_rotationSpeedWZ);
}

void D3D12Renderer::UpdateConstantBuffer()
{
    // Create view-projection matrix with orbit camera
    // Convert spherical coordinates to Cartesian
    float camX = m_cameraDistance * sinf(m_cameraAngleY) * cosf(m_cameraAngleX);
    float camY = m_cameraDistance * cosf(m_cameraAngleY);
    float camZ = m_cameraDistance * sinf(m_cameraAngleY) * sinf(m_cameraAngleX);

    XMVECTOR eye = XMVectorSet(camX, camY, camZ, 0.0f);
    XMVECTOR at = XMVectorSet(0.0f, 0.0f, 0.0f, 0.0f);
    XMVECTOR up = XMVectorSet(0.0f, 1.0f, 0.0f, 0.0f);

    XMMATRIX view = XMMatrixLookAtLH(eye, at, up);
    XMMATRIX projection = XMMatrixPerspectiveFovLH(XM_PIDIV4, m_aspectRatio, 0.1f, 100.0f);
    XMMATRIX viewProj = XMMatrixMultiply(view, projection);

    // Get 4D rotation matrix
    const Math4D::Matrix4D& rot4D = m_hypersphereGen.GetRotationMatrix();

    // Fill constant buffer
    XMStoreFloat4x4(&m_cbData[m_frameIndex].viewProj, XMMatrixTranspose(viewProj));

    // Store 4D rotation matrix (transpose for HLSL column-major layout)
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            m_cbData[m_frameIndex].rotation4D[j * 4 + i] = rot4D.m[i][j];

    m_cbData[m_frameIndex].projectionDistance = m_projectionDistance;
    m_cbData[m_frameIndex].sphereRadius = 1.0f;
    m_cbData[m_frameIndex].projectionType = static_cast<int>(m_projectionType);
    m_cbData[m_frameIndex].pointCount = m_pointCount;
    m_cbData[m_frameIndex].time = m_time;
}

void D3D12Renderer::PopulateCommandList()
{
    ThrowIfFailed(m_commandAllocators[m_frameIndex]->Reset());
    ThrowIfFailed(m_commandList->Reset(m_commandAllocators[m_frameIndex].Get(), m_pipelineState.Get()));

    // Set necessary state
    m_commandList->SetGraphicsRootSignature(m_rootSignature.Get());

    ID3D12DescriptorHeap* ppHeaps[] = { m_cbvHeap.Get() };
    m_commandList->SetDescriptorHeaps(_countof(ppHeaps), ppHeaps);

    CD3DX12_GPU_DESCRIPTOR_HANDLE cbvHandle(m_cbvHeap->GetGPUDescriptorHandleForHeapStart(), m_frameIndex, m_cbvDescriptorSize);
    m_commandList->SetGraphicsRootDescriptorTable(0, cbvHandle);

    m_commandList->RSSetViewports(1, &m_viewport);
    m_commandList->RSSetScissorRects(1, &m_scissorRect);

    // Transition back buffer to render target
    CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        m_renderTargets[m_frameIndex].Get(),
        D3D12_RESOURCE_STATE_PRESENT,
        D3D12_RESOURCE_STATE_RENDER_TARGET);
    m_commandList->ResourceBarrier(1, &barrier);

    // Get render target and depth stencil handles
    CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(m_rtvHeap->GetCPUDescriptorHandleForHeapStart(), m_frameIndex, m_rtvDescriptorSize);
    CD3DX12_CPU_DESCRIPTOR_HANDLE dsvHandle(m_dsvHeap->GetCPUDescriptorHandleForHeapStart());
    m_commandList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);

    // Clear
    const float clearColor[] = { 0.0f, 0.0f, 0.0f, 1.0f };
    m_commandList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
    m_commandList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

    // Draw points
    m_commandList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
    m_commandList->IASetVertexBuffers(0, 1, &m_pointBufferView);
    m_commandList->DrawInstanced(m_pointCount, 1, 0, 0);

    // Transition back buffer to present
    barrier = CD3DX12_RESOURCE_BARRIER::Transition(
        m_renderTargets[m_frameIndex].Get(),
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        D3D12_RESOURCE_STATE_PRESENT);
    m_commandList->ResourceBarrier(1, &barrier);

    ThrowIfFailed(m_commandList->Close());
}

void D3D12Renderer::Render()
{
    UpdateConstantBuffer();
    PopulateCommandList();

    // Execute command list
    ID3D12CommandList* ppCommandLists[] = { m_commandList.Get() };
    m_commandQueue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);

    // Present
    ThrowIfFailed(m_swapChain->Present(1, 0));

    MoveToNextFrame();
}

void D3D12Renderer::WaitForGPU()
{
    ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), m_fenceValues[m_frameIndex]));
    ThrowIfFailed(m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex], m_fenceEvent));
    WaitForSingleObject(m_fenceEvent, INFINITE);
    m_fenceValues[m_frameIndex]++;
}

void D3D12Renderer::MoveToNextFrame()
{
    const UINT64 currentFenceValue = m_fenceValues[m_frameIndex];
    ThrowIfFailed(m_commandQueue->Signal(m_fence.Get(), currentFenceValue));

    m_frameIndex = m_swapChain->GetCurrentBackBufferIndex();

    if (m_fence->GetCompletedValue() < m_fenceValues[m_frameIndex])
    {
        ThrowIfFailed(m_fence->SetEventOnCompletion(m_fenceValues[m_frameIndex], m_fenceEvent));
        WaitForSingleObject(m_fenceEvent, INFINITE);
    }

    m_fenceValues[m_frameIndex] = currentFenceValue + 1;
}

void D3D12Renderer::OnResize(uint32_t width, uint32_t height)
{
    // TODO: Implement resize logic
}

void D3D12Renderer::SetPointCount(uint32_t count)
{
    m_pointCount = count;
    // TODO: Regenerate point buffer
}

void D3D12Renderer::SetProjectionType(Math4D::ProjectionType type)
{
    m_projectionType = type;
}

void D3D12Renderer::SetRotationSpeeds(float speedWX, float speedWY, float speedWZ)
{
    m_rotationSpeedWX = speedWX;
    m_rotationSpeedWY = speedWY;
    m_rotationSpeedWZ = speedWZ;
}

void D3D12Renderer::SetProjectionDistance(float distance)
{
    m_projectionDistance = distance;
}

void D3D12Renderer::RotateCamera(float deltaX, float deltaY)
{
    // Update camera angles based on mouse movement
    m_cameraAngleX += deltaX * 0.005f;  // Horizontal rotation (yaw)
    m_cameraAngleY += deltaY * 0.005f;  // Vertical rotation (pitch)

    // Clamp vertical angle to prevent flipping
    const float PI = 3.14159265359f;
    if (m_cameraAngleY < 0.1f) m_cameraAngleY = 0.1f;
    if (m_cameraAngleY > PI - 0.1f) m_cameraAngleY = PI - 0.1f;
}

void D3D12Renderer::ZoomCamera(float delta)
{
    m_cameraDistance -= delta * 0.1f;
    if (m_cameraDistance < 1.0f) m_cameraDistance = 1.0f;
    if (m_cameraDistance > 50.0f) m_cameraDistance = 50.0f;
}
