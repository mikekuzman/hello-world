#include <windows.h>
#include <iostream>
#include "../include/D3D12Renderer.h"
#include <chrono>

// Global variables
HWND g_hwnd = nullptr;
D3D12Renderer* g_renderer = nullptr;
bool g_running = true;

// Rotation speed tracking
float g_rotationSpeedWX = 0.5f;
float g_rotationSpeedWY = 0.3f;
float g_rotationSpeedWZ = 0.7f;

// Camera speed settings
const float CAMERA_MOVE_SPEED = 0.1f;
const float CAMERA_MOUSE_SENSITIVITY = 0.002f;

// Keyboard state
bool g_keyState[256] = { false };

// Window procedure
LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {
    case WM_DESTROY:
        g_running = false;
        PostQuitMessage(0);
        return 0;

    case WM_SIZE:
        if (g_renderer && wParam != SIZE_MINIMIZED)
        {
            UINT width = LOWORD(lParam);
            UINT height = HIWORD(lParam);
            g_renderer->OnResize(width, height);
        }
        return 0;

    case WM_INPUT:
    {
        // Raw mouse input for camera look
        UINT dwSize = sizeof(RAWINPUT);
        static BYTE lpb[sizeof(RAWINPUT)];

        GetRawInputData((HRAWINPUT)lParam, RID_INPUT, lpb, &dwSize, sizeof(RAWINPUTHEADER));
        RAWINPUT* raw = (RAWINPUT*)lpb;

        if (raw->header.dwType == RIM_TYPEMOUSE && g_renderer)
        {
            float deltaX = (float)raw->data.mouse.lLastX;
            float deltaY = (float)raw->data.mouse.lLastY;
            g_renderer->RotateCamera(deltaX * CAMERA_MOUSE_SENSITIVITY, -deltaY * CAMERA_MOUSE_SENSITIVITY);
        }
        return 0;
    }

    case WM_KEYDOWN:
        if (wParam < 256)
            g_keyState[wParam] = true;

        switch (wParam)
        {
        case VK_ESCAPE:
            g_running = false;
            PostQuitMessage(0);
            return 0;

        // Projection selection
        case '1':
            if (g_renderer)
                g_renderer->SetProjectionType(Math4D::ProjectionType::Perspective);
            return 0;
        case '2':
            if (g_renderer)
                g_renderer->SetProjectionType(Math4D::ProjectionType::Stereographic);
            return 0;
        case '3':
            if (g_renderer)
                g_renderer->SetProjectionType(Math4D::ProjectionType::Orthographic);
            return 0;

        // Increase rotation speeds
        case '4':  // Increase WX rotation
            g_rotationSpeedWX += 0.1f;
            if (g_renderer)
                g_renderer->SetRotationSpeeds(g_rotationSpeedWX, g_rotationSpeedWY, g_rotationSpeedWZ);
            return 0;
        case '5':  // Increase WY rotation
            g_rotationSpeedWY += 0.1f;
            if (g_renderer)
                g_renderer->SetRotationSpeeds(g_rotationSpeedWX, g_rotationSpeedWY, g_rotationSpeedWZ);
            return 0;
        case '6':  // Increase WZ rotation
            g_rotationSpeedWZ += 0.1f;
            if (g_renderer)
                g_renderer->SetRotationSpeeds(g_rotationSpeedWX, g_rotationSpeedWY, g_rotationSpeedWZ);
            return 0;

        // Decrease rotation speeds
        case 'R':  // Decrease WX rotation
            g_rotationSpeedWX -= 0.1f;
            if (g_renderer)
                g_renderer->SetRotationSpeeds(g_rotationSpeedWX, g_rotationSpeedWY, g_rotationSpeedWZ);
            return 0;
        case 'T':  // Decrease WY rotation
            g_rotationSpeedWY -= 0.1f;
            if (g_renderer)
                g_renderer->SetRotationSpeeds(g_rotationSpeedWX, g_rotationSpeedWY, g_rotationSpeedWZ);
            return 0;
        case 'Y':  // Decrease WZ rotation
            g_rotationSpeedWZ -= 0.1f;
            if (g_renderer)
                g_renderer->SetRotationSpeeds(g_rotationSpeedWX, g_rotationSpeedWY, g_rotationSpeedWZ);
            return 0;
        }
        break;

    case WM_KEYUP:
        if (wParam < 256)
            g_keyState[wParam] = false;
        break;
    }

    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

// Create window
bool CreateAppWindow(HINSTANCE hInstance, int nCmdShow, int width, int height)
{
    // Register window class
    WNDCLASSEX wc = {};
    wc.cbSize = sizeof(WNDCLASSEX);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)GetStockObject(BLACK_BRUSH);
    wc.lpszClassName = L"HypersphereVizWindowClass";

    if (!RegisterClassEx(&wc))
        return false;

    // Calculate window size for desired client area
    RECT rc = { 0, 0, static_cast<LONG>(width), static_cast<LONG>(height) };
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW, FALSE);

    // Create window
    g_hwnd = CreateWindowEx(
        0,
        L"HypersphereVizWindowClass",
        L"4D Hypersphere Visualizer - DirectX 12",
        WS_OVERLAPPEDWINDOW,
        CW_USEDEFAULT, CW_USEDEFAULT,
        rc.right - rc.left,
        rc.bottom - rc.top,
        nullptr,
        nullptr,
        hInstance,
        nullptr);

    if (!g_hwnd)
        return false;

    ShowWindow(g_hwnd, nCmdShow);
    UpdateWindow(g_hwnd);

    // Register raw input device for mouse
    RAWINPUTDEVICE rid[1];
    rid[0].usUsagePage = 0x01;  // Generic desktop controls
    rid[0].usUsage = 0x02;      // Mouse
    rid[0].dwFlags = 0;
    rid[0].hwndTarget = g_hwnd;

    if (!RegisterRawInputDevices(rid, 1, sizeof(rid[0])))
    {
        return false;
    }

    // Hide cursor for FPS-style controls
    ShowCursor(FALSE);

    return true;
}

// Main entry point
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
{
    // Create console for debug output
#ifdef _DEBUG
    AllocConsole();
    FILE* pFile;
    freopen_s(&pFile, "CONOUT$", "w", stdout);
    freopen_s(&pFile, "CONOUT$", "w", stderr);
    std::cout << "4D Hypersphere Visualizer - DirectX 12\n";
    std::cout << "======================================\n\n";
    std::cout << "Controls:\n";
    std::cout << "  WASD     - Move camera (forward/left/back/right)\n";
    std::cout << "  Space/C  - Move camera up/down\n";
    std::cout << "  Shift    - Sprint (hold with movement keys)\n";
    std::cout << "  Mouse    - Look around (FPS style)\n\n";
    std::cout << "  1/2/3    - Perspective/Stereographic/Orthographic projection\n\n";
    std::cout << "  4/R      - Increase/Decrease WX rotation speed\n";
    std::cout << "  5/T      - Increase/Decrease WY rotation speed\n";
    std::cout << "  6/Y      - Increase/Decrease WZ rotation speed\n\n";
    std::cout << "  ESC      - Exit\n\n";
#endif

    // Create window
    const int width = 1920;
    const int height = 1080;
    if (!CreateAppWindow(hInstance, nCmdShow, width, height))
    {
        MessageBox(nullptr, L"Failed to create window", L"Error", MB_OK | MB_ICONERROR);
        return 1;
    }

    // Create renderer
    g_renderer = new D3D12Renderer();
    if (!g_renderer->Initialize(g_hwnd, width, height))
    {
        MessageBox(nullptr, L"Failed to initialize DirectX 12", L"Error", MB_OK | MB_ICONERROR);
        delete g_renderer;
        return 1;
    }

    std::cout << "Renderer initialized successfully!\n";
    std::cout << "Rendering 100,000 points on 4D hypersphere...\n\n";

    // Main loop
    auto lastTime = std::chrono::high_resolution_clock::now();
    MSG msg = {};

    while (g_running)
    {
        // Process messages
        while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        // Calculate delta time
        auto currentTime = std::chrono::high_resolution_clock::now();
        float deltaTime = std::chrono::duration<float>(currentTime - lastTime).count();
        lastTime = currentTime;

        // Process camera movement (WASD)
        if (g_renderer)
        {
            float moveSpeed = CAMERA_MOVE_SPEED;
            if (g_keyState[VK_SHIFT])
                moveSpeed *= 3.0f;  // Sprint when holding shift

            if (g_keyState['W'])
                g_renderer->MoveCameraForward(moveSpeed);
            if (g_keyState['S'])
                g_renderer->MoveCameraForward(-moveSpeed);
            if (g_keyState['A'])
                g_renderer->MoveCameraRight(-moveSpeed);
            if (g_keyState['D'])
                g_renderer->MoveCameraRight(moveSpeed);
            if (g_keyState[VK_SPACE])
                g_renderer->MoveCameraUp(moveSpeed);
            if (g_keyState['C'])
                g_renderer->MoveCameraUp(-moveSpeed);
        }

        // Update and render
        g_renderer->Update(deltaTime);
        g_renderer->Render();

        // Print FPS every second
        static float fpsTimer = 0.0f;
        static int frameCount = 0;
        fpsTimer += deltaTime;
        frameCount++;
        if (fpsTimer >= 1.0f)
        {
            float fps = frameCount / fpsTimer;
            std::cout << "FPS: " << fps << "\r";
            fpsTimer = 0.0f;
            frameCount = 0;
        }
    }

    // Cleanup
    delete g_renderer;

#ifdef _DEBUG
    FreeConsole();
#endif

    return static_cast<int>(msg.wParam);
}
