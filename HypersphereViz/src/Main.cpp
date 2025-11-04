#include <windows.h>
#include <windowsx.h>  // For GET_X_LPARAM, GET_Y_LPARAM
#include <iostream>
#include "../include/D3D12Renderer.h"
#include <chrono>

// Global variables
HWND g_hwnd = nullptr;
D3D12Renderer* g_renderer = nullptr;
bool g_running = true;

// Rotation speed state (shared across all key handlers)
float g_speedWX = 0.5f;
float g_speedWY = 0.3f;
float g_speedWZ = 0.7f;

// Current projection type
const char* g_projectionNames[] = { "Perspective", "Stereographic", "Orthographic" };
int g_currentProjection = 0;  // 0 = Perspective

// Mouse state for camera control
bool g_mouseLeftDown = false;
bool g_mouseRightDown = false;
int g_lastMouseX = 0;
int g_lastMouseY = 0;

// Function to update window title with current state
void UpdateWindowTitle()
{
    if (!g_renderer) return;

    wchar_t title[256];
    swprintf_s(title, L"4D Hypersphere Viz | %S | WX:%.1f WY:%.1f WZ:%.1f | Dist:%.1f",
        g_projectionNames[g_currentProjection],
        g_speedWX, g_speedWY, g_speedWZ,
        g_renderer->GetCameraDistance());

    SetWindowText(g_hwnd, title);
}

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

    case WM_LBUTTONDOWN:
        g_mouseLeftDown = true;
        g_lastMouseX = GET_X_LPARAM(lParam);
        g_lastMouseY = GET_Y_LPARAM(lParam);
        SetCapture(hwnd);
        return 0;

    case WM_LBUTTONUP:
        g_mouseLeftDown = false;
        ReleaseCapture();
        return 0;

    case WM_RBUTTONDOWN:
        g_mouseRightDown = true;
        g_lastMouseX = GET_X_LPARAM(lParam);
        g_lastMouseY = GET_Y_LPARAM(lParam);
        SetCapture(hwnd);
        return 0;

    case WM_RBUTTONUP:
        g_mouseRightDown = false;
        ReleaseCapture();
        return 0;

    case WM_MOUSEMOVE:
        if (g_renderer && (g_mouseLeftDown || g_mouseRightDown))
        {
            int mouseX = GET_X_LPARAM(lParam);
            int mouseY = GET_Y_LPARAM(lParam);
            int deltaX = mouseX - g_lastMouseX;
            int deltaY = mouseY - g_lastMouseY;

            if (g_mouseLeftDown)
            {
                // Left mouse: orbit camera
                g_renderer->RotateCamera((float)deltaX, (float)deltaY);
                UpdateWindowTitle();
            }

            g_lastMouseX = mouseX;
            g_lastMouseY = mouseY;
        }
        return 0;

    case WM_MOUSEWHEEL:
        if (g_renderer)
        {
            int delta = GET_WHEEL_DELTA_WPARAM(wParam);
            g_renderer->ZoomCamera((float)delta / 120.0f);
            UpdateWindowTitle();
        }
        return 0;

    case WM_KEYDOWN:
        switch (wParam)
        {
        case VK_ESCAPE:
            g_running = false;
            PostQuitMessage(0);
            return 0;
        case '1':  // Perspective projection
            if (g_renderer)
            {
                g_currentProjection = 0;
                g_renderer->SetProjectionType(Math4D::ProjectionType::Perspective);
                std::cout << "\n[PROJECTION] " << g_projectionNames[g_currentProjection] << "\n";
                UpdateWindowTitle();
            }
            return 0;
        case '2':  // Stereographic projection
            if (g_renderer)
            {
                g_currentProjection = 1;
                g_renderer->SetProjectionType(Math4D::ProjectionType::Stereographic);
                std::cout << "\n[PROJECTION] " << g_projectionNames[g_currentProjection] << "\n";
                UpdateWindowTitle();
            }
            return 0;
        case '3':  // Orthographic projection
            if (g_renderer)
            {
                g_currentProjection = 2;
                g_renderer->SetProjectionType(Math4D::ProjectionType::Orthographic);
                std::cout << "\n[PROJECTION] " << g_projectionNames[g_currentProjection] << "\n";
                UpdateWindowTitle();
            }
            return 0;
        case 'Q':  // Increase WX rotation
            if (g_renderer)
            {
                g_speedWX += 0.1f;
                g_renderer->SetRotationSpeeds(g_speedWX, g_speedWY, g_speedWZ);
                std::cout << "\n[ROTATION] WX=" << g_speedWX << " WY=" << g_speedWY << " WZ=" << g_speedWZ << "\n";
                UpdateWindowTitle();
            }
            return 0;
        case 'A':  // Decrease WX rotation
            if (g_renderer)
            {
                g_speedWX -= 0.1f;
                g_renderer->SetRotationSpeeds(g_speedWX, g_speedWY, g_speedWZ);
                std::cout << "\n[ROTATION] WX=" << g_speedWX << " WY=" << g_speedWY << " WZ=" << g_speedWZ << "\n";
                UpdateWindowTitle();
            }
            return 0;
        case 'W':  // Increase WY rotation
            if (g_renderer)
            {
                g_speedWY += 0.1f;
                g_renderer->SetRotationSpeeds(g_speedWX, g_speedWY, g_speedWZ);
                std::cout << "\n[ROTATION] WX=" << g_speedWX << " WY=" << g_speedWY << " WZ=" << g_speedWZ << "\n";
                UpdateWindowTitle();
            }
            return 0;
        case 'S':  // Decrease WY rotation
            if (g_renderer)
            {
                g_speedWY -= 0.1f;
                g_renderer->SetRotationSpeeds(g_speedWX, g_speedWY, g_speedWZ);
                std::cout << "\n[ROTATION] WX=" << g_speedWX << " WY=" << g_speedWY << " WZ=" << g_speedWZ << "\n";
                UpdateWindowTitle();
            }
            return 0;
        case 'E':  // Increase WZ rotation
            if (g_renderer)
            {
                g_speedWZ += 0.1f;
                g_renderer->SetRotationSpeeds(g_speedWX, g_speedWY, g_speedWZ);
                std::cout << "\n[ROTATION] WX=" << g_speedWX << " WY=" << g_speedWY << " WZ=" << g_speedWZ << "\n";
                UpdateWindowTitle();
            }
            return 0;
        case 'D':  // Decrease WZ rotation
            if (g_renderer)
            {
                g_speedWZ -= 0.1f;
                g_renderer->SetRotationSpeeds(g_speedWX, g_speedWY, g_speedWZ);
                std::cout << "\n[ROTATION] WX=" << g_speedWX << " WY=" << g_speedWY << " WZ=" << g_speedWZ << "\n";
                UpdateWindowTitle();
            }
            return 0;
        }
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
    std::cout << "  1/2/3       - Perspective/Stereographic/Orthographic Projection\n";
    std::cout << "  Q/A         - Increase/Decrease WX rotation speed\n";
    std::cout << "  W/S         - Increase/Decrease WY rotation speed\n";
    std::cout << "  E/D         - Increase/Decrease WZ rotation speed\n";
    std::cout << "  LEFT MOUSE  - Drag to orbit camera\n";
    std::cout << "  MOUSE WHEEL - Zoom in/out\n";
    std::cout << "  ESC         - Exit\n\n";
    std::cout << "State displayed in window title bar\n\n";
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
    std::cout << "Rendering 100,000 points on 4D hypersphere SHELL...\n";
    std::cout << "  Shell radius: 1.0\n";
    std::cout << "  Shell thickness: 0.02 (2% of radius)\n";
    std::cout << "\n=== INITIAL STATE ===\n";
    std::cout << "[PROJECTION] " << g_projectionNames[g_currentProjection] << "\n";
    std::cout << "[ROTATION] WX=" << g_speedWX << " WY=" << g_speedWY << " WZ=" << g_speedWZ << "\n";
    std::cout << "=====================\n\n";

    // Set initial window title
    UpdateWindowTitle();

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
