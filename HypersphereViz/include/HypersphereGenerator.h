#pragma once

#include <vector>
#include <d3d12.h>
#include <wrl/client.h>
#include "Math4D.h"

using Microsoft::WRL::ComPtr;

// Point structure for compute shader
struct Point4D
{
    float x, y, z, w;
    uint32_t flags;  // 0 = normal, 1 = north pole, 2 = south pole
    float padding[3];  // Align to 32 bytes
};

class HypersphereGenerator
{
public:
    HypersphereGenerator();
    ~HypersphereGenerator();

    // Initialize with DirectX 12 device
    bool Initialize(ID3D12Device* device, ID3D12GraphicsCommandList* commandList, uint32_t pointCount);

    // Generate points on CPU (for initialization)
    void GeneratePointsCPU(std::vector<Point4D>& points, uint32_t count, float radius = 1.0f, float shellThickness = 0.01f);

    // Update rotation angles
    void UpdateRotation(float deltaTime, float speedWX, float speedWY, float speedWZ);

    // Get current rotation matrix
    const Math4D::Matrix4D& GetRotationMatrix() const { return m_rotationMatrix; }

    // Get rotation angles
    float GetAngleWX() const { return m_angleWX; }
    float GetAngleWY() const { return m_angleWY; }
    float GetAngleWZ() const { return m_angleWZ; }

private:
    Math4D::Matrix4D m_rotationMatrix;
    float m_angleWX;
    float m_angleWY;
    float m_angleWZ;
    float m_radius;
};
