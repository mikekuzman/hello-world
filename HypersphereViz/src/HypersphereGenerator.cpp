#include "../include/HypersphereGenerator.h"
#include <random>
#include <cmath>

HypersphereGenerator::HypersphereGenerator()
    : m_angleWX(0.0f)
    , m_angleWY(0.0f)
    , m_angleWZ(0.0f)
    , m_radius(1.0f)
{
    m_rotationMatrix = Math4D::Matrix4D::Identity();
}

HypersphereGenerator::~HypersphereGenerator()
{
}

bool HypersphereGenerator::Initialize(ID3D12Device* device, ID3D12GraphicsCommandList* commandList, uint32_t pointCount)
{
    // For now, we'll generate points on CPU
    // Future optimization: use compute shader for GPU generation
    return true;
}

void HypersphereGenerator::GeneratePointsCPU(std::vector<Point4D>& points, uint32_t count, float radius, float shellThickness)
{
    points.resize(count);
    m_radius = radius;

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> normalDist(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniformDist(0.0f, 1.0f);

    // Generate random points on hypersphere SHELL
    for (uint32_t i = 0; i < count; ++i)
    {
        Math4D::Vector4D point;

        if (i == 0)
        {
            // North pole: (0, 0, 0, radius) - at outer surface
            point = Math4D::Vector4D(0.0f, 0.0f, 0.0f, radius);
            points[i].flags = 1;
        }
        else if (i == 1)
        {
            // South pole: (0, 0, 0, -radius) - at outer surface
            point = Math4D::Vector4D(0.0f, 0.0f, 0.0f, -radius);
            points[i].flags = 2;
        }
        else
        {
            // Random point using Marsaglia method
            point.x = normalDist(gen);
            point.y = normalDist(gen);
            point.z = normalDist(gen);
            point.w = normalDist(gen);

            // Normalize to unit sphere
            float length = point.Length();
            if (length > 0.0001f)
            {
                point.x /= length;
                point.y /= length;
                point.z /= length;
                point.w /= length;
            }

            // Add random radial variation for shell thickness
            // Random radius between (radius - thickness/2) and (radius + thickness/2)
            float shellRadius = radius + (uniformDist(gen) - 0.5f) * shellThickness;
            point.x *= shellRadius;
            point.y *= shellRadius;
            point.z *= shellRadius;
            point.w *= shellRadius;

            points[i].flags = 0;
        }

        points[i].x = point.x;
        points[i].y = point.y;
        points[i].z = point.z;
        points[i].w = point.w;
        points[i].padding[0] = 0.0f;
        points[i].padding[1] = 0.0f;
        points[i].padding[2] = 0.0f;
    }
}

void HypersphereGenerator::UpdateRotation(float deltaTime, float speedWX, float speedWY, float speedWZ)
{
    // Update rotation angles
    m_angleWX += speedWX * deltaTime;
    m_angleWY += speedWY * deltaTime;
    m_angleWZ += speedWZ * deltaTime;

    // Keep angles in [0, 2π] range
    const float TWO_PI = 6.28318530718f;
    m_angleWX = fmodf(m_angleWX, TWO_PI);
    m_angleWY = fmodf(m_angleWY, TWO_PI);
    m_angleWZ = fmodf(m_angleWZ, TWO_PI);

    // Compute combined rotation matrix
    Math4D::Matrix4D rotWX = Math4D::Matrix4D::RotationXW(m_angleWX);
    Math4D::Matrix4D rotWY = Math4D::Matrix4D::RotationYW(m_angleWY);
    Math4D::Matrix4D rotWZ = Math4D::Matrix4D::RotationZW(m_angleWZ);

    m_rotationMatrix = rotWX * rotWY * rotWZ;
}
