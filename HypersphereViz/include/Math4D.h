#pragma once

#include <cmath>
#include <array>
#include <DirectXMath.h>

namespace Math4D
{
    // 4D vector structure
    struct Vector4D
    {
        float x, y, z, w;

        Vector4D() : x(0), y(0), z(0), w(0) {}
        Vector4D(float x, float y, float z, float w) : x(x), y(y), z(z), w(w) {}

        float Length() const
        {
            return sqrtf(x * x + y * y + z * z + w * w);
        }

        Vector4D Normalized() const
        {
            float len = Length();
            return len > 0 ? Vector4D(x / len, y / len, z / len, w / len) : Vector4D();
        }
    };

    // 4x4 rotation matrix for 4D space
    struct Matrix4D
    {
        float m[4][4];

        Matrix4D()
        {
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    m[i][j] = (i == j) ? 1.0f : 0.0f;
        }

        static Matrix4D Identity()
        {
            return Matrix4D();
        }

        // Rotation in XW plane (w-x rotation)
        static Matrix4D RotationXW(float angle)
        {
            Matrix4D mat;
            float c = cosf(angle);
            float s = sinf(angle);
            mat.m[0][0] = c;  mat.m[0][3] = -s;
            mat.m[3][0] = s;  mat.m[3][3] = c;
            return mat;
        }

        // Rotation in YW plane (w-y rotation)
        static Matrix4D RotationYW(float angle)
        {
            Matrix4D mat;
            float c = cosf(angle);
            float s = sinf(angle);
            mat.m[1][1] = c;  mat.m[1][3] = -s;
            mat.m[3][1] = s;  mat.m[3][3] = c;
            return mat;
        }

        // Rotation in ZW plane (w-z rotation)
        static Matrix4D RotationZW(float angle)
        {
            Matrix4D mat;
            float c = cosf(angle);
            float s = sinf(angle);
            mat.m[2][2] = c;  mat.m[2][3] = -s;
            mat.m[3][2] = s;  mat.m[3][3] = c;
            return mat;
        }

        // Rotation in XY plane (standard 3D z-axis rotation)
        static Matrix4D RotationXY(float angle)
        {
            Matrix4D mat;
            float c = cosf(angle);
            float s = sinf(angle);
            mat.m[0][0] = c;  mat.m[0][1] = -s;
            mat.m[1][0] = s;  mat.m[1][1] = c;
            return mat;
        }

        // Rotation in XZ plane (standard 3D y-axis rotation)
        static Matrix4D RotationXZ(float angle)
        {
            Matrix4D mat;
            float c = cosf(angle);
            float s = sinf(angle);
            mat.m[0][0] = c;  mat.m[0][2] = -s;
            mat.m[2][0] = s;  mat.m[2][2] = c;
            return mat;
        }

        // Rotation in YZ plane (standard 3D x-axis rotation)
        static Matrix4D RotationYZ(float angle)
        {
            Matrix4D mat;
            float c = cosf(angle);
            float s = sinf(angle);
            mat.m[1][1] = c;  mat.m[1][2] = -s;
            mat.m[2][1] = s;  mat.m[2][2] = c;
            return mat;
        }

        // Matrix multiplication
        Matrix4D operator*(const Matrix4D& other) const
        {
            Matrix4D result;
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    result.m[i][j] = 0;
                    for (int k = 0; k < 4; k++)
                        result.m[i][j] += m[i][k] * other.m[k][j];
                }
            }
            return result;
        }

        // Transform a 4D vector
        Vector4D Transform(const Vector4D& v) const
        {
            return Vector4D(
                m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w,
                m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w,
                m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w,
                m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w
            );
        }
    };

    // Projection types
    enum class ProjectionType : int
    {
        Perspective = 0,
        Stereographic = 1,
        Orthographic = 2
    };

    // Project 4D point to 3D
    inline DirectX::XMFLOAT3 ProjectTo3D(const Vector4D& point4D, ProjectionType type, float distance = 2.0f, float radius = 1.0f)
    {
        DirectX::XMFLOAT3 result;

        switch (type)
        {
        case ProjectionType::Perspective:
        {
            // Perspective projection: scale by distance/(distance - w)
            float scale = distance / (distance - point4D.w);
            result.x = point4D.x * scale;
            result.y = point4D.y * scale;
            result.z = point4D.z * scale;
            break;
        }
        case ProjectionType::Stereographic:
        {
            // Stereographic projection from north pole
            float scale = radius / (radius - point4D.w);
            result.x = point4D.x * scale;
            result.y = point4D.y * scale;
            result.z = point4D.z * scale;
            break;
        }
        case ProjectionType::Orthographic:
        default:
        {
            // Orthographic: simply drop w coordinate
            result.x = point4D.x;
            result.y = point4D.y;
            result.z = point4D.z;
            break;
        }
        }

        return result;
    }

    // Generate random point on 4D hypersphere using Marsaglia method
    inline Vector4D RandomPointOnHypersphere(float radius = 1.0f)
    {
        // Generate 4 Gaussian random numbers
        auto gaussian = []() -> float {
            float u1 = (float)rand() / RAND_MAX;
            float u2 = (float)rand() / RAND_MAX;
            return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
        };

        Vector4D point(gaussian(), gaussian(), gaussian(), gaussian());
        point = point.Normalized();
        return Vector4D(point.x * radius, point.y * radius, point.z * radius, point.w * radius);
    }
}
