// Vertex shader for 4D to 3D projection and rendering

struct Point4D
{
    float4 position : POSITION;  // x, y, z, w
    uint flags : TEXCOORD0;      // 0 = normal, 1 = north pole, 2 = south pole
    float3 padding : TEXCOORD1;
};

struct VSOutput
{
    float4 position : SV_POSITION;
    float4 color : COLOR0;
    float pointSize : PSIZE;
    float2 texcoord : TEXCOORD0;  // For triangle mode
};

cbuffer SceneConstants : register(b0)
{
    float4x4 viewProj;
    float4x4 rotation4D;
    float projectionDistance;
    float sphereRadius;
    int projectionType;  // 0=perspective, 1=stereographic, 2=orthographic
    int pointCount;
    float time;
    float3 padding;
};

// Project 4D point to 3D space
float3 ProjectTo3D(float4 point4D)
{
    float3 result;

    if (projectionType == 0)  // Perspective
    {
        // Perspective projection from viewpoint at (0,0,0,projectionDistance)
        // Projects onto w=0 hyperplane
        // This is the natural mathematical projection - no normalization
        float scale = projectionDistance / (projectionDistance - point4D.w);
        result = point4D.xyz * scale;
    }
    else if (projectionType == 1)  // Stereographic
    {
        // Stereographic projection from north pole (w = sphereRadius)
        // Projects onto w=0 hyperplane
        // This projection is conformal (preserves angles) and unbounded
        float scale = sphereRadius / (sphereRadius - point4D.w);
        result = point4D.xyz * scale;
    }
    else  // Orthographic (projectionType == 2)
    {
        // Orthographic projection - simply drop the w coordinate
        // Maps 4D sphere surface (S³) to 3D solid ball (B³) naturally
        result = point4D.xyz;
    }

    return result;
}

// Color based on w-coordinate and flags
float4 GetPointColor(float w, uint flags)
{
    // Special colors for poles
    if (flags == 1)  // North pole
        return float4(1.0, 0.0, 0.0, 1.0);  // Red
    if (flags == 2)  // South pole
        return float4(0.0, 0.0, 1.0, 1.0);  // Blue

    // Color gradient based on w-coordinate
    // Map w from [-sphereRadius, +sphereRadius] to [0, 1]
    float t = (w + sphereRadius) / (2.0 * sphereRadius);
    t = saturate(t);

    // Create a nice color gradient: blue -> cyan -> green -> yellow -> red
    float4 color = float4(1.0, 0.0, 0.0, 1.0);  // Default to red
    if (t < 0.25)
    {
        float local_t = t / 0.25;
        color = lerp(float4(0.0, 0.0, 1.0, 1.0), float4(0.0, 1.0, 1.0, 1.0), local_t);
    }
    else if (t < 0.5)
    {
        float local_t = (t - 0.25) / 0.25;
        color = lerp(float4(0.0, 1.0, 1.0, 1.0), float4(0.0, 1.0, 0.0, 1.0), local_t);
    }
    else if (t < 0.75)
    {
        float local_t = (t - 0.5) / 0.25;
        color = lerp(float4(0.0, 1.0, 0.0, 1.0), float4(1.0, 1.0, 0.0, 1.0), local_t);
    }
    else
    {
        float local_t = (t - 0.75) / 0.25;
        color = lerp(float4(1.0, 1.0, 0.0, 1.0), float4(1.0, 0.0, 0.0, 1.0), local_t);
    }

    return color;
}

VSOutput main(Point4D input)
{
    VSOutput output;

    // Apply 4D rotation
    float4 rotated4D = mul(rotation4D, input.position);

    // Project to 3D
    float3 pos3D = ProjectTo3D(rotated4D);

    // Transform to screen space
    output.position = mul(viewProj, float4(pos3D, 1.0));

    // Set color based on w-coordinate and flags
    output.color = GetPointColor(rotated4D.w, input.flags);

    // Larger points for poles
    output.pointSize = (input.flags > 0) ? 8.0 : 2.0;

    // Texcoord for triangle mode (unused in point mode)
    output.texcoord = float2(0.0, 0.0);

    return output;
}
