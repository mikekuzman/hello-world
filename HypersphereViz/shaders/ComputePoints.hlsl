// Compute shader for generating and updating 4D hypersphere points
// This can be used for real-time point generation on GPU for millions of points

struct Point4D
{
    float4 position;  // x, y, z, w
    uint flags;       // 0 = normal, 1 = north pole, 2 = south pole
    float3 padding;
};

cbuffer ComputeConstants : register(b0)
{
    float4x4 rotation4D;
    float radius;
    float time;
    uint seed;
    uint padding;
};

RWStructuredBuffer<Point4D> OutputPoints : register(u0);

// Random number generator (using thread ID and time as seed)
float Random(uint seed, uint index)
{
    uint state = seed + index * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return float((word >> 22u) ^ word) / 4294967295.0;
}

// Box-Muller transform for Gaussian random numbers
float GaussianRandom(uint seed, uint index, uint offset)
{
    float u1 = Random(seed, index * 2 + offset);
    float u2 = Random(seed, index * 2 + offset + 1);
    return sqrt(-2.0 * log(u1 + 0.0001)) * cos(6.28318530718 * u2);
}

[numthreads(256, 1, 1)]
void main(uint3 threadID : SV_DispatchThreadID)
{
    uint index = threadID.x;

    // Generate point on 4D hypersphere using Marsaglia method
    float x = GaussianRandom(seed, index, 0);
    float y = GaussianRandom(seed, index, 100);
    float z = GaussianRandom(seed, index, 200);
    float w = GaussianRandom(seed, index, 300);

    // Normalize to unit sphere
    float len = sqrt(x * x + y * y + z * z + w * w);
    if (len > 0.0001)
    {
        x /= len;
        y /= len;
        z /= len;
        w /= len;
    }

    // Scale to desired radius
    float4 pos4D = float4(x, y, z, w) * radius;

    // Apply 4D rotation
    pos4D = mul(rotation4D, pos4D);

    // Check if this is a pole point
    uint flags = 0;
    if (index == 0)
        flags = 1;  // North pole
    else if (index == 1)
        flags = 2;  // South pole

    // Write to output buffer
    OutputPoints[index].position = pos4D;
    OutputPoints[index].flags = flags;
}
