// Geometry shader to expand points into billboarded triangles

struct GSInput
{
    float4 position : SV_POSITION;
    float4 color : COLOR0;
    float pointSize : PSIZE;
};

struct GSOutput
{
    float4 position : SV_POSITION;
    float4 color : COLOR0;
    float2 texcoord : TEXCOORD0;
};

cbuffer SceneConstants : register(b0)
{
    float4x4 viewProj;
    float4x4 rotation4D;
    float projectionDistance;
    float sphereRadius;
    int projectionType;
    int pointCount;
    float time;
    float padding[27];
};

[maxvertexcount(3)]
void main(point GSInput input[1], inout TriangleStream<GSOutput> outputStream)
{
    GSOutput output;

    // Get the input point's screen position
    float4 center = input[0].position;

    // Size of the triangle in screen space
    float size = input[0].pointSize * 0.002;  // Scale to NDC space

    // Create a billboarded triangle (facing camera)
    // We'll make an equilateral triangle
    const float sqrt3 = 1.732051;

    // Vertex 0 (top)
    output.position = center + float4(0.0, size * sqrt3 / 3.0 * 2.0, 0.0, 0.0);
    output.color = input[0].color;
    output.texcoord = float2(0.5, 0.0);
    outputStream.Append(output);

    // Vertex 1 (bottom left)
    output.position = center + float4(-size, -size * sqrt3 / 3.0, 0.0, 0.0);
    output.color = input[0].color;
    output.texcoord = float2(0.0, 1.0);
    outputStream.Append(output);

    // Vertex 2 (bottom right)
    output.position = center + float4(size, -size * sqrt3 / 3.0, 0.0, 0.0);
    output.color = input[0].color;
    output.texcoord = float2(1.0, 1.0);
    outputStream.Append(output);

    outputStream.RestartStrip();
}
