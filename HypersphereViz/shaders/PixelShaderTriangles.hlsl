// Pixel shader for point and triangle rendering

struct PSInput
{
    float4 position : SV_POSITION;
    float4 color : COLOR0;
    float2 texcoord : TEXCOORD0;
};

float4 main(PSInput input) : SV_TARGET
{
    // Debug: Just return solid color to verify triangles are rendering
    return input.color;
}
