// Pixel shader for point rendering

struct PSInput
{
    float4 position : SV_POSITION;
    float4 color : COLOR0;
};

float4 main(PSInput input) : SV_TARGET
{
    // For point mode, just return the color as-is
    return input.color;
}
