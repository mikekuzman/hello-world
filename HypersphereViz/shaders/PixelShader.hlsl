// Pixel shader for point rendering with smooth edges

struct PSInput
{
    float4 position : SV_POSITION;
    float4 color : COLOR0;
    float pointSize : PSIZE;
};

float4 main(PSInput input) : SV_TARGET
{
    // Output the interpolated color
    // For point sprites, we could add circular falloff here
    // but for now we'll keep it simple for maximum performance
    return input.color;
}
