// Pixel shader for point and triangle rendering

struct PSInput
{
    float4 position : SV_POSITION;
    float4 color : COLOR0;
    float2 texcoord : TEXCOORD0;
};

float4 main(PSInput input) : SV_TARGET
{
    // For triangle mode with geometry shader, add soft edges using texcoord

    // Calculate distance from triangle centroid for soft falloff
    float2 center = float2(0.5, 0.666667);  // Centroid of triangle
    float dist = length(input.texcoord - center);

    // Soft falloff - triangle vertices are at distance ~0.67 from center
    // Fade from full opacity at center to 50% at edges
    float alpha = 1.0 - smoothstep(0.0, 0.8, dist);

    // Return color with alpha (70% base opacity with soft edges)
    return float4(input.color.rgb, input.color.a * alpha * 0.7);
}
