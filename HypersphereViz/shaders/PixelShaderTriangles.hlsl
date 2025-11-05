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
    // For point mode, texcoord will be (0,0) and this just returns the color

    // Calculate distance from triangle center for soft falloff
    float2 center = float2(0.5, 0.66);  // Approximate center of triangle
    float dist = length(input.texcoord - center);

    // Soft falloff for nicer looking triangles
    float alpha = 1.0 - smoothstep(0.0, 0.6, dist);

    // Return color with alpha (for triangles: 60% opacity with soft edges)
    return float4(input.color.rgb, input.color.a * alpha * 0.6);
}
