#ifndef UNIVERSAL_POSTPROCESSING_COMMON_INCLUDED
#define UNIVERSAL_POSTPROCESSING_COMMON_INCLUDED

#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Color.hlsl"
#include "Packages/com.unity.render-pipelines.core/Runtime/Utilities/Blit.hlsl"

// ----------------------------------------------------------------------------------
// Utility functions

half GetLuminance(half3 colorLinear)
{
#if _TONEMAP_ACES
    return AcesLuminance(colorLinear);
#else
    return Luminance(colorLinear);
#endif
}

real3 GetSRGBToLinear(real3 c)
{
#if _USE_FAST_SRGB_LINEAR_CONVERSION
    return FastSRGBToLinear(c);
#else
    return SRGBToLinear(c);
#endif
}

real4 GetSRGBToLinear(real4 c)
{
#if _USE_FAST_SRGB_LINEAR_CONVERSION
    return FastSRGBToLinear(c);
#else
    return SRGBToLinear(c);
#endif
}

real3 GetLinearToSRGB(real3 c)
{
#if _USE_FAST_SRGB_LINEAR_CONVERSION
    return FastLinearToSRGB(c);
#else
    return LinearToSRGB(c);
#endif
}

real4 GetLinearToSRGB(real4 c)
{
#if _USE_FAST_SRGB_LINEAR_CONVERSION
    return FastLinearToSRGB(c);
#else
    return LinearToSRGB(c);
#endif
}

// ----------------------------------------------------------------------------------
// Shared functions for uber & fast path (on-tile)
// These should only process an input color, don't sample in neighbor pixels!

half3 ApplyVignette(half3 input, float2 uv, float2 center, float intensity, float roundness, float smoothness, half3 color)
{
    center = UnityStereoTransformScreenSpaceTex(center);
    float2 dist = abs(uv - center) * intensity;

#if defined(UNITY_SINGLE_PASS_STEREO)
    dist.x /= unity_StereoScaleOffset[unity_StereoEyeIndex].x;
#endif

    dist.x *= roundness;
    float vfactor = pow(saturate(1.0 - dot(dist, dist)), smoothness);
    return input * lerp(color, (1.0).xxx, vfactor);
}

// For GranTurismo Tonemapping
float W_f(float x, float e0, float e1)
{
    if (x <= e0)
        return 0;
    if (x >= e1)
        return 1;
    float a = (x - e0) / (e1 - e0);
    return a * a * (3 - 2 * a);
}
// For GranTurismo Tonemapping
float H_f(float x, float e0, float e1)
{
    if (x <= e0)
        return 0;
    if (x >= e1)
        return 1;
    return (x - e0) / (e1 - e0);
}
// For GranTurismo Tonemapping
float GranTurismoTonemap(float x, float P, float a, float m, float l, float c, float b)
{
    // float P = 1; // Maximum brightness
    // float a = 1; // Contrast
    // float m = 0.22; // Linear section start
    // float l = 0.4; // Linear section length
    // float c = 1; // Black pow
    // float b = 0; // Black min
    float l0 = (P - m) * l / a;
    float L0 = m - m / a;
    float L1 = m + (1 - m) / a;
    float L_x = m + a * (x - m);
    float T_x = m * pow(x / m, c) + b;
    float S0 = m + l0;
    float S1 = m + a * l0;
    float C2 = a * P / (P - S1);
    float S_x = P - (P - S1) * exp(-(C2 * (x - S0) / P));
    float w0_x = 1 - W_f(x, 0, m);
    float w2_x = H_f(x, m + l0, m + l0);
    float w1_x = 1 - w0_x - w2_x;
    float f_x = T_x * w0_x + L_x * w1_x + S_x * w2_x;
    return f_x;
}

half3 ApplyTonemap(half3 input
#if _TONEMAP_GT
    , float4 tonemapParams0
    , float4 tonemapParams1
#endif
)
{
#if _TONEMAP_ACES
    float3 aces = unity_to_ACES(input);
    input = AcesTonemap(aces);
#elif _TONEMAP_NEUTRAL
    input = NeutralTonemap(input);
#elif _TONEMAP_GT
    input.r = GranTurismoTonemap(input.r, tonemapParams0.x, tonemapParams0.y, tonemapParams0.z, tonemapParams0.w, tonemapParams1.x, tonemapParams1.y);
    input.g = GranTurismoTonemap(input.g, tonemapParams0.x, tonemapParams0.y, tonemapParams0.z, tonemapParams0.w, tonemapParams1.x, tonemapParams1.y);
    input.b = GranTurismoTonemap(input.b, tonemapParams0.x, tonemapParams0.y, tonemapParams0.z, tonemapParams0.w, tonemapParams1.x, tonemapParams1.y);

#endif

    return saturate(input);
}

half3 ApplyColorGrading(half3 input, float postExposure, TEXTURE2D_PARAM(lutTex, lutSampler), float3 lutParams, TEXTURE2D_PARAM(userLutTex, userLutSampler), float3 userLutParams, float userLutContrib
#if _TONEMAP_GT
    , float4 tonemapParams0
    , float4 tonemapParams1
#endif
)
{
    // Artist request to fine tune exposure in post without affecting bloom, dof etc
    input *= postExposure;

    // HDR Grading:
    //   - Apply internal LogC LUT
    //   - (optional) Clamp result & apply user LUT
    #if _HDR_GRADING
    {
        float3 inputLutSpace = saturate(LinearToLogC(input)); // LUT space is in LogC
        input = ApplyLut2D(TEXTURE2D_ARGS(lutTex, lutSampler), inputLutSpace, lutParams);

        UNITY_BRANCH
        if (userLutContrib > 0.0)
        {
            input = saturate(input);
            input.rgb = GetLinearToSRGB(input.rgb); // In LDR do the lookup in sRGB for the user LUT
            half3 outLut = ApplyLut2D(TEXTURE2D_ARGS(userLutTex, userLutSampler), input, userLutParams);
            input = lerp(input, outLut, userLutContrib);
            input.rgb = GetSRGBToLinear(input.rgb);
        }
    }

    // LDR Grading:
    //   - Apply tonemapping (result is clamped)
    //   - (optional) Apply user LUT
    //   - Apply internal linear LUT
    #else
    {
        #if _TONEMAP_GT
        input = ApplyTonemap(input, tonemapParams0, tonemapParams1);
        #else
        input = ApplyTonemap(input);
        #endif
        

        UNITY_BRANCH
        if (userLutContrib > 0.0)
        {
            input.rgb = GetLinearToSRGB(input.rgb); // In LDR do the lookup in sRGB for the user LUT
            half3 outLut = ApplyLut2D(TEXTURE2D_ARGS(userLutTex, userLutSampler), input, userLutParams);
            input = lerp(input, outLut, userLutContrib);
            input.rgb = GetSRGBToLinear(input.rgb);
        }

        input = ApplyLut2D(TEXTURE2D_ARGS(lutTex, lutSampler), input, lutParams);
    }
    #endif

    return input;
}

half3 ApplyGrain(half3 input, float2 uv, TEXTURE2D_PARAM(GrainTexture, GrainSampler), float intensity, float response, float2 scale, float2 offset, float oneOverPaperWhite)
{
    // Grain in range [0;1] with neutral at 0.5
    half grain = SAMPLE_TEXTURE2D(GrainTexture, GrainSampler, uv * scale + offset).w;

    // Remap [-1;1]
    grain = (grain - 0.5) * 2.0;

    // Noisiness response curve based on scene luminance
    float lum = Luminance(input);
    #ifdef HDR_INPUT
    lum *= oneOverPaperWhite;
    #endif
    lum = 1.0 - sqrt(lum);
    lum = lerp(1.0, lum, response);

    return input + input * grain * intensity * lum;
}

half3 ApplyDithering(half3 input, float2 uv, TEXTURE2D_PARAM(BlueNoiseTexture, BlueNoiseSampler), float2 scale, float2 offset, float paperWhite, float oneOverPaperWhite)
{
    // Symmetric triangular distribution on [-1,1] with maximal density at 0
    float noise = SAMPLE_TEXTURE2D(BlueNoiseTexture, BlueNoiseSampler, uv * scale + offset).a * 2.0 - 1.0;
    noise = FastSign(noise) * (1.0 - sqrt(1.0 - abs(noise)));

#if UNITY_COLORSPACE_GAMMA
    input += noise / 255.0;
#elif defined(HDR_INPUT)
    input = input * oneOverPaperWhite;
    // Do not call GetSRGBToLinear/GetLinearToSRGB because the "fast" version will clamp values!
    input = SRGBToLinear(LinearToSRGB(input) + noise / 255.0);
    input = input * paperWhite;
#else
    input = GetSRGBToLinear(GetLinearToSRGB(input) + noise / 255.0);
#endif

    return input;
}

#define FXAA_SPAN_MAX   (8.0)
#define FXAA_REDUCE_MUL (1.0 / 8.0)
#define FXAA_REDUCE_MIN (1.0 / 128.0)

half3 FXAAFetch(float2 coords, float2 offset, TEXTURE2D_X(inputTexture))
{
    float2 uv = coords + offset;
    return SAMPLE_TEXTURE2D_X(inputTexture, sampler_LinearClamp, uv).xyz;
}

half3 FXAALoad(int2 icoords, int idx, int idy, float4 sourceSize, TEXTURE2D_X(inputTexture))
{
    #if SHADER_API_GLES
    float2 uv = (icoords + int2(idx, idy)) * sourceSize.zw;
    return SAMPLE_TEXTURE2D_X(inputTexture, sampler_PointClamp, uv).xyz;
    #else
    return LOAD_TEXTURE2D_X(inputTexture, clamp(icoords + int2(idx, idy), 0, sourceSize.xy - 1.0)).xyz;
    #endif
}

half3 ApplyFXAA(half3 color, float2 positionNDC, int2 positionSS, float4 sourceSize, TEXTURE2D_X(inputTexture), float paperWhite, float oneOverPaperWhite)
{
    // Edge detection
    half3 rgbNW = FXAALoad(positionSS, -1, -1, sourceSize, inputTexture);
    half3 rgbNE = FXAALoad(positionSS,  1, -1, sourceSize, inputTexture);
    half3 rgbSW = FXAALoad(positionSS, -1,  1, sourceSize, inputTexture);
    half3 rgbSE = FXAALoad(positionSS,  1,  1, sourceSize, inputTexture);

    #ifdef HDR_INPUT
        // The pixel values we have are already tonemapped but in the range [0, 10000] nits. To run FXAA properly, we need to convert them
        // to a SDR range [0; 1]. Since the tonemapped values are not evenly distributed and mostly close to the paperWhite nits value, we can
        // normalize by paperWhite to get most of the scene in [0; 1] range. For the remaining pixels, we can use the FastTonemap() to remap
        // them to [0, 1] range.
        float lumaNW = Luminance(FastTonemap(rgbNW.xyz * oneOverPaperWhite));
        float lumaNE = Luminance(FastTonemap(rgbNE.xyz * oneOverPaperWhite));
        float lumaSW = Luminance(FastTonemap(rgbSW.xyz * oneOverPaperWhite));
        float lumaSE = Luminance(FastTonemap(rgbSE.xyz * oneOverPaperWhite));
        float lumaM = Luminance(FastTonemap(color.xyz * oneOverPaperWhite));
    #else
        rgbNW = saturate(rgbNW);
        rgbNE = saturate(rgbNE);
        rgbSW = saturate(rgbSW);
        rgbSE = saturate(rgbSE);
        color = saturate(color);

        half lumaNW = Luminance(rgbNW);
        half lumaNE = Luminance(rgbNE);
        half lumaSW = Luminance(rgbSW);
        half lumaSE = Luminance(rgbSE);
        half lumaM = Luminance(color);
    #endif

    float2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y = ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    half lumaSum = lumaNW + lumaNE + lumaSW + lumaSE;
    float dirReduce = max(lumaSum * (0.25 * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);
    float rcpDirMin = rcp(min(abs(dir.x), abs(dir.y)) + dirReduce);

    dir = min((FXAA_SPAN_MAX).xx, max((-FXAA_SPAN_MAX).xx, dir * rcpDirMin)) * sourceSize.zw;

    // Blur
    half3 rgb03 = FXAAFetch(positionNDC, dir * (0.0 / 3.0 - 0.5), inputTexture);
    half3 rgb13 = FXAAFetch(positionNDC, dir * (1.0 / 3.0 - 0.5), inputTexture);
    half3 rgb23 = FXAAFetch(positionNDC, dir * (2.0 / 3.0 - 0.5), inputTexture);
    half3 rgb33 = FXAAFetch(positionNDC, dir * (3.0 / 3.0 - 0.5), inputTexture);

    #ifdef HDR_INPUT
        rgb03 = FastTonemap(rgb03 * oneOverPaperWhite);
        rgb13 = FastTonemap(rgb13 * oneOverPaperWhite);
        rgb23 = FastTonemap(rgb23 * oneOverPaperWhite);
        rgb33 = FastTonemap(rgb33 * oneOverPaperWhite);
    #else
        rgb03 = saturate(rgb03);
        rgb13 = saturate(rgb13);
        rgb23 = saturate(rgb23);
        rgb33 = saturate(rgb33);
    #endif

    half3 rgbA = 0.5 * (rgb13 + rgb23);
    half3 rgbB = rgbA * 0.5 + 0.25 * (rgb03 + rgb33);

    half lumaB = Luminance(rgbB);

    half lumaMin = Min3(lumaM, lumaNW, Min3(lumaNE, lumaSW, lumaSE));
    half lumaMax = Max3(lumaM, lumaNW, Max3(lumaNE, lumaSW, lumaSE));

    half3 rgb = ((lumaB < lumaMin) || (lumaB > lumaMax)) ? rgbA : rgbB;

    #ifdef HDR_INPUT
        rgb.xyz = FastTonemapInvert(rgb) * paperWhite;;
    #endif
    return rgb;
}

#endif // UNIVERSAL_POSTPROCESSING_COMMON_INCLUDED
