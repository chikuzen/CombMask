#include "CombMask.h"
#include "simd.h"


template <typename V>
static bool __stdcall
check_combed_simd(PVideoFrame& cmask, int mi, int blockx, int blocky,
                  bool is_avsplus, ise_t* env)
{
    const int width = cmask->GetRowSize(PLANAR_Y) & (~(blockx - 1));
    const int height = cmask->GetHeight(PLANAR_Y) & (~(blocky - 1));
    const int pitch = cmask->GetPitch(PLANAR_Y);

    const uint8_t* srcp = cmask->GetReadPtr(PLANAR_Y);

    size_t pitch_a = (width + 31) & (~31);
    uint8_t* arr = reinterpret_cast<uint8_t*>(
        alloc_buffer(pitch_a * 4, 32, is_avsplus, env));
    int64_t* array[] = {
        reinterpret_cast<int64_t*>(arr),
        reinterpret_cast<int64_t*>(arr + pitch_a),
        reinterpret_cast<int64_t*>(arr + pitch_a * 2),
        reinterpret_cast<int64_t*>(arr + pitch_a * 3),
    };
    int length = width / sizeof(int64_t);
    int stepx = blockx / 8;
    int stepy = blocky / 8;

    const V zero = setzero<V>();

    for (int y = 0; y < height; y += 32) {
        for (int j = 0; j < 4; ++j) {
            for (int x = 0; x < width; x += sizeof(V)) {
                // 0xFF == -1, thus the range of each bytes of sum is -8 to 0.
                V sum = load<V>(srcp + x);
                sum = add_i8(sum, load<V>(srcp + x + pitch * 1));
                sum = add_i8(sum, load<V>(srcp + x + pitch * 2));
                sum = add_i8(sum, load<V>(srcp + x + pitch * 3));
                sum = add_i8(sum, load<V>(srcp + x + pitch * 4));
                sum = add_i8(sum, load<V>(srcp + x + pitch * 5));
                sum = add_i8(sum, load<V>(srcp + x + pitch * 6));
                sum = add_i8(sum, load<V>(srcp + x + pitch * 7));
                sum = sad_u8(sub_i8(zero, sum), zero);
                store(arr + x + pitch_a * j, sum);
            }
            srcp += pitch * 8;
        }

        for (int xx = 0; xx < length; xx += stepx) {
            int64_t sum = 0;
            for (int by = 0; by < stepy; ++by) {
                for (int bx = 0; bx < stepx; ++bx) {
                    sum += array[by][xx + bx];
                }
            }
            if (sum > mi) {
                free_buffer(arr, is_avsplus, env);
                return true;
            }
        }
    }
    free_buffer(arr, is_avsplus, env);
    return false;
}


static bool __stdcall
check_combed_c(PVideoFrame& cmask, int mi, int blockx, int blocky, bool, ise_t*)
{
    const int width = cmask->GetRowSize(PLANAR_Y) & (~(blockx - 1));
    const int height = cmask->GetHeight(PLANAR_Y) & (~(blocky - 1));
    const int pitch = cmask->GetPitch(PLANAR_Y);

    const uint8_t* srcp = cmask->GetReadPtr(PLANAR_Y);

    for (int y = 0; y < height; y += blocky) {
        for (int x = 0; x < width; x += blockx) {
            int count = 0;
            for (int i = 0; i < blocky; ++i) {
                for (int j = 0; j < blockx; ++j) {
                    count += (srcp[x + j + i * pitch] & 1);
                }
            }
            if (count > mi) {
                return true;
            }
        }
        srcp += pitch * blocky;
    }
    return false;
}


template <typename V>
static void __stdcall
merge_frames_simd(int num_planes, PVideoFrame& src, PVideoFrame& alt,
                  PVideoFrame& mask, PVideoFrame& dst)
{
    static const int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

    for (int p = 0; p < num_planes; ++p) {
        const int plane = planes[p];

        const uint8_t* srcp = src->GetReadPtr(plane);
        const uint8_t* altp = alt->GetReadPtr(plane);
        const uint8_t* mskp = mask->GetReadPtr(plane);
        uint8_t* dstp = dst->GetWritePtr(plane);

        const int width = src->GetRowSize(plane);
        const int height = src->GetHeight(plane);

        const int spitch = src->GetPitch(plane);
        const int apitch = alt->GetPitch(plane);
        const int mpitch = mask->GetPitch(plane);
        const int dpitch = dst->GetPitch(plane);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x += sizeof(V)) {
                const V s = load<V>(srcp + x);
                const V a = load<V>(altp + x);
                const V m = load<V>(mskp + x);

                stream(dstp + x, blendv(s, a, m));
            }
            srcp += spitch;
            altp += apitch;
            mskp += mpitch;
            dstp += dpitch;
        }
    }
}


static void __stdcall
merge_frames_c(int num_planes, PVideoFrame& src, PVideoFrame& alt,
               PVideoFrame& mask, PVideoFrame& dst)
{
    static const int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

    for (int p = 0; p < num_planes; ++p) {
        const int plane = planes[p];
        const uint8_t* srcp = src->GetReadPtr(plane);
        const uint8_t* altp = alt->GetReadPtr(plane);
        const uint8_t* mskp = mask->GetReadPtr(plane);
        uint8_t* dstp = dst->GetWritePtr(plane);

        const int width = src->GetRowSize(plane);
        const int height = src->GetHeight(plane);

        const int spitch = src->GetPitch(plane);
        const int apitch = alt->GetPitch(plane);
        const int mpitch = mask->GetPitch(plane);
        const int dpitch = dst->GetPitch(plane);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                dstp[x] = (srcp[x] & (~mskp[x])) | (altp[x] & mskp[x]);
            }
            srcp += spitch;
            altp += apitch;
            mskp += mpitch;
            dstp += dpitch;
        }
    }
}




check_combed_t get_check_combed(arch_t arch)
{

#if defined(__AVX2__)
    if (arch == USE_AVX2) {
        return check_combed_simd<__m256i>;
    }
#endif
    if (arch == USE_SSE2) {
        return check_combed_simd<__m128i>;
    }
    return check_combed_c;
}



MaskedMerge::
MaskedMerge(PClip c, PClip a, PClip m, int _mi, int bx, int by, bool chroma,
            arch_t arch, bool ip) :
    GVFmod(c, chroma, arch, ip), altc(a), maskc(m), mi(_mi), blockx(bx),
    blocky(by)
{
    validate(!vi.IsPlanar(), "planar format only.");
    validate(mi < 0 || mi > 128, "mi must be between 0 and 128.");
    validate(blockx < 8 || blockx > 32 || blockx % 8 > 0,
             "blockx must be set to 8, 16 or 32.");
    validate(blocky < 8 || blocky > 32 || blocky % 8 > 0,
             "blocky must be set to 8, 16 or 32.");

    const VideoInfo& a_vi = altc->GetVideoInfo();
    const VideoInfo& m_vi = maskc->GetVideoInfo();
    validate(!vi.IsSameColorspace(a_vi) || !vi.IsSameColorspace(m_vi),
             "unmatch colorspaces.");
    validate(vi.width != a_vi.width || vi.width != m_vi.width ||
             vi.height != a_vi.height || vi.height != m_vi.height,
             "unmatch resolutions.");

    switch (arch) {
#if defined(__AVX2__)
    case USE_AVX2:
        mergeFrames = merge_frames_simd<__m256i>;
        break;
#endif
    case USE_SSE2:
        mergeFrames = merge_frames_simd<__m128i>;
        break;
    default:
        mergeFrames = merge_frames_c;
    }

    checkCombed = get_check_combed(arch);
}


PVideoFrame __stdcall MaskedMerge::GetFrame(int n, ise_t* env)
{
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame mask = maskc->GetFrame(n, env);
    if (mi > 0 && !checkCombed(mask, mi, blockx, blocky, isPlus, env)) {
        return src;
    }

    PVideoFrame alt = altc->GetFrame(n, env);
    PVideoFrame dst = env->NewVideoFrame(vi);

    mergeFrames(numPlanes, src, alt, mask, dst);

    if (numPlanes == 1 && !vi.IsY8()) {
        const int src_pitch = src->GetPitch(PLANAR_U);
        const int dst_pitch = dst->GetPitch(PLANAR_U);
        const int width = src->GetRowSize(PLANAR_U);
        const int height = src->GetHeight(PLANAR_U);
        env->BitBlt(dst->GetWritePtr(PLANAR_U), dst_pitch,
            src->GetReadPtr(PLANAR_U), src_pitch, width, height);
        env->BitBlt(dst->GetWritePtr(PLANAR_V), dst_pitch,
            src->GetReadPtr(PLANAR_V), src_pitch, width, height);
    }

    return dst;
}

