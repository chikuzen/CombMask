/*
  CombMask for AviSynth2.6x

  Copyright (C) 2013 Oka Motofumi

  Authors: Oka Motofumi (chikuzen.mo at gmail dot com)

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program; if not, write to the Free Software
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
*/


#include <emmintrin.h>
#include <windows.h>
#include "avisynth.h"

#define CMASK_VERSION "0.0.1"

static const AVS_Linkage* AVS_linkage = 0;


static void __stdcall
write_mmask_sse2(int num_planes, int mthresh, PVideoFrame& src, PVideoFrame& prev)
{
    const int planes[] = {PLANAR_Y, PLANAR_U, PLANAR_V};
    const __m128i xmth = _mm_set1_epi8((char)mthresh);
    const __m128i zero = _mm_setzero_si128();
    const __m128i all1 = _mm_cmpeq_epi32(zero, zero);

    for (int p = 0; p < num_planes; p++) {
        const int width = (src->GetRowSize(planes[p]) + 15) / 16;
        const int height = src->GetHeight(planes[p]);
        const int src_pitch = src->GetPitch(planes[p]) / 16;
        const int prev_pitch = prev->GetPitch(planes[p]) / 16;

        const __m128i* srcp = (__m128i*)src->GetReadPtr(planes[p]);
        __m128i* prevp = (__m128i*)prev->GetWritePtr(planes[p]);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                __m128i xmm0 = _mm_load_si128(srcp + x);
                __m128i xmm1 = _mm_load_si128(prevp + x);

                __m128i xmm2 = _mm_subs_epu8(xmm0, xmm1);
                xmm1 = _mm_subs_epu8(xmm1, xmm0);
                xmm0 = _mm_or_si128(xmm1, xmm2);
                xmm1 = _mm_subs_epu8(xmm0, xmth);
                xmm1 = _mm_cmpeq_epi8(xmm1, zero);
                xmm1 = _mm_andnot_si128(xmm1, all1);
                _mm_store_si128(prevp + x, xmm1);
            }
            srcp += src_pitch;
            prevp += prev_pitch;
        }

        __m128i* prevpt = (__m128i*)prev->GetWritePtr(planes[p]);
        __m128i* prevpc = prevpt + prev_pitch;
        __m128i* prevpb = prevpc + prev_pitch;

        for (int y = 1; y < height - 1; y++) {
            for (int x = 0; x < width; x++) {
                __m128i xmm0 = _mm_load_si128(prevpt + x);
                __m128i xmm1 = _mm_load_si128(prevpc + x);
                __m128i xmm2 = _mm_load_si128(prevpb + x);
                xmm0 = _mm_and_si128(xmm0, xmm2);
                xmm1 = _mm_or_si128(xmm1, xmm0);
                _mm_store_si128(prevpc + x, xmm1);
            }
            prevpt += prev_pitch;
            prevpc += prev_pitch;
            prevpb += prev_pitch;
        }
    }
}


static void __stdcall
write_mmask_c(const int num_planes, const int mthresh, PVideoFrame& src, PVideoFrame& prev)
{
    const int planes[] = {PLANAR_Y, PLANAR_U, PLANAR_V};

    for (int p = 0; p < num_planes; p++) {
        int width = src->GetRowSize(planes[p]);
        const int height = src->GetHeight(planes[p]);
        const int src_pitch = src->GetPitch(planes[p]);
        int prv_pitch = prev->GetPitch(planes[p]);

        const BYTE* srcp = src->GetReadPtr(planes[p]);
        BYTE* prvp = prev->GetWritePtr(planes[p]);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                prvp[x] = abs((int)srcp[x] - prvp[x]) > mthresh ? 0xFF : 0x00;
            }
            srcp += src_pitch;
            prvp += prv_pitch;
        }

        width = (width + 3) / 4;
        prv_pitch /= 4;

        DWORD* prvt = (DWORD*)prev->GetWritePtr(planes[p]);
        DWORD* prvc = prvt + prv_pitch;
        DWORD* prvb = prvc + prv_pitch;

        for (int y = 1; y < height - 1; y++) {
            for (int x = 0; x < width; x++) {
                prvc[x] |= (prvt[x] & prvb[x]);
            }
            prvt += prv_pitch;
            prvc += prv_pitch;
            prvb += prv_pitch;
        }
    }
}


/*
How to detect combs (quoted from TFM - README.txt written by tritical)

    Assume 5 neighboring pixels (a,b,c,d,e) positioned vertically.

      a
      b
      c
      d
      e

    d1 = c - b;
    d2 = c - d;
    if ((d1 > cthresh && d2 > cthresh) || (d1 < -cthresh && d2 < -cthresh)) {
        if (abs(a+4*c+e-3*(b+d)) > cthresh*6) it's combed;
    }
--------------------------------------------------------------------------------

    x = a - b -> -x = b - a

    (d1 > cthresh && d2 > cthresh)    == !(d1 <= cthresh || d2 <= cthresh)
                                      == !(d1 - cthresh <= 0 || d2 - cthresh <= 0)

    (d1 < -cthresh && d2 < -cthresh)) == (-d1 > cthresh && -d2 > cthresh)
                                      == !(-d1 - cthresh <= 0 || -d2 - cthresh <= 0)

    !A || !B == !(A && B)

    abs(x) > cthresh * 6 == (x > cthresh * 6 || x < cthresh * -6)
*/

static void __stdcall
write_cmask_sse2(int num_planes, int cthresh, PVideoFrame& src, PVideoFrame& dst)
{
    const int planes[] = {PLANAR_Y, PLANAR_U, PLANAR_V};

    const __m128i xcth = _mm_set1_epi8((char)cthresh);
    const __m128i xct6p = _mm_set1_epi16((short)(cthresh * 6));
    const __m128i xct6n = _mm_set1_epi16((short)(cthresh * -6));
    const __m128i zero = _mm_setzero_si128();

    for (int p = 0; p < num_planes; p++) {
        const int src_pitch = src->GetPitch(planes[p]) / 16;
        const int width = (src->GetRowSize(planes[p]) + 15) / 16;
        const int height = src->GetHeight(planes[p]);

        const __m128i* srcpc = (__m128i*)src->GetReadPtr(planes[p]);
        const __m128i* srcpb = srcpc + src_pitch;
        const __m128i* srcpa = srcpb + src_pitch;
        const __m128i* srcpd = srcpc + src_pitch;
        const __m128i* srcpe = srcpd + src_pitch;

        __m128i* dstp = (__m128i*)dst->GetWritePtr(planes[p]);
        const int dst_pitch = dst->GetPitch(planes[p]) / 16;

        memset(dstp, 0, dst_pitch * height * 16);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                __m128i xmm0 = _mm_load_si128(srcpc + x);
                __m128i xmm1 = _mm_load_si128(srcpb + x);
                __m128i xmm2 = _mm_load_si128(srcpd + x);

                __m128i xmm3 = _mm_or_si128(_mm_cmpeq_epi8(zero, _mm_subs_epu8(_mm_subs_epu8(xmm0, xmm1), xcth)),
                                            _mm_cmpeq_epi8(zero, _mm_subs_epu8(_mm_subs_epu8(xmm0, xmm2), xcth)));

                __m128i xmm4 = _mm_or_si128(_mm_cmpeq_epi8(zero, _mm_subs_epu8(_mm_subs_epu8(xmm1, xmm0), xcth)),
                                            _mm_cmpeq_epi8(zero, _mm_subs_epu8(_mm_subs_epu8(xmm2, xmm0), xcth)));

                xmm3 = _mm_and_si128(xmm3, xmm4);

                xmm4 = _mm_add_epi16(_mm_unpacklo_epi8(xmm1, zero), _mm_unpacklo_epi8(xmm2, zero)); // lo of (b+d)
                xmm1 = _mm_add_epi16(_mm_unpackhi_epi8(xmm1, zero), _mm_unpackhi_epi8(xmm2, zero)); // hi of (b+d)
                xmm4 = _mm_add_epi16(xmm4, _mm_add_epi16(xmm4, xmm4));      // lo of 3*(b+d)
                xmm1 = _mm_add_epi16(xmm1, _mm_add_epi16(xmm1, xmm1));      // hi of 3*(b+d)

                xmm4 = _mm_sub_epi16(_mm_slli_epi16(_mm_unpacklo_epi8(xmm0, zero), 2), xmm4); // lo of 4*c-3*(b+d)
                xmm1 = _mm_sub_epi16(_mm_slli_epi16(_mm_unpackhi_epi8(xmm0, zero), 2), xmm1); // hi of 4*c-3*(b+d)

                xmm0 = _mm_load_si128(srcpa + x);
                xmm2 = _mm_load_si128(srcpe + x);
                xmm4 = _mm_add_epi16(xmm4, _mm_unpacklo_epi8(xmm0, zero)); // lo of a+4*c-3*(b+d)
                xmm1 = _mm_add_epi16(xmm1, _mm_unpackhi_epi8(xmm0, zero)); // hi of a+4*c-3*(b+d)
                xmm4 = _mm_add_epi16(xmm4, _mm_unpacklo_epi8(xmm2, zero)); // lo of a+4*c+e-3*(b+d)
                xmm1 = _mm_add_epi16(xmm1, _mm_unpackhi_epi8(xmm2, zero)); // hi of a+4*c+e-3*(b+d)

                xmm4 = _mm_or_si128(_mm_cmpgt_epi16(xmm4, xct6p), _mm_cmplt_epi16(xmm4, xct6n));
                xmm1 = _mm_or_si128(_mm_cmpgt_epi16(xmm1, xct6p), _mm_cmplt_epi16(xmm1, xct6n));
                xmm1 = _mm_packs_epi16(xmm4, xmm1);

                xmm3 = _mm_andnot_si128(xmm3, xmm1);

                _mm_store_si128(dstp + x, xmm3);
            }
            dstp += dst_pitch;
            srcpa = srcpb;
            srcpb = srcpc;
            srcpc = srcpd;
            srcpd = srcpe;
            srcpe = (y < height - 2) ? srcpe + src_pitch : srcpe - src_pitch;
        }
    }
}


static void __stdcall
write_cmask_c(int num_planes, int cthresh, PVideoFrame& src, PVideoFrame& dst)
{
    const int planes[] = {PLANAR_Y, PLANAR_U, PLANAR_V};

    for (int p = 0; p < num_planes; p++) {
        const int width = src->GetRowSize(planes[p]);
        const int height = src->GetHeight(planes[p]);
        const int src_pitch = src->GetPitch(planes[p]);
        const int dst_pitch = dst->GetPitch(planes[p]);

        const BYTE* srcpc = src->GetReadPtr(planes[p]);
        const BYTE* srcpb = srcpc + src_pitch;
        const BYTE* srcpa = srcpb + src_pitch;
        const BYTE* srcpd = srcpc + src_pitch;
        const BYTE* srcpe = srcpc + src_pitch;

        BYTE* dstp = dst->GetWritePtr(planes[p]);

        memset(dstp, 0, dst_pitch * height);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int d1 = srcpc[x] - srcpb[x];
                int d2 = srcpc[x] - srcpd[x];
                if ((d1 > cthresh && d2 > cthresh) || (d1 < -cthresh && d2 < -cthresh)) {
                    if (abs(srcpa[x] + 4 * srcpc[x] + srcpe[x] - 3 * (srcpb[x] + srcpd[x])) > cthresh * 6) {
                        dstp[x] = 0xFF;
                    }
                }
            }
            srcpa = srcpb;
            srcpb = srcpc;
            srcpc = srcpd;
            srcpd = srcpe;
            srcpe = (y < height - 2) ? srcpe + src_pitch : srcpe - src_pitch;
            dstp += dst_pitch;
        }
    }
}


static void __stdcall
c_and_m_sse2(int num_planes, PVideoFrame& cmask, PVideoFrame& mmask)
{
    const int planes[] = {PLANAR_Y, PLANAR_U, PLANAR_V};

    for (int p = 0; p < num_planes; p++) {
        __m128i* cp = (__m128i*)cmask->GetWritePtr(planes[p]);
        const __m128i* mp = (__m128i*)mmask->GetReadPtr(planes[p]);

        const int pitch_c = cmask->GetPitch(planes[p]) / 16;
        const int pitch_m = mmask->GetPitch(planes[p]) / 16;
        const int width = (cmask->GetRowSize(planes[p]) + 15) / 16;
        const int height = cmask->GetHeight(planes[p]);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                __m128i xmm0 = _mm_load_si128(cp + x);
                __m128i xmm1 = _mm_load_si128(mp + x);
                xmm0 = _mm_and_si128(xmm0, xmm1);
                _mm_store_si128(cp + x, xmm0);
            }
            cp += pitch_c;
            mp += pitch_m;
        }
    }
}


static void __stdcall
c_and_m_c(int num_planes, PVideoFrame& cmask, PVideoFrame& mmask)
{
    const int planes[] = {PLANAR_Y, PLANAR_U, PLANAR_V};

    for (int p = 0; p < num_planes; p++) {
        DWORD* cmskp = (DWORD*)cmask->GetWritePtr(planes[p]);
        const DWORD* mmskp = (DWORD*)mmask->GetReadPtr(planes[p]);

        const int pitch_c = cmask->GetPitch(planes[p]) / 4;
        const int pitch_m = mmask->GetPitch(planes[p]) / 4;
        const int width = (cmask->GetRowSize(planes[p]) + 3) / 4;
        const int height = cmask->GetHeight(planes[p]);

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                cmskp[x] &= mmskp[x];
            }
            cmskp += pitch_c;
            mmskp += pitch_m;
        }
    }
}


class CombMask : public GenericVideoFilter {
    int cthresh;
    int mthresh;
    int num_planes;

    void (__stdcall *write_motion_mask)(int num_planes, int mthresh, PVideoFrame& src, PVideoFrame& prev);
    void (__stdcall *write_comb_mask)(int num_planes, int cthresh, PVideoFrame& src, PVideoFrame& dst);
    void (__stdcall *comb_and_motion)(int num_planes, PVideoFrame& cmask, PVideoFrame& mmask);

public:
    CombMask(PClip c, int cth, int mth, bool sse2, IScriptEnvironment* env);
    ~CombMask() { }
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
};


CombMask::CombMask(PClip c, int cth, int mth, bool sse2, IScriptEnvironment* env)
    : GenericVideoFilter(c), cthresh(cth), mthresh(mth)
{
    if (cthresh < 0 || cthresh > 255) {
        env->ThrowError("CombMask: cthresh must be between 0 and 255.");
    }

    if (mthresh < 0 || mthresh > 255) {
        env->ThrowError("CombMask: mthresh must be between 0 and 255.");
    }

    if (!vi.IsPlanar()) {
        env->ThrowError("CombMask: planar format only.");
    }

    num_planes = vi.IsY8() ? 1 : 3;

    if (!(env->GetCPUFlags() & CPUF_SSE2) && sse2) {
        sse2 = false;
    }

    write_motion_mask = sse2 ? write_mmask_sse2 : write_mmask_c;
    write_comb_mask = sse2 ? write_cmask_sse2 : write_cmask_c;
    comb_and_motion = sse2 ? c_and_m_sse2 : c_and_m_c;
}


PVideoFrame __stdcall CombMask::GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame cmask = env->NewVideoFrame(vi);
    write_comb_mask(num_planes, cthresh, src, cmask);

    if (mthresh > 0) {
        PVideoFrame mmask = child->GetFrame(n == 0 ? 1 : n - 1, env);
        env->MakeWritable(&mmask);
        write_motion_mask(num_planes, mthresh, src, mmask);
        comb_and_motion(num_planes, cmask, mmask);
    }

    return cmask;
}


static AVSValue __cdecl
create_combmask(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    return new CombMask(args[0].AsClip(),  args[1].AsInt(6), args[2].AsInt(9),
                         args[3].AsBool(true), env);
}


extern "C" __declspec(dllexport) const char* __stdcall
AvisynthPluginInit3(IScriptEnvironment* env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;
    env->AddFunction("CombMask", "c[cthresh]i[mthresh]i[sse2]b", create_combmask, 0);
    return "CombMask filter for Avisynth2.6 version "CMASK_VERSION;
}
