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


#include <cstdint>
#include <string>
#include <algorithm>
#include <malloc.h>
#include "CombMask.h"
#include "simd.h"



static __forceinline int absdiff(int x, int y) noexcept
{
    return x > y ? x - y : y - x;
}

/*
How to detect combs (quoted from TFM - README.txt written by tritical)

    Assume 5 neighboring pixels (a,b,c,d,e) positioned vertically.

      a
      b
      c
      d
      e

    metric 0:

    d1 = c - b;
    d2 = c - d;
    if ((d1 > cthresh && d2 > cthresh) || (d1 < -cthresh && d2 < -cthresh)) {
        if (abs(a + 4*c + e - 3*(b + d)) > cthresh*6) it's combed;
    }

    metric 1:

    val = (b - c) * (d - c);
    if (val > cthresh * cthresh) it's combed;
*/


static void __stdcall
comb_mask_0_c(uint8_t* dstp, const uint8_t* srcp, const int dpitch,
              const int spitch, const int cthresh, const int width,
              const int height) noexcept
{
    const uint8_t* sc = srcp;
    const uint8_t* sb = sc + spitch;
    const uint8_t* sa = sb + spitch;
    const uint8_t* sd = sc + spitch;
    const uint8_t* se = sd + spitch;

    const int cth6 = cthresh * 6;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            dstp[x] = 0;
            int d1 = sc[x] - sb[x];
            int d2 = sc[x] - sd[x];
            if ((d1 > cthresh && d2 > cthresh)
                    || (d1 < -cthresh && d2 < -cthresh)) {
                int f0 = sa[x] + 4 * sc[x] + se[x];
                int f1 = 3 * (sb[x] + sd[x]);
                if (absdiff(f0, f1) > cth6) {
                    dstp[x] = 0xFF;
                }
            }
        }
        sa = sb;
        sb = sc;
        sc = sd;
        sd = se;
        se += (y < height - 3) ? spitch : -spitch;
        dstp += dpitch;
    }
}


static void __stdcall
comb_mask_1_c(uint8_t* dstp, const uint8_t* srcp, const int dpitch,
              const int spitch, const int cthresh, const int width,
              const int height) noexcept
{
    const uint8_t* sc = srcp;
    const uint8_t* sb = sc + spitch;
    const uint8_t* sd = sc + spitch;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int val = (sb[x] - sc[x]) * (sd[x] - sc[x]);
            dstp[x] = val > cthresh ? 0xFF : 0;
        }
        sb = sc;
        sc = sd;
        sd += (y < height - 2) ? spitch : -spitch;
        dstp += dpitch;
    }
}


template <typename V>
static void __stdcall
comb_mask_0_simd(uint8_t* dstp, const uint8_t* srcp, const int dpitch,
                 const int spitch, const int cthresh, const int width,
                 const int height) noexcept
{
    const uint8_t* sc = srcp;
    const uint8_t* sb = sc + spitch;
    const uint8_t* sa = sb + spitch;
    const uint8_t* sd = sc + spitch;
    const uint8_t* se = sd + spitch;

    int16_t cth16 = static_cast<int16_t>(cthresh);
    const V cthp = set1_i16<V>(cth16);
    const V cthn = set1_i16<V>(-cth16);
    const V cth6 = set1_i16<V>(cth16 * 6);

    constexpr int step = sizeof(V) / 2;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; x += step) {
            V xc = load_half<V>(sc + x);
            V xb = load_half<V>(sb + x);
            V xd = load_half<V>(sd + x);
            V d1 = sub_i16(xc, xb);
            V d2 = sub_i16(xc, xd);
            V mask0 = or_reg(
                and_reg(cmpgt_i16(d1, cthp), cmpgt_i16(d2, cthp)),
                and_reg(cmpgt_i16(cthn, d1), cmpgt_i16(cthn, d2)));
            d2 = mul3(add_i16(xb, xd));
            d1 = add_i16(load_half<V>(sa + x), load_half<V>(se + x));
            d1 = add_i16(d1, lshift_i16(xc, 2));
            mask0 = and_reg(mask0, cmpgt_i16(absdiff_i16(d1, d2), cth6));
            store_half(dstp + x, mask0);
        }
        sa = sb;
        sb = sc;
        sc = sd;
        sd = se;
        se += (y < height - 3) ? spitch : -spitch;
        dstp += dpitch;
    }
}


template <typename V>
static void __stdcall
comb_mask_1_simd(uint8_t* dstp, const uint8_t* srcp, const int dpitch,
                 const int spitch, const int cthresh, const int width,
                 const int height) noexcept
{
    const uint8_t* sc = srcp;
    const uint8_t* sb = sc + spitch;
    const uint8_t* sd = sc + spitch;

    const V cth = set1_i16<V>(static_cast<int16_t>(cthresh));
    const V all = cmpeq_i8(cth, cth);

    constexpr int step = sizeof(V) / 2;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; x += step) {
            V xb = load_half<V>(sb + x);
            V xc = load_half<V>(sc + x);
            V xd = load_half<V>(sd + x);
            xb = sub_i16(xb, xc);
            xd = sub_i16(xd, xc);
            xc = andnot(mulhi(xb, xd), mullo(xb, xd));
            xc = cmpgt_u16(xc, cth, all);
            store_half(dstp + x, xc);
        }
        sb = sc;
        sc = sd;
        sd += (y < height - 2) ? spitch : -spitch;
        dstp += dpitch;
    }
}


static void __stdcall
motion_mask_c(uint8_t* tmpp, uint8_t* dstp, const uint8_t* srcp,
              const uint8_t* prevp, const int tpitch, const int dpitch,
              const int spitch, const int ppitch, const int mthresh,
              const int width, const int height) noexcept
{
    uint8_t* tx = tmpp;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            tx[x] = absdiff(srcp[x], prevp[x]) > mthresh ? 0xFF : 0;
        }
        tx += tpitch;
        srcp += spitch;
        prevp += ppitch;
    }

    const uint8_t* t0 = tmpp;
    const uint8_t *t1 = tmpp;
    const uint8_t *t2 = tmpp + tpitch;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            dstp[x] = (t0[x] | t1[x] | t2[x]);
        }
        t0 = t1;
        t1 = t2;
        if (y < height - 2) {
            t2 += tpitch;
        }
        dstp += dpitch;
    }
}


template <typename V>
static void __stdcall
motion_mask_simd(uint8_t* tmpp, uint8_t* dstp, const uint8_t* srcp,
                 const uint8_t* prevp, const int tpitch, const int dpitch,
                 const int spitch, const int ppitch, const int mthresh,
                 const int width, const int height) noexcept
{
    uint8_t* tx = tmpp;
    const V mth = set1_i8<V>(static_cast<int8_t>(mthresh));
    const V all = cmpeq_i8(mth, mth);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; x += sizeof(V)) {
            V diff = absdiff_u8(load<V>(srcp + x), load<V>(prevp + x));
            V dst = cmpgt_u8(diff, mth, all);
            store(tx + x, dst);
        }
        tx += tpitch;
        srcp += spitch;
        prevp += ppitch;
    }

    const uint8_t* t0 = tmpp;
    const uint8_t* t1 = tmpp;
    const uint8_t* t2 = tmpp + tpitch;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; x += sizeof(V)) {
            V dst = or_reg(load<V>(t0 + x), load<V>(t1 + x));
            dst = or_reg(dst, load<V>(t2 + x));
            store(dstp + x, dst);
        }
        t0 = t1;
        t1 = t2;
        if (y < height - 2) {
            t2 += tpitch;
        }
        dstp += dpitch;
    }
}


static void __stdcall
and_masks_c(uint8_t* dstp, const uint8_t* altp, const int dpitch,
            const int apitch, const int width, const int height) noexcept
{
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            dstp[x] &= altp[x];
        }
        dstp += dpitch;
        altp += apitch;
    }
}


template <typename V>
static void __stdcall
and_masks_simd(uint8_t* dstp, const uint8_t* altp, const int dpitch,
               const int apitch, const int width, const int height) noexcept
{
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; x += sizeof(V)) {
            V d = load<V>(dstp + x);
            V a = load<V>(altp + x);
            store(dstp + x, and_reg(d, a));
        }
        dstp += dpitch;
        altp += apitch;
    }
}


static void __stdcall
expand_mask_c(uint8_t* dstp, uint8_t* srcp, const int dpitch, const int spitch,
              const int width, const int height) noexcept
{
    for (int y = 0; y < height; ++y) {
        srcp[-1] = srcp[0];
        srcp[width] = srcp[width - 1];
        for (int x = 0; x < width; ++x) {
            dstp[x] = (srcp[x - 1] | srcp[x] | srcp[x + 1]);
        }
        srcp += spitch;
        dstp += dpitch;
    }
}


template <typename V>
static void __stdcall
expand_mask_simd(uint8_t* dstp, uint8_t* srcp, const int dpitch,
                 const int spitch, const int width, const int height) noexcept
{
    for (int y = 0; y < height; ++y) {
        srcp[-1] = srcp[0];
        srcp[width] = srcp[width - 1];
        for (int x = 0; x < width; x += sizeof(V)) {
            V s0 = loadu<V>(srcp + x - 1);
            V s1 = load<V>(srcp + x);
            V s2 = loadu<V>(srcp + x + 1);
            stream(dstp + x, or_reg(or_reg(s0, s1), s2));
        }
        srcp += spitch;
        dstp += dpitch;
    }
}


Buffer::Buffer(size_t pitch, int height, int hsize, size_t align, bool ip,
    ise_t* e) :
    env(e), isPlus(ip)
{
    size_t size = pitch * height * hsize + align;
    orig = alloc_buffer(size, align, isPlus, env);
    buffp = reinterpret_cast<uint8_t*>(orig) + align;
}


Buffer::~Buffer()
{
    free_buffer(orig, isPlus, env);
}



CombMask::CombMask(PClip c, int cth, int mth, bool ch, arch_t arch, bool e,
                   int metric, bool plus) :
    GVFmod(c, ch, arch, plus), cthresh(cth), mthresh(mth), expand(e),
    buff(nullptr)
{
    validate(!vi.IsPlanar(), "planar format only.");
    validate(metric != 0 && metric != 1, "metric must be set to 0 or 1.");
    if (metric == 0) {
        validate(cthresh < 0 || cthresh > 255,
                 "cthresh must be between 0 and 255 on metric 0.");
    } else {
        validate(cthresh < 0 || cthresh > 65025,
                 "cthresh must be between 0 and 65025 on metric 1.");
    }
    validate(mthresh < 0 || mthresh > 255, "mthresh must be between 0 and 255.");


    buffPitch = vi.width + align - 1;
    if (expand) {
        buffPitch += 2;
    }
    buffPitch &= (~(align - 1));
    needBuff = mthresh > 0 || expand;

    switch (arch) {
#if defined(__AVX2__)
    case USE_AVX2:
        writeCombMask = metric == 0 ? comb_mask_0_simd<__m256i>
                      : comb_mask_1_simd<__m256i>;
        writeMotionMask = motion_mask_simd<__m256i>;
        andMasks = and_masks_simd<__m256i>;
        expandMask = expand_mask_simd<__m256i>;
        break;
#endif
    case USE_SSE2:
        writeCombMask = metric == 0 ? comb_mask_0_simd<__m128i>
                      : comb_mask_1_simd<__m128i>;
        writeMotionMask = motion_mask_simd<__m128i>;
        andMasks = and_masks_simd<__m128i>;
        expandMask = expand_mask_simd<__m128i>;
        break;
    default:
        writeCombMask = metric == 0 ? comb_mask_0_c : comb_mask_1_c;
        writeMotionMask = motion_mask_c;
        andMasks = and_masks_c;
        expandMask = expand_mask_c;
    }

    if (mthresh > 0 && child->SetCacheHints(CACHE_GET_WINDOW, 0) < 3) {
        child->SetCacheHints(CACHE_WINDOW, 3);
    }

    if (!isPlus && needBuff) {
        buff = new Buffer(buffPitch, vi.height, mthresh > 0 ? 2 : 1, align,
                          false, nullptr);

    }
}


CombMask::~CombMask()
{
    if (!isPlus && needBuff) {
        delete buff;
    }
}


PVideoFrame __stdcall CombMask::GetFrame(int n, ise_t* env)
{
    static const int planes[] = { PLANAR_Y, PLANAR_U, PLANAR_V };

    PVideoFrame src = child->GetFrame(n, env);

    PVideoFrame prev = mthresh == 0 ? PVideoFrame() 
                     : child->GetFrame(std::max(n - 1, 0), env);

    PVideoFrame dst = env->NewVideoFrame(vi, align);

    Buffer* b = buff;
    uint8_t *buffp, *tmpp;
    if (needBuff) {
        if (isPlus) {
            b = new Buffer(buffPitch, vi.height, mthresh > 0 ? 2 : 1, align,
                           isPlus, env);
        }
        buffp = b->buffp;
        tmpp = buffp + vi.height * buffPitch;
    }

    for (int p = 0; p < numPlanes; ++p) {
        const int plane = planes[p];

        const uint8_t* srcp = src->GetReadPtr(plane);
        uint8_t* dstp = dst->GetWritePtr(plane);
        const int spitch = src->GetPitch(plane);
        const int dpitch = dst->GetPitch(plane);
        const int width = src->GetRowSize(plane);
        const int height = src->GetHeight(plane);

        if (!needBuff) {
            writeCombMask(dstp, srcp, dpitch, spitch, cthresh, width, height);
            continue;
        }

        writeCombMask(buffp, srcp, buffPitch, spitch, cthresh, width, height);

        if (mthresh == 0) {
            expandMask(dstp, buffp, dpitch, buffPitch, width, height);
            continue;
        }

        writeMotionMask(tmpp, dstp, srcp, prev->GetReadPtr(plane), buffPitch,
                        dpitch, spitch, prev->GetPitch(plane), mthresh, width,
                        height);

        if (!expand) {
            andMasks(dstp, buffp, dpitch, buffPitch, width, height);
            continue;
        }

        andMasks(buffp, dstp, buffPitch, dpitch, width, height);

        expandMask(dstp, buffp, dpitch, buffPitch, width, height);
    }

    if (isPlus && needBuff) {
        delete b;
    }

    return dst;
}

