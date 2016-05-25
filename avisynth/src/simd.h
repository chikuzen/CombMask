#ifndef COMB_MASK_SIMD_H
#define COMB_MASK_SIMD_H


#include <cstdint>
#if defined(__AVX2__)
    #include <immintrin.h>
#else
    #include <emmintrin.h>
#endif

#define FINLINE __forceinline
#define SFINLINE static __forceinline


template <typename V>
SFINLINE V load(const uint8_t* p);

template <typename V>
SFINLINE V loadu(const uint8_t* p);

template <typename V>
SFINLINE V load_half(const uint8_t* p);

template <typename V>
SFINLINE V set1_i16(int16_t val);

template <typename V>
SFINLINE V set1_i8(int8_t val);

template <typename V>
SFINLINE V setzero();



template <>
FINLINE __m128i load(const uint8_t* p)
{
    return _mm_load_si128(reinterpret_cast<const __m128i*>(p));
}

template <>
FINLINE __m128i loadu(const uint8_t* p)
{
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
}

template <>
FINLINE __m128i load_half(const uint8_t* p)
{
    __m128i t = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
    return _mm_unpacklo_epi8(t, _mm_setzero_si128());
}

template <>
FINLINE __m128i set1_i16(int16_t val)
{
    return _mm_set1_epi16(val);
}

template <>
FINLINE __m128i set1_i8(int8_t val)
{
    return _mm_set1_epi8(val);
}

template <>
FINLINE __m128i setzero()
{
    return _mm_setzero_si128();
}

SFINLINE void store_half(uint8_t* p, const __m128i& x)
{
    __m128i t = _mm_packs_epi16(x, x);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(p), t);
}

SFINLINE void store_half_us(uint8_t* p, const __m128i& x)
{
    __m128i t = _mm_packus_epi16(x, x);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(p), t);
}

SFINLINE void store(uint8_t* p, const __m128i& x)
{
    _mm_store_si128(reinterpret_cast<__m128i*>(p), x);
}

SFINLINE void stream(uint8_t* p, const __m128i& x)
{
    _mm_stream_si128(reinterpret_cast<__m128i*>(p), x);
}

SFINLINE __m128i add_i16(const __m128i& x, const __m128i& y)
{
    return _mm_add_epi16(x, y);
}

SFINLINE __m128i add_i8(const __m128i& x, const __m128i& y)
{
    return _mm_add_epi8(x, y);
}

SFINLINE __m128i sub_i16(const __m128i& x, const __m128i& y)
{
    return _mm_sub_epi16(x, y);
}

SFINLINE __m128i sub_i8(const __m128i& x, const __m128i& y)
{
    return _mm_sub_epi8(x, y);
}

SFINLINE __m128i subs(const __m128i& x, const __m128i& y)
{
    return _mm_subs_epu8(x, y);
}

SFINLINE __m128i mullo(const __m128i& x, const __m128i& y)
{
    return _mm_mullo_epi16(x, y);
}

SFINLINE __m128i mulhi(const __m128i& x, const __m128i& y)
{
    return _mm_mulhi_epi16(x, y);
}

SFINLINE __m128i or_reg(const __m128i& x, const __m128i& y)
{
    return _mm_or_si128(x, y);
}

SFINLINE __m128i xor_reg(const __m128i& x, const __m128i& y)
{
    return _mm_xor_si128(x, y);
}

SFINLINE __m128i and_reg(const __m128i& x, const __m128i& y)
{
    return _mm_and_si128(x, y);
}

SFINLINE __m128i andnot(const __m128i& x, const __m128i& y)
{
    return _mm_andnot_si128(x, y);
}

SFINLINE __m128i min_i16(const __m128i& x, const __m128i& y)
{
    return _mm_min_epi16(x, y);
}

SFINLINE __m128i min_u16(const __m128i& x, const __m128i& y)
{
    return _mm_subs_epu16(x, _mm_subs_epu16(x, y));
}

SFINLINE __m128i min_u8(const __m128i& x, const __m128i& y)
{
    return _mm_min_epu8(x, y);
}

SFINLINE __m128i max_i16(const __m128i& x, const __m128i& y)
{
    return _mm_max_epi16(x, y);
}

SFINLINE __m128i max_u16(const __m128i& x, const __m128i& y)
{
    return _mm_adds_epu16(y, _mm_subs_epu16(x, y));
}

SFINLINE __m128i max_u8(const __m128i& x, const __m128i& y)
{
    return _mm_max_epu8(x, y);
}

SFINLINE __m128i cmpeq_i8(const __m128i& x, const __m128i& y)
{
    return _mm_cmpeq_epi8(x, y);
}

SFINLINE __m128i cmpeq_i16(const __m128i& x, const __m128i& y)
{
    return _mm_cmpeq_epi16(x, y);
}

SFINLINE __m128i cmpgt_i16(const __m128i& x, const __m128i& y)
{
    return _mm_cmpgt_epi16(x, y);
}

SFINLINE __m128i absdiff_i16(const __m128i& x, const __m128i& y)
{
    return max_i16(sub_i16(x, y), sub_i16(y, x));
}

SFINLINE __m128i lshift_i16(const __m128i& x, int n)
{
    return _mm_slli_epi16(x, n);
}

SFINLINE __m128i sad_u8(const __m128i& x, const __m128i& y)
{
    return _mm_sad_epu8(x, y);
}

SFINLINE __m128i blendv(const __m128i& x, const __m128i& y, const __m128i& m)
{
    return or_reg(and_reg(m, y), andnot(m, x));
}

#if defined(__AVX2__)

template <>
FINLINE __m256i load(const uint8_t* p)
{
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(p));
}

template<>
FINLINE __m256i loadu(const uint8_t* p)
{
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}

template <>
FINLINE __m256i load_half(const uint8_t* p)
{
    __m128i t = _mm_load_si128(reinterpret_cast<const __m128i*>(p));
    return _mm256_cvtepu8_epi16(t);
}

template <>
FINLINE __m256i set1_i16(int16_t val)
{
    return _mm256_set1_epi16(val);
}

template <>
FINLINE __m256i set1_i8(int8_t val)
{
    return _mm256_set1_epi8(val);
}

template <>
FINLINE __m256i setzero()
{
    return _mm256_setzero_si256();
}

SFINLINE void store_half(uint8_t* p, const __m256i& x)
{
    __m256i t = _mm256_packs_epi16(x, _mm256_permute2x128_si256(x, x, 0x01));
    _mm_store_si128(reinterpret_cast<__m128i*>(p), _mm256_extracti128_si256(t, 0));
}

SFINLINE void store(uint8_t* p, const __m256i& x)
{
    _mm256_store_si256(reinterpret_cast<__m256i*>(p), x);
}

SFINLINE void stream(uint8_t* p, const __m256i& x)
{
    _mm256_stream_si256(reinterpret_cast<__m256i*>(p), x);
}

SFINLINE __m256i add_i16(const __m256i& x, const __m256i& y)
{
    return _mm256_add_epi16(x, y);
}

SFINLINE __m256i add_i8(const __m256i& x, const __m256i& y)
{
    return _mm256_add_epi8(x, y);
}

SFINLINE __m256i sub_i16(const __m256i& x, const __m256i& y)
{
    return _mm256_sub_epi16(x, y);
}

SFINLINE __m256i sub_i8(const __m256i& x, const __m256i& y)
{
    return _mm256_sub_epi8(x, y);
}

SFINLINE __m256i subs(const __m256i& x, const __m256i& y)
{
    return _mm256_subs_epu8(x, y);
}

SFINLINE __m256i mullo(const __m256i& x, const __m256i& y)
{
    return _mm256_mullo_epi16(x, y);
}

SFINLINE __m256i mulhi(const __m256i& x, const __m256i& y)
{
    return _mm256_mulhi_epi16(x, y);
}

SFINLINE __m256i or_reg(const __m256i& x, const __m256i& y)
{
    return _mm256_or_si256(x, y);
}

SFINLINE __m256i xor_reg(const __m256i& x, const __m256i& y)
{
    return _mm256_xor_si256(x, y);
}

SFINLINE __m256i and_reg(const __m256i& x, const __m256i& y)
{
    return _mm256_and_si256(x, y);
}

SFINLINE __m256i andnot(const __m256i& x, const __m256i& y)
{
    return _mm256_andnot_si256(x, y);
}

SFINLINE __m256i min_i16(const __m256i& x, const __m256i& y)
{
    return _mm256_min_epi16(x, y);
}

SFINLINE __m256i min_u16(const __m256i& x, const __m256i& y)
{
    return _mm256_min_epu16(x, y);
}

SFINLINE __m256i min_u8(const __m256i& x, const __m256i& y)
{
    return _mm256_min_epu8(x, y);
}

SFINLINE __m256i max_i16(const __m256i& x, const __m256i& y)
{
    return _mm256_max_epi16(x, y);
}

SFINLINE __m256i max_u16(const __m256i& x, const __m256i& y)
{
    return _mm256_max_epu16(x, y);
}

SFINLINE __m256i max_u8(const __m256i& x, const __m256i& y)
{
    return _mm256_max_epu8(x, y);
}

SFINLINE __m256i cmpgt_i16(const __m256i& x, const __m256i& y)
{
    return _mm256_cmpgt_epi16(x, y);
}

SFINLINE __m256i cmpeq_i8(const __m256i& x, const __m256i& y)
{
    return _mm256_cmpeq_epi8(x, y);
}

SFINLINE __m256i cmpeq_i16(const __m256i& x, const __m256i& y)
{
    return _mm256_cmpeq_epi16(x, y);
}

SFINLINE __m256i absdiff_i16(const __m256i& x, const __m256i& y)
{
    return _mm256_abs_epi16(sub_i16(x, y));
}

SFINLINE __m256i lshift_i16(const __m256i& x, int n)
{
    return _mm256_slli_epi16(x, n);
}

SFINLINE __m256i sad_u8(const __m256i& x, const __m256i& y)
{
    return _mm256_sad_epu8(x, y);
}

SFINLINE __m256i blendv(const __m256i& x, const __m256i& y, const __m256i& m)
{
    return _mm256_blendv_epi8(x, y, m);
}
#endif


template <typename V>
SFINLINE V mul3(const V& x)
{
    return add_i16(x, add_i16(x, x));
}

template <typename V>
SFINLINE V absdiff_u8(const V& x, const V& y)
{
    return or_reg(subs(x, y), subs(y, x));
}

template <typename V>
SFINLINE V cmpgt_u8(const V& x, const V& y, const V& all)
{
    return xor_reg(cmpeq_i8(max_u8(x, y), y), all);
}

template <typename V>
SFINLINE V cmpgt_u16(const V& x, const V& y, const V& all)
{
    return xor_reg(cmpeq_i16(max_u16(x, y), y), all);
}

template <typename V>
SFINLINE V cmplt_u8(const V& x, const V& y, const V& all)
{
    return xor_reg(cmpeq_i8(max_u8(x, y), y), all);
}






#endif

