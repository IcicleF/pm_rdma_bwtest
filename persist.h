#ifndef _PERSIST_H_
#define _PERSIST_H_

#include <assert.h>
#include <immintrin.h>
#include <stdint.h>

#define force_inline __attribute__((always_inline)) inline
#define NORETURN __attribute__((noreturn))
#define barrier_memory() asm volatile("" ::: "memory")

typedef uint64_t ua_uint64_t __attribute__((aligned(1)));
typedef uint32_t ua_uint32_t __attribute__((aligned(1)));
typedef uint16_t ua_uint16_t __attribute__((aligned(1)));

#define CACHE_LINE_SIZE 64
#define FLUSH_ALIGN ((uintptr_t)CACHE_LINE_SIZE)
#define NVM_BLOCK_SIZE 256

static force_inline int util_is_pow2(uint64_t v) { return v && !(v & (v - 1)); }

static inline void mfence() { asm volatile("mfence" ::: "memory"); }

static inline void sfence() { _mm_sfence(); }

static inline void clflush(const void *data, int len) {
  volatile char *ptr = (char *)((unsigned long)data & ~(CACHE_LINE_SIZE - 1));
  for (; ptr < (char *)data + len; ptr += CACHE_LINE_SIZE) {
    asm volatile("clflush %0" : "+m"(*(volatile char *)ptr));
  }
  sfence();
}

static force_inline void pmem_clflushopt(const void *addr) {
  asm volatile(".byte 0x66; clflush %0" : "+m"(*(volatile char *)(addr)));
}
static force_inline void pmem_clwb(const void *addr) {
  asm volatile(".byte 0x66; xsaveopt %0" : "+m"(*(volatile char *)(addr)));
}

typedef void flush_fn(const void *, size_t);

static force_inline void flush_clflush_nolog(const void *addr, size_t len) {
  uintptr_t uptr;

  /*
   * Loop through cache-line-size (typically 64B) aligned chunks
   * covering the given range.
   */
  for (uptr = (uintptr_t)addr & ~(FLUSH_ALIGN - 1);
       uptr < (uintptr_t)addr + len; uptr += FLUSH_ALIGN)
    _mm_clflush((char *)uptr);
}

static force_inline void clflush_fence(const void *addr, size_t len) {
  uintptr_t uptr;
  for (uptr = (uintptr_t)addr & ~(FLUSH_ALIGN - 1);
       uptr < (uintptr_t)addr + len; uptr += FLUSH_ALIGN)
    _mm_clflush((char *)uptr);
  _mm_sfence();
}

static force_inline void clflushopt_fence(const void *addr, size_t len) {
  uintptr_t uptr;
  _mm_sfence();
  for (uptr = (uintptr_t)addr & ~(FLUSH_ALIGN - 1);
       uptr < (uintptr_t)addr + len; uptr += FLUSH_ALIGN)
    pmem_clflushopt((char *)uptr);
  _mm_sfence();
}

static force_inline void clwb_fence(const void *addr, size_t len) {
  uintptr_t uptr;
  for (uptr = (uintptr_t)addr & ~(FLUSH_ALIGN - 1);
       uptr < (uintptr_t)addr + len; uptr += FLUSH_ALIGN)
    pmem_clwb((char *)uptr);
  _mm_sfence();
}

/*
 * flush_clflushopt_nolog -- flush the CPU cache, using clflushopt
 */
static force_inline void flush_clflushopt_nolog(const void *addr, size_t len) {
  uintptr_t uptr;

  /*
   * Loop through cache-line-size (typically 64B) aligned chunks
   * covering the given range.
   */
  for (uptr = (uintptr_t)addr & ~(FLUSH_ALIGN - 1);
       uptr < (uintptr_t)addr + len; uptr += FLUSH_ALIGN) {
    pmem_clflushopt((char *)uptr);
  }
}

/*
 * flush_clwb_nolog -- flush the CPU cache, using clwb
 */
static force_inline void flush_clwb_nolog(const void *addr, size_t len) {
  uintptr_t uptr;

  /*
   * Loop through cache-line-size (typically 64B) aligned chunks
   * covering the given range.
   */
  for (uptr = (uintptr_t)addr & ~(FLUSH_ALIGN - 1);
       uptr < (uintptr_t)addr + len; uptr += FLUSH_ALIGN) {
    pmem_clwb((char *)uptr);
  }
}

static inline void noflush(const void *addr, size_t len) {
  /* NOP, not even pmemcheck annotation */
}

static force_inline void flush_empty_nolog(const void *addr, size_t len) {
  /* NOP, but tell pmemcheck about it */
}

typedef void barrier_fn(void);
static inline void barrier_after_ntstores(void) {
  /*
   * In this configuration pmem_drain does not contain sfence, so we have
   * to serialize non-temporal store instructions.
   */
  _mm_sfence();
}

static inline void no_barrier_after_ntstores(void) {
  /*
   * In this configuration pmem_drain contains sfence, so we don't have
   * to serialize non-temporal store instructions
   */
}

static force_inline __m512i mm512_loadu_si512(const char *src, unsigned idx) {
  return _mm512_loadu_si512((const __m512i *)src + idx);
}

static force_inline void mm512_stream_si512(char *dest, unsigned idx,
                                            __m512i src) {
  _mm512_stream_si512((__m512i *)dest + idx, src);
  barrier_memory();
}

static force_inline void memmove_movnt32x64b(char *dest, const char *src) {
  __m512i zmm0 = mm512_loadu_si512(src, 0);
  __m512i zmm1 = mm512_loadu_si512(src, 1);
  __m512i zmm2 = mm512_loadu_si512(src, 2);
  __m512i zmm3 = mm512_loadu_si512(src, 3);
  __m512i zmm4 = mm512_loadu_si512(src, 4);
  __m512i zmm5 = mm512_loadu_si512(src, 5);
  __m512i zmm6 = mm512_loadu_si512(src, 6);
  __m512i zmm7 = mm512_loadu_si512(src, 7);
  __m512i zmm8 = mm512_loadu_si512(src, 8);
  __m512i zmm9 = mm512_loadu_si512(src, 9);
  __m512i zmm10 = mm512_loadu_si512(src, 10);
  __m512i zmm11 = mm512_loadu_si512(src, 11);
  __m512i zmm12 = mm512_loadu_si512(src, 12);
  __m512i zmm13 = mm512_loadu_si512(src, 13);
  __m512i zmm14 = mm512_loadu_si512(src, 14);
  __m512i zmm15 = mm512_loadu_si512(src, 15);
  __m512i zmm16 = mm512_loadu_si512(src, 16);
  __m512i zmm17 = mm512_loadu_si512(src, 17);
  __m512i zmm18 = mm512_loadu_si512(src, 18);
  __m512i zmm19 = mm512_loadu_si512(src, 19);
  __m512i zmm20 = mm512_loadu_si512(src, 20);
  __m512i zmm21 = mm512_loadu_si512(src, 21);
  __m512i zmm22 = mm512_loadu_si512(src, 22);
  __m512i zmm23 = mm512_loadu_si512(src, 23);
  __m512i zmm24 = mm512_loadu_si512(src, 24);
  __m512i zmm25 = mm512_loadu_si512(src, 25);
  __m512i zmm26 = mm512_loadu_si512(src, 26);
  __m512i zmm27 = mm512_loadu_si512(src, 27);
  __m512i zmm28 = mm512_loadu_si512(src, 28);
  __m512i zmm29 = mm512_loadu_si512(src, 29);
  __m512i zmm30 = mm512_loadu_si512(src, 30);
  __m512i zmm31 = mm512_loadu_si512(src, 31);

  mm512_stream_si512(dest, 0, zmm0);
  mm512_stream_si512(dest, 1, zmm1);
  mm512_stream_si512(dest, 2, zmm2);
  mm512_stream_si512(dest, 3, zmm3);
  mm512_stream_si512(dest, 4, zmm4);
  mm512_stream_si512(dest, 5, zmm5);
  mm512_stream_si512(dest, 6, zmm6);
  mm512_stream_si512(dest, 7, zmm7);
  mm512_stream_si512(dest, 8, zmm8);
  mm512_stream_si512(dest, 9, zmm9);
  mm512_stream_si512(dest, 10, zmm10);
  mm512_stream_si512(dest, 11, zmm11);
  mm512_stream_si512(dest, 12, zmm12);
  mm512_stream_si512(dest, 13, zmm13);
  mm512_stream_si512(dest, 14, zmm14);
  mm512_stream_si512(dest, 15, zmm15);
  mm512_stream_si512(dest, 16, zmm16);
  mm512_stream_si512(dest, 17, zmm17);
  mm512_stream_si512(dest, 18, zmm18);
  mm512_stream_si512(dest, 19, zmm19);
  mm512_stream_si512(dest, 20, zmm20);
  mm512_stream_si512(dest, 21, zmm21);
  mm512_stream_si512(dest, 22, zmm22);
  mm512_stream_si512(dest, 23, zmm23);
  mm512_stream_si512(dest, 24, zmm24);
  mm512_stream_si512(dest, 25, zmm25);
  mm512_stream_si512(dest, 26, zmm26);
  mm512_stream_si512(dest, 27, zmm27);
  mm512_stream_si512(dest, 28, zmm28);
  mm512_stream_si512(dest, 29, zmm29);
  mm512_stream_si512(dest, 30, zmm30);
  mm512_stream_si512(dest, 31, zmm31);
}

static force_inline void memmove_movnt16x64b(char *dest, const char *src) {
  __m512i zmm0 = mm512_loadu_si512(src, 0);
  __m512i zmm1 = mm512_loadu_si512(src, 1);
  __m512i zmm2 = mm512_loadu_si512(src, 2);
  __m512i zmm3 = mm512_loadu_si512(src, 3);
  __m512i zmm4 = mm512_loadu_si512(src, 4);
  __m512i zmm5 = mm512_loadu_si512(src, 5);
  __m512i zmm6 = mm512_loadu_si512(src, 6);
  __m512i zmm7 = mm512_loadu_si512(src, 7);
  __m512i zmm8 = mm512_loadu_si512(src, 8);
  __m512i zmm9 = mm512_loadu_si512(src, 9);
  __m512i zmm10 = mm512_loadu_si512(src, 10);
  __m512i zmm11 = mm512_loadu_si512(src, 11);
  __m512i zmm12 = mm512_loadu_si512(src, 12);
  __m512i zmm13 = mm512_loadu_si512(src, 13);
  __m512i zmm14 = mm512_loadu_si512(src, 14);
  __m512i zmm15 = mm512_loadu_si512(src, 15);

  mm512_stream_si512(dest, 0, zmm0);
  mm512_stream_si512(dest, 1, zmm1);
  mm512_stream_si512(dest, 2, zmm2);
  mm512_stream_si512(dest, 3, zmm3);
  mm512_stream_si512(dest, 4, zmm4);
  mm512_stream_si512(dest, 5, zmm5);
  mm512_stream_si512(dest, 6, zmm6);
  mm512_stream_si512(dest, 7, zmm7);
  mm512_stream_si512(dest, 8, zmm8);
  mm512_stream_si512(dest, 9, zmm9);
  mm512_stream_si512(dest, 10, zmm10);
  mm512_stream_si512(dest, 11, zmm11);
  mm512_stream_si512(dest, 12, zmm12);
  mm512_stream_si512(dest, 13, zmm13);
  mm512_stream_si512(dest, 14, zmm14);
  mm512_stream_si512(dest, 15, zmm15);
}

static force_inline void memmove_movnt8x64b(char *dest, const char *src) {
  __m512i zmm0 = mm512_loadu_si512(src, 0);
  __m512i zmm1 = mm512_loadu_si512(src, 1);
  __m512i zmm2 = mm512_loadu_si512(src, 2);
  __m512i zmm3 = mm512_loadu_si512(src, 3);
  __m512i zmm4 = mm512_loadu_si512(src, 4);
  __m512i zmm5 = mm512_loadu_si512(src, 5);
  __m512i zmm6 = mm512_loadu_si512(src, 6);
  __m512i zmm7 = mm512_loadu_si512(src, 7);

  mm512_stream_si512(dest, 0, zmm0);
  mm512_stream_si512(dest, 1, zmm1);
  mm512_stream_si512(dest, 2, zmm2);
  mm512_stream_si512(dest, 3, zmm3);
  mm512_stream_si512(dest, 4, zmm4);
  mm512_stream_si512(dest, 5, zmm5);
  mm512_stream_si512(dest, 6, zmm6);
  mm512_stream_si512(dest, 7, zmm7);
}

static force_inline void memmove_movnt4x64b(char *dest, const char *src) {
  __m512i zmm0 = mm512_loadu_si512(src, 0);
  __m512i zmm1 = mm512_loadu_si512(src, 1);
  __m512i zmm2 = mm512_loadu_si512(src, 2);
  __m512i zmm3 = mm512_loadu_si512(src, 3);

  mm512_stream_si512(dest, 0, zmm0);
  mm512_stream_si512(dest, 1, zmm1);
  mm512_stream_si512(dest, 2, zmm2);
  mm512_stream_si512(dest, 3, zmm3);
}

static force_inline void memmove_movnt2x64b(char *dest, const char *src) {
  __m512i zmm0 = mm512_loadu_si512(src, 0);
  __m512i zmm1 = mm512_loadu_si512(src, 1);

  mm512_stream_si512(dest, 0, zmm0);
  mm512_stream_si512(dest, 1, zmm1);
}

static force_inline void memmove_movnt1x64b(char *dest, const char *src) {
  __m512i zmm0 = mm512_loadu_si512(src, 0);

  mm512_stream_si512(dest, 0, zmm0);
}

static force_inline void memmove_movnt1x32b(char *dest, const char *src) {
  __m256i zmm0 = _mm256_loadu_si256((__m256i *)src);

  _mm256_stream_si256((__m256i *)dest, zmm0);
}

static force_inline void memmove_movnt1x16b(char *dest, const char *src) {
  __m128i ymm0 = _mm_loadu_si128((__m128i *)src);

  _mm_stream_si128((__m128i *)dest, ymm0);
}

static force_inline void memmove_movnt1x8b(char *dest, const char *src) {
  _mm_stream_si64((long long *)dest, *(long long *)src);
}

static force_inline void memmove_movnt1x4b(char *dest, const char *src) {
  _mm_stream_si32((int *)dest, *(int *)src);
}

static force_inline void memmove_small_avx_noflush(char *dest, const char *src,
                                                   size_t len) {
  assert(len <= 64);

  if (len <= 8)
    goto le8;
  if (len <= 32)
    goto le32;

  {
    /* 33..64 */
    __m256i ymm0 = _mm256_loadu_si256((__m256i *)src);
    __m256i ymm1 = _mm256_loadu_si256((__m256i *)(src + len - 32));

    _mm256_storeu_si256((__m256i *)dest, ymm0);
    _mm256_storeu_si256((__m256i *)(dest + len - 32), ymm1);
    return;
  }

le32 : {
  if (len > 16) {
    /* 17..32 */
    __m128i xmm0 = _mm_loadu_si128((__m128i *)src);
    __m128i xmm1 = _mm_loadu_si128((__m128i *)(src + len - 16));

    _mm_storeu_si128((__m128i *)dest, xmm0);
    _mm_storeu_si128((__m128i *)(dest + len - 16), xmm1);
    return;
  }

  /* 9..16 */
  ua_uint64_t d80 = *(ua_uint64_t *)src;
  ua_uint64_t d81 = *(ua_uint64_t *)(src + len - 8);

  *(ua_uint64_t *)dest = d80;
  *(ua_uint64_t *)(dest + len - 8) = d81;
  return;
}

le8:
  if (len <= 2)
    goto le2;

  {
    if (len > 4) {
      /* 5..8 */
      ua_uint32_t d40 = *(ua_uint32_t *)src;
      ua_uint32_t d41 = *(ua_uint32_t *)(src + len - 4);

      *(ua_uint32_t *)dest = d40;
      *(ua_uint32_t *)(dest + len - 4) = d41;
      return;
    }

    /* 3..4 */
    ua_uint16_t d20 = *(ua_uint16_t *)src;
    ua_uint16_t d21 = *(ua_uint16_t *)(src + len - 2);

    *(ua_uint16_t *)dest = d20;
    *(ua_uint16_t *)(dest + len - 2) = d21;
    return;
  }

le2:
  if (len == 2) {
    *(ua_uint16_t *)dest = *(ua_uint16_t *)src;
    return;
  }

  *(uint8_t *)dest = *(uint8_t *)src;
}

static force_inline void memmove_small_avx(char *dest, const char *src,
                                           size_t len, flush_fn flush) {
  /*
   * pmemcheck complains about "overwritten stores before they were made
   * persistent" for overlapping stores (last instruction in each code
   * path) in the optimized version.
   * libc's memcpy also does that, so we can't use it here.
   */
  memmove_small_avx_noflush(dest, src, len);
  flush(dest, len);
}

static force_inline void memmove_small_avx512f(char *dest, const char *src,
                                               size_t len, flush_fn flush) {
  /* We can't do better than AVX here. */
  memmove_small_avx(dest, src, len, flush);
}

static force_inline void avx_zeroupper(void) { _mm256_zeroupper(); }

static force_inline void memmove_movnt_avx512f_fw(char *dest, const char *src,
                                                  size_t len, flush_fn flush) {
  size_t cnt = (uint64_t)dest & 63;
  if (cnt > 0) {
    cnt = 64 - cnt;

    if (cnt > len)
      cnt = len;

    memmove_small_avx512f(dest, src, cnt, flush);

    dest += cnt;
    src += cnt;
    len -= cnt;
  }

  while (len >= 32 * 64) {
    memmove_movnt32x64b(dest, src);
    dest += 32 * 64;
    src += 32 * 64;
    len -= 32 * 64;
  }

  if (len >= 16 * 64) {
    memmove_movnt16x64b(dest, src);
    dest += 16 * 64;
    src += 16 * 64;
    len -= 16 * 64;
  }

  if (len >= 8 * 64) {
    memmove_movnt8x64b(dest, src);
    dest += 8 * 64;
    src += 8 * 64;
    len -= 8 * 64;
  }

  if (len >= 4 * 64) {
    memmove_movnt4x64b(dest, src);
    dest += 4 * 64;
    src += 4 * 64;
    len -= 4 * 64;
  }

  if (len >= 2 * 64) {
    memmove_movnt2x64b(dest, src);
    dest += 2 * 64;
    src += 2 * 64;
    len -= 2 * 64;
  }

  if (len >= 1 * 64) {
    memmove_movnt1x64b(dest, src);

    dest += 1 * 64;
    src += 1 * 64;
    len -= 1 * 64;
  }

  if (len == 0)
    goto end;

  /* There's no point in using more than 1 nt store for 1 cache line. */
  if (util_is_pow2(len)) {
    if (len == 32)
      memmove_movnt1x32b(dest, src);
    else if (len == 16)
      memmove_movnt1x16b(dest, src);
    else if (len == 8)
      memmove_movnt1x8b(dest, src);
    else if (len == 4)
      memmove_movnt1x4b(dest, src);
    else
      goto nonnt;

    goto end;
  }

nonnt:
  memmove_small_avx512f(dest, src, len, flush);
end:
  avx_zeroupper();
}

static force_inline void memmove_movnt_avx512f_bw(char *dest, const char *src,
                                                  size_t len, flush_fn flush) {
  dest += len;
  src += len;

  size_t cnt = (uint64_t)dest & 63;
  if (cnt > 0) {
    if (cnt > len)
      cnt = len;

    dest -= cnt;
    src -= cnt;
    len -= cnt;

    memmove_small_avx512f(dest, src, cnt, flush);
  }

  while (len >= 32 * 64) {
    dest -= 32 * 64;
    src -= 32 * 64;
    len -= 32 * 64;
    memmove_movnt32x64b(dest, src);
  }

  if (len >= 16 * 64) {
    dest -= 16 * 64;
    src -= 16 * 64;
    len -= 16 * 64;
    memmove_movnt16x64b(dest, src);
  }

  if (len >= 8 * 64) {
    dest -= 8 * 64;
    src -= 8 * 64;
    len -= 8 * 64;
    memmove_movnt8x64b(dest, src);
  }

  if (len >= 4 * 64) {
    dest -= 4 * 64;
    src -= 4 * 64;
    len -= 4 * 64;
    memmove_movnt4x64b(dest, src);
  }

  if (len >= 2 * 64) {
    dest -= 2 * 64;
    src -= 2 * 64;
    len -= 2 * 64;
    memmove_movnt2x64b(dest, src);
  }

  if (len >= 1 * 64) {
    dest -= 1 * 64;
    src -= 1 * 64;
    len -= 1 * 64;

    memmove_movnt1x64b(dest, src);
  }

  if (len == 0)
    goto end;

  /* There's no point in using more than 1 nt store for 1 cache line. */
  if (util_is_pow2(len)) {
    if (len == 32) {
      dest -= 32;
      src -= 32;
      memmove_movnt1x32b(dest, src);
    } else if (len == 16) {
      dest -= 16;
      src -= 16;
      memmove_movnt1x16b(dest, src);
    } else if (len == 8) {
      dest -= 8;
      src -= 8;
      memmove_movnt1x8b(dest, src);
    } else if (len == 4) {
      dest -= 4;
      src -= 4;
      memmove_movnt1x4b(dest, src);
    } else {
      goto nonnt;
    }

    goto end;
  }

nonnt:
  dest -= len;
  src -= len;

  memmove_small_avx512f(dest, src, len, flush);
end:
  avx_zeroupper();
}

static force_inline void memmove_movnt_avx512f(char *dest, const char *src,
                                               size_t len, flush_fn flush,
                                               barrier_fn barrier) {
  if ((uintptr_t)dest - (uintptr_t)src >= len)
    memmove_movnt_avx512f_fw(dest, src, len, flush);
  else
    memmove_movnt_avx512f_bw(dest, src, len, flush);

  barrier();
}

static force_inline void
memmove_movnt_avx512f_noflush(char *dest, const char *src, size_t len) {
  memmove_movnt_avx512f(dest, src, len, noflush, barrier_after_ntstores);
}

static force_inline void
memmove_movnt_avx512f_empty(char *dest, const char *src, size_t len) {
  memmove_movnt_avx512f(dest, src, len, flush_empty_nolog,
                        barrier_after_ntstores);
}

static force_inline void
memmove_movnt_avx512f_clflush(char *dest, const char *src, size_t len) {
  memmove_movnt_avx512f(dest, src, len, flush_clflush_nolog,
                        barrier_after_ntstores);
}

static force_inline void
memmove_movnt_avx512f_clflushopt(char *dest, const char *src, size_t len) {
  memmove_movnt_avx512f(dest, src, len, flush_clflushopt_nolog,
                        barrier_after_ntstores);
}

static force_inline void memmove_movnt_avx512f_clwb(char *dest, const char *src,
                                                    size_t len) {
  memmove_movnt_avx512f(dest, src, len, flush_clwb_nolog,
                        barrier_after_ntstores);
}

#endif // _PERSIST_H_