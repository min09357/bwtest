// bw_width.h
// Compile-time selection of the SIMD load width for the bandwidth benchmarks.
// No function abstraction — each .cpp branches its own hot loop with #if on
// BW_SIMD_WIDTH. This header only resolves the width and provides a label.
//
// Precedence:
//   1. User override:  -DBW_SIMD_WIDTH=512|256|64   (e.g. make WIDTH=256)
//   2. Auto-detect from the build target's enabled ISA:
//        __AVX512F__ -> 512,  __AVX2__ -> 256,  otherwise -> 64 (scalar)
//
// Forcing a width wider than the build target supports also needs the matching
// arch flag (e.g. make WIDTH=512 EXTRA_CXXFLAGS=-mavx512f).

#ifndef BW_WIDTH_H
#define BW_WIDTH_H

#ifndef BW_SIMD_WIDTH
#  if defined(__AVX512F__)
#    define BW_SIMD_WIDTH 512
#  elif defined(__AVX2__)
#    define BW_SIMD_WIDTH 256
#  else
#    define BW_SIMD_WIDTH 64
#  endif
#endif

#if   BW_SIMD_WIDTH == 512
#  if !defined(__AVX512F__)
#    error "BW_SIMD_WIDTH=512 needs AVX-512F (pass -mavx512f or an arch with it)"
#  endif
#  define BW_SIMD_WIDTH_STR "512 (AVX-512)"
#elif BW_SIMD_WIDTH == 256
#  if !defined(__AVX2__)
#    error "BW_SIMD_WIDTH=256 needs AVX2 (pass -mavx2 or an arch with it)"
#  endif
#  define BW_SIMD_WIDTH_STR "256 (AVX2)"
#elif BW_SIMD_WIDTH == 64
#  define BW_SIMD_WIDTH_STR "64 (scalar)"
#else
#  error "BW_SIMD_WIDTH must be 512, 256, or 64"
#endif

// Print the resolved width at compile time (visible in the make log).
#pragma message("BW_SIMD_WIDTH = " BW_SIMD_WIDTH_STR)

#endif // BW_WIDTH_H
