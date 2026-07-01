CXX      = g++

# Base flags. -march=native auto-enables whatever ISA the build host has
# (AVX-512 if present, else AVX2, etc). Do NOT force -mavx512f so the binary
# runs on hosts without AVX-512; bw_width.h auto-detects the right width.
CXXFLAGS = -O3 -march=native -std=c++20 -Wall -Wextra -pthread

# Optional: force the SIMD load width regardless of -march=native detection.
# Useful for cross-compiles / simulators / other target machines.
#   make WIDTH=512   ->  -DBW_SIMD_WIDTH=512   (also needs -mavx512f, see below)
#   make WIDTH=256   ->  -DBW_SIMD_WIDTH=256
#   make WIDTH=64    ->  -DBW_SIMD_WIDTH=64    (scalar; no SIMD ISA required)
# If WIDTH is unset, bw_width.h auto-detects from the build target's ISA.
#
# NOTE: forcing a width wider than -march=native provides requires also adding
# the matching arch flag via EXTRA_CXXFLAGS, e.g.:
#   make WIDTH=512 EXTRA_CXXFLAGS=-mavx512f
ifdef WIDTH
CXXFLAGS += -DBW_SIMD_WIDTH=$(WIDTH)
endif

# Escape hatch for extra arch/codegen flags (e.g. -mavx512f when forcing 512).
CXXFLAGS += $(EXTRA_CXXFLAGS)

TARGETS = randread_bw stream_bw

.PHONY: all clean config

all: $(TARGETS)

# config.py is a USER INPUT (chooses the default ADDR_MAP and sweep/hardware
# settings), not a build artifact — bootstrap it once with `make config`, edit,
# then build. `make config` overwrites any existing config.py from the template.
config:
	@cp config_template.py config.py
	@echo ">> config.py (re)created from config_template.py — local edits overwritten. Adjust settings, then 'make'."

# If config.py is missing, fail with guidance instead of silently scaffolding.
config.py:
	@echo "error: config.py not found. Run 'make config', adjust settings, then 'make'." >&2; exit 1

# access_masks.h is generated from config.ADDR_MAP (address_mapping.py) and
# consumed by randread_bw's ACCESS_MODE=1 (samebank) access pattern.
access_masks.h: gen_access_masks.py address_mapping.py config.py
	python3 gen_access_masks.py -o $@

randread_bw: randread_bw.cpp bw_width.h access_masks.h
	$(CXX) $(CXXFLAGS) -o $@ $<

stream_bw: stream_bw.cpp bw_width.h
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) access_masks.h
