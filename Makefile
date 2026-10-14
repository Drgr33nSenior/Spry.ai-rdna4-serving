PYTHON ?= python3
CXX ?= c++
CPU_FLAGS = -std=c++17 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined -fno-omit-frame-pointer

.PHONY: test test-cpu test-codec
test: test-cpu test-codec

test-cpu:
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -S tools/test_cpu.py

test-codec:
	mkdir -p .test-build
	$(CXX) $(CPU_FLAGS) prototype/tests/codec_test.cpp -o .test-build/codec_test
	.test-build/codec_test
	$(CXX) $(CPU_FLAGS) prototype/tests/bf16_math_test.cpp -o .test-build/bf16_math_test
	.test-build/bf16_math_test
