PYTHON ?= python3
CXX ?= c++
CPU_FLAGS = -std=c++17 -O1 -g -Wall -Wextra -Werror -fsanitize=address,undefined -fno-omit-frame-pointer -ffp-contract=off -fno-fast-math

.PHONY: test test-cpu test-fp8 test-benchmark test-benchmark-profile test-benchmark-readout test-host-owner test-platform-collector test-benchmark-bindings test-codec test-cache-validator test-cache-gpu test-grouped-reader test-cache-native-inspection test-cache-phase-profile
test: test-cpu test-fp8 test-benchmark test-benchmark-profile test-benchmark-readout test-host-owner test-platform-collector test-benchmark-bindings test-codec test-cache-validator test-cache-gpu test-grouped-reader test-cache-native-inspection test-cache-phase-profile

test-cpu:
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -S tools/test_cpu.py

test-fp8:
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -S tools/test_fp8_baseline.py

test-benchmark:
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -S tests/test_model_benchmark.py

test-benchmark-profile:
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -S tests/test_model_benchmark_profile.py
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -S tests/test_profile_readout.py

test-benchmark-readout:
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -S tests/test_benchmark_readout.py

test-host-owner:
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -S tests/test_host_process_owner.py
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -S tests/test_benchmark_owner_adapter.py

test-platform-collector:
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -S tests/test_platform_collector.py

test-benchmark-bindings:
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -S tests/test_verify_benchmark_runtime.py
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -S tests/test_benchmark_artifacts.py
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -S tests/test_benchmark_host_transport.py
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -S tests/test_prepare_benchmark_package.py
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -S tests/test_benchmark_release_observer.py
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -S tests/test_benchmark_resource_gate.py
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -S tests/test_benchmark_physical_controller.py

test-codec:
	mkdir -p .test-build
	$(CXX) $(CPU_FLAGS) prototype/tests/codec_test.cpp -o .test-build/codec_test
	.test-build/codec_test
	$(CXX) $(CPU_FLAGS) prototype/tests/bf16_math_test.cpp -o .test-build/bf16_math_test
	.test-build/bf16_math_test

test-cache-validator:
	mkdir -p .test-build
	$(CXX) $(CPU_FLAGS) -DUQ_VALIDATION_EXTRACTOR prototype/tests/bf16_validation_test.cpp -o .test-build/extract_validator
	.test-build/extract_validator tests/fixtures/original-native/prototype/runtime/attention.hip prototype/runtime/attention.hip "$(CURDIR)/.test-build/validator"
	$(CXX) $(CPU_FLAGS) '-DUQ_VALIDATION_BASELINE_FILE="$(CURDIR)/.test-build/validator.baseline.inc"' '-DUQ_VALIDATION_CANDIDATE_FILE="$(CURDIR)/.test-build/validator.candidate.inc"' prototype/tests/bf16_validation_test.cpp -o .test-build/bf16_validation_test
	.test-build/bf16_validation_test prototype/runtime/attention.hip

# Offline composition and declaration-only HIP checks; this target never uses a GPU.
test-grouped-reader:
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 CXX="$(CXX)" $(PYTHON) -I -B -S tests/test_grouped_cache_reader.py

test-cache-native-inspection:
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -S tests/test_cache_native_inspection.py

test-cache-gpu:
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -S tests/test_cache_gpu_fixture.py
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -S tests/test_cache_gpu_owner.py
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -S tests/test_cache_benchmark_readout.py

# Host-clock projection and CPU-only descriptive readout; no native library load.
test-cache-phase-profile:
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -S tests/test_cache_phase_projection.py
	OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -I -B -S tests/test_cache_phase_readout.py
