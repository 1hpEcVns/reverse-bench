.PHONY: all bench bench-rs bench-all bench-run bench-run-rs plot asm-check

CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++23
RUSTC ?= rustc

all: bench

bench: bench.cpp
	$(CXX) $(CXXFLAGS) bench.cpp -o bench

bench-rs: bench.rs
	$(RUSTC) --edition=2024 -O -C target-cpu=native bench.rs -o bench_rs

bench-run: bench
	taskset -c 0 ./bench > results.csv

bench-run-rs: bench-rs
	taskset -c 0 ./bench_rs > results_rs.csv

bench-all: bench-run bench-run-rs

plot: bench-all
	nix develop --command python3 plot.py

asm-check:
	bash asm_check.sh
