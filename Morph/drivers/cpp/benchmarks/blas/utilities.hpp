#pragma once

/* Utilities for BLAS benchmarks in Morph framework
 * This file provides common utilities needed by all BLAS drivers
 */

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>

// MPI utilities (if MPI is available)
#ifdef USE_MPI
#include <mpi.h>
#define BCAST(vec, type) do { \
    if (type == DOUBLE) { \
        MPI_Bcast(vec.data(), vec.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD); \
    } \
} while(0)
#else
#define BCAST(vec, type) do { /* no-op */ } while(0)
#endif

// Type definitions for broadcast
#define DOUBLE 1

// Problem size macro (can be overridden at compile time)
#ifndef DRIVER_PROBLEM_SIZE
#define DRIVER_PROBLEM_SIZE (1 << 10)  // Default: 1024
#endif

// Optimization prevention macro
#define NO_OPTIMIZE __attribute__((noinline))

// Random number generation utilities
namespace blas_utils {
    
    // Thread-safe random number generator
    class RandomGenerator {
    private:
        static thread_local std::mt19937 gen;
        static thread_local bool initialized;
        
    public:
        static std::mt19937& get() {
            if (!initialized) {
                std::random_device rd;
                gen.seed(rd());
                initialized = true;
            }
            return gen;
        }
    };
    
    // Fill vector with random values in range [min_val, max_val]
    template<typename T>
    void fillRand(std::vector<T>& vec, T min_val, T max_val) {
        std::uniform_real_distribution<T> dis(min_val, max_val);
        auto& rng = RandomGenerator::get();
        
        for (auto& val : vec) {
            val = dis(rng);
        }
    }
    
    // Fill vector with specific pattern for testing
    template<typename T>
    void fillPattern(std::vector<T>& vec, T start_val = 1.0, T increment = 1.0) {
        T val = start_val;
        for (auto& elem : vec) {
            elem = val;
            val += increment;
        }
    }
    
    // Compare vectors with tolerance
    template<typename T>
    bool compareVectors(const std::vector<T>& a, const std::vector<T>& b, T tolerance = 1e-12) {
        if (a.size() != b.size()) return false;
        
        for (size_t i = 0; i < a.size(); i++) {
            if (std::abs(a[i] - b[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }
    
    // Compute relative error between vectors
    template<typename T>
    T relativeError(const std::vector<T>& computed, const std::vector<T>& reference) {
        if (computed.size() != reference.size()) {
            return std::numeric_limits<T>::max();
        }
        
        T max_error = 0.0;
        for (size_t i = 0; i < computed.size(); i++) {
            T error = std::abs(computed[i] - reference[i]);
            if (std::abs(reference[i]) > 1e-15) {
                error /= std::abs(reference[i]);
            }
            max_error = std::max(max_error, error);
        }
        return max_error;
    }
    
    // Compute matrix Frobenius norm
    template<typename T>
    T frobeniusNorm(const std::vector<T>& matrix, size_t rows, size_t cols) {
        T sum = 0.0;
        for (const auto& val : matrix) {
            sum += val * val;
        }
        return std::sqrt(sum);
    }
    
    // Print vector (for debugging)
    template<typename T>
    void printVector(const std::vector<T>& vec, const std::string& name, size_t max_elements = 10) {
        printf("%s: [", name.c_str());
        size_t n = std::min(vec.size(), max_elements);
        for (size_t i = 0; i < n; i++) {
            printf("%.6e", static_cast<double>(vec[i]));
            if (i < n - 1) printf(", ");
        }
        if (vec.size() > max_elements) {
            printf(", ... (%zu more)", vec.size() - max_elements);
        }
        printf("]\n");
    }
    
    // Performance measurement utilities
    class Timer {
    private:
        std::chrono::high_resolution_clock::time_point start_time;
        
    public:
        void start() {
            start_time = std::chrono::high_resolution_clock::now();
        }
        
        double elapsed() const {
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
            return duration.count() / 1000000.0;  // Return seconds
        }
    };
}

// Thread-local storage definitions
thread_local std::mt19937 blas_utils::RandomGenerator::gen;
thread_local bool blas_utils::RandomGenerator::initialized = false;
