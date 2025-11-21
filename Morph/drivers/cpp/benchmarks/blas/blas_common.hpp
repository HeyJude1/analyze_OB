#pragma once
#include <vector>
#include <cmath>
#include <algorithm>

// BLAS type definitions
typedef long BLASLONG;

// Common BLAS utilities
namespace blas_utils {
    
    // Fill vector with random values
    template<typename T>
    void fillRand(std::vector<T>& vec, T min_val, T max_val) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(min_val, max_val);
        
        for (auto& val : vec) {
            val = dis(gen);
        }
    }
    
    // Fill matrix with random values
    template<typename T>
    void fillRandMatrix(std::vector<T>& mat, size_t rows, size_t cols, T min_val, T max_val) {
        fillRand(mat, min_val, max_val);
    }
    
    // Compare vectors with tolerance
    template<typename T>
    bool compareVectors(const std::vector<T>& a, const std::vector<T>& b, T tolerance = 1e-6) {
        if (a.size() != b.size()) return false;
        
        for (size_t i = 0; i < a.size(); i++) {
            if (std::abs(a[i] - b[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }
    
    // Compare matrices with tolerance
    template<typename T>
    bool compareMatrices(const std::vector<T>& a, const std::vector<T>& b, 
                        size_t rows, size_t cols, T tolerance = 1e-6) {
        return compareVectors(a, b, tolerance);
    }
    
    // Compute relative error
    template<typename T>
    T relativeError(const std::vector<T>& computed, const std::vector<T>& reference) {
        if (computed.size() != reference.size()) return std::numeric_limits<T>::max();
        
        T max_error = 0.0;
        for (size_t i = 0; i < computed.size(); i++) {
            T error = std::abs(computed[i] - reference[i]);
            if (std::abs(reference[i]) > 1e-12) {
                error /= std::abs(reference[i]);
            }
            max_error = std::max(max_error, error);
        }
        return max_error;
    }
}
