#!/usr/bin/env python3
"""
BLAS Driver Generator for Morph Framework
Generates CPU drivers for all BLAS operations based on templates
"""

import os
from pathlib import Path

# BLAS operation definitions
BLAS_OPS = {
    "05_blas_swap": {
        "name": "SWAP",
        "description": "Vector Swap: swap(x, y)",
        "signature": "swap(n, x, inc_x, y, inc_y)",
        "return_type": "void",
        "validation_tolerance": "1e-15",
        "context_members": ["x", "y", "x_ref", "y_ref"],
        "reset_logic": """
    // Fill vectors with different random values
    fillRand(ctx->x, -10.0, 10.0);
    fillRand(ctx->y, 100.0, 200.0);
    
    // Broadcast for MPI consistency
    BCAST(ctx->x, DOUBLE);
    BCAST(ctx->y, DOUBLE);
    
    // Copy for reference computation
    ctx->x_ref = ctx->x;
    ctx->y_ref = ctx->y;""",
        "compute_call": "swap(ctx->n, ctx->x, ctx->inc_x, ctx->y, ctx->inc_y);",
        "best_call": "correctSwap(ctx->n, ctx->x_ref, ctx->inc_x, ctx->y_ref, ctx->inc_y);",
        "validation": """
    // For swap, both vectors should be exchanged
    std::vector<double> x_test = ctx->x;
    std::vector<double> y_test = ctx->y;
    correctSwap(ctx->n, x_test, ctx->inc_x, y_test, ctx->inc_y);
    
    bool x_valid = blas_utils::compareVectors(ctx->x, x_test, tolerance);
    bool y_valid = blas_utils::compareVectors(ctx->y, y_test, tolerance);
    bool isValid = x_valid && y_valid;"""
    },
    
    "06_blas_nrm2": {
        "name": "NRM2",
        "description": "Vector 2-Norm: result = ||x||_2",
        "signature": "nrm2(n, x, inc_x)",
        "return_type": "double",
        "validation_tolerance": "1e-12",
        "context_members": ["x", "result", "result_ref"],
        "reset_logic": """
    // Fill vector with random values
    fillRand(ctx->x, -1.0, 1.0);
    
    // Broadcast for MPI consistency
    BCAST(ctx->x, DOUBLE);
    
    // Reset results
    ctx->result = 0.0;
    ctx->result_ref = 0.0;""",
        "compute_call": "ctx->result = nrm2(ctx->n, ctx->x, ctx->inc_x);",
        "best_call": "ctx->result_ref = correctNrm2(ctx->n, ctx->x, ctx->inc_x);",
        "validation": """
    double reference = correctNrm2(ctx->n, ctx->x, ctx->inc_x);
    double error = std::abs(ctx->result - reference);
    if (std::abs(reference) > 1e-12) {
        error /= std::abs(reference);
    }
    bool isValid = error <= tolerance;"""
    },
    
    "07_blas_asum": {
        "name": "ASUM", 
        "description": "Vector Absolute Sum: result = sum(|x[i]|)",
        "signature": "asum(n, x, inc_x)",
        "return_type": "double",
        "validation_tolerance": "1e-12",
        "context_members": ["x", "result", "result_ref"],
        "reset_logic": """
    // Fill vector with random values
    fillRand(ctx->x, -10.0, 10.0);
    
    // Broadcast for MPI consistency
    BCAST(ctx->x, DOUBLE);
    
    // Reset results
    ctx->result = 0.0;
    ctx->result_ref = 0.0;""",
        "compute_call": "ctx->result = asum(ctx->n, ctx->x, ctx->inc_x);",
        "best_call": "ctx->result_ref = correctAsum(ctx->n, ctx->x, ctx->inc_x);",
        "validation": """
    double reference = correctAsum(ctx->n, ctx->x, ctx->inc_x);
    double error = std::abs(ctx->result - reference);
    if (std::abs(reference) > 1e-12) {
        error /= std::abs(reference);
    }
    bool isValid = error <= tolerance;"""
    },
    
    "08_blas_iamax": {
        "name": "IAMAX",
        "description": "Index of Maximum Absolute Value: result = argmax(|x[i]|)",
        "signature": "iamax(n, x, inc_x)",
        "return_type": "BLASLONG",
        "validation_tolerance": "0",  # Exact match for index
        "context_members": ["x", "result", "result_ref"],
        "reset_logic": """
    // Fill vector with random values, ensure some variation
    fillRand(ctx->x, -10.0, 10.0);
    
    // Broadcast for MPI consistency
    BCAST(ctx->x, DOUBLE);
    
    // Reset results
    ctx->result = 0;
    ctx->result_ref = 0;""",
        "compute_call": "ctx->result = iamax(ctx->n, ctx->x, ctx->inc_x);",
        "best_call": "ctx->result_ref = correctIamax(ctx->n, ctx->x, ctx->inc_x);",
        "validation": """
    BLASLONG reference = correctIamax(ctx->n, ctx->x, ctx->inc_x);
    bool isValid = (ctx->result == reference);"""
    },
    
    "09_blas_iamin": {
        "name": "IAMIN",
        "description": "Index of Minimum Absolute Value: result = argmin(|x[i]|)",
        "signature": "iamin(n, x, inc_x)",
        "return_type": "BLASLONG", 
        "validation_tolerance": "0",  # Exact match for index
        "context_members": ["x", "result", "result_ref"],
        "reset_logic": """
    // Fill vector with random values, ensure some variation
    fillRand(ctx->x, -10.0, 10.0);
    
    // Broadcast for MPI consistency
    BCAST(ctx->x, DOUBLE);
    
    // Reset results
    ctx->result = 0;
    ctx->result_ref = 0;""",
        "compute_call": "ctx->result = iamin(ctx->n, ctx->x, ctx->inc_x);",
        "best_call": "ctx->result_ref = correctIamin(ctx->n, ctx->x, ctx->inc_x);",
        "validation": """
    BLASLONG reference = correctIamin(ctx->n, ctx->x, ctx->inc_x);
    bool isValid = (ctx->result == reference);"""
    }
}

def generate_cpu_driver(op_id, op_info):
    """Generate CPU driver for a BLAS operation"""
    
    # Determine context structure based on members
    context_struct = "struct Context {\n"
    for member in op_info["context_members"]:
        if member in ["x", "y", "x_ref", "y_ref"]:
            context_struct += f"    std::vector<double> {member};\n"
        elif member in ["result", "result_ref"]:
            if op_info["return_type"] == "double":
                context_struct += f"    double {member};\n"
            else:
                context_struct += f"    BLASLONG {member};\n"
    
    context_struct += """    BLASLONG n;
    BLASLONG inc_x, inc_y;
};"""
    
    # Generate validation logic
    validation_block = op_info["validation"]
    if not validation_block.strip().endswith("bool isValid"):
        validation_block += """
    
    if (!isValid) {
        printf(\"""" + op_info["name"] + """ validation failed\\n");
    }"""
    
    driver_content = f'''// Driver for {op_id} for Serial, OpenMP, MPI, and MPI+OpenMP
// {op_info["description"]}
// Based on OpenBLAS implementation

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "utilities.hpp"
#include "baseline.hpp"
#include "generated-code.hpp"   // code generated by LLM

{context_struct}

void reset(Context *ctx) {{
{op_info["reset_logic"]}
}}

Context *init() {{
    Context *ctx = new Context();
    
    // Problem size
    ctx->n = DRIVER_PROBLEM_SIZE;
    
    // Stride parameters (unit stride for simplicity)
    ctx->inc_x = 1;
    ctx->inc_y = 1;
    
    // Allocate memory
    ctx->x.resize(ctx->n);'''
    
    # Add additional allocations based on context members
    for member in op_info["context_members"]:
        if member == "y":
            driver_content += "\n    ctx->y.resize(ctx->n);"
        elif member == "x_ref":
            driver_content += "\n    ctx->x_ref.resize(ctx->n);"
        elif member == "y_ref":
            driver_content += "\n    ctx->y_ref.resize(ctx->n);"
    
    driver_content += f'''
    
    reset(ctx);
    return ctx;
}}

void NO_OPTIMIZE compute(Context *ctx) {{
    // Call the LLM-generated function
    // Expected signature: {op_info["signature"]}
    {op_info["compute_call"]}
}}

void NO_OPTIMIZE best(Context *ctx) {{
    // Call the reference implementation
    {op_info["best_call"]}
}}

bool validate(Context *ctx) {{
    const double tolerance = {op_info["validation_tolerance"]};
    
{validation_block}
    
    return isValid;
}}

void destroy(Context *ctx) {{
    delete ctx;
}}'''
    
    return driver_content

def main():
    """Generate all BLAS CPU drivers"""
    base_dir = Path(__file__).parent
    
    for op_id, op_info in BLAS_OPS.items():
        # Create directory
        op_dir = base_dir / op_id
        op_dir.mkdir(exist_ok=True)
        
        # Generate CPU driver
        cpu_content = generate_cpu_driver(op_id, op_info)
        cpu_file = op_dir / "cpu.cc"
        
        with open(cpu_file, 'w') as f:
            f.write(cpu_content)
        
        print(f"Generated {cpu_file}")

if __name__ == "__main__":
    main()
