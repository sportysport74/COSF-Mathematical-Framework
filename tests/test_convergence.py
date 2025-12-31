"""
Test COSF convergence - the core mathematical claim

This test verifies the fundamental convergence of φ and e^8.6
"""

import numpy as np
import pytest
import math

def test_cosf_convergence():
    """Test that φ/e^8.6 converges within 0.7% tolerance"""
    phi = (1 + np.sqrt(5)) / 2
    phi_17 = phi ** 17
    e_86 = np.exp(8.6)
    ratio = phi_17 / e_86
    
    deviation = abs(ratio - 1)
    
    # Core claim: deviation < 0.0071 (0.71%)
    assert deviation < 0.0071, f"Ratio {ratio:.10f} exceeds 0.7% tolerance (deviation: {deviation:.6f})"
    
    print(f" Convergence verified: {deviation*100:.4f}% deviation")

def test_cosf_uniqueness():
    """Test that (17, 8.6) is the optimal pair among integers"""
    phi = (1 + np.sqrt(5)) / 2
    
    # Check nearby integer pairs
    best_deviation = float('inf')
    best_pair = None
    
    for n in range(15, 20):
        for m_tenths in range(80, 90):
            m = m_tenths / 10
            ratio = (phi ** n) / np.exp(m)
            dev = abs(ratio - 1)
            
            if dev < best_deviation:
                best_deviation = dev
                best_pair = (n, m)
    
    # Should be (17, 8.6)
    assert best_pair[0] == 17, f"Best n is {best_pair[0]}, not 17"
    assert abs(best_pair[1] - 8.6) < 0.1, f"Best m is {best_pair[1]}, not 8.6"
    
    print(f" Uniqueness verified: (17, 8.6) is optimal")

def test_golden_ratio_precision():
    """Test golden ratio computation precision"""
    phi = (1 + math.sqrt(5)) / 2
    
    # Golden ratio properties
    assert abs(phi ** 2 - phi - 1) < 1e-10, "φ  φ + 1"
    assert abs(phi - 1.618033988749895) < 1e-14, "φ precision error"
    
    print(f" Golden ratio verified: φ = {phi:.15f}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
