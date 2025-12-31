"""
Unit tests for COSF mathematical framework

Run with: pytest tests/
"""

import pytest
import numpy as np
import math

def test_golden_ratio():
    """Test golden ratio value"""
    phi = (1 + math.sqrt(5)) / 2
    assert abs(phi - 1.618033988749) < 1e-10
    
def test_phi_17():
    """Test φ computation"""
    phi = (1 + math.sqrt(5)) / 2
    phi_17 = phi ** 17
    assert abs(phi_17 - 5472.999288) < 0.001
    
def test_e_86():
    """Test e^8.6 computation"""
    e_86 = math.exp(8.6)
    assert abs(e_86 - 5434.644064) < 0.001
    
def test_convergence():
    """Test φ/e^8.6 convergence"""
    phi = (1 + math.sqrt(5)) / 2
    ratio = (phi ** 17) / math.exp(8.6)
    deviation = abs(ratio - 1)
    
    # Should be within 0.71%
    assert deviation < 0.0071
    
def test_cosf_value():
    """Test COSF = C/C"""
    C1 = 7.83
    C2 = 42800
    cosf = C2 / C1
    
    assert abs(cosf - 5466) < 1.0
    
def test_fcc_packing_density():
    """Test FCC packing density"""
    eta = math.pi / (3 * math.sqrt(2))
    assert abs(eta - 0.74048) < 0.00001
    
def test_torus_volume():
    """Test toroidal volume formula"""
    R, r = 10, 2
    V = 2 * math.pi**2 * R * r**2
    expected = 2 * math.pi**2 * 10 * 4
    assert abs(V - expected) < 0.001

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
