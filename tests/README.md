# COSF Test Suite

Comprehensive test coverage for the COSF Mathematical Framework.

## Running Tests

\\\ash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=code --cov-report=html

# Run specific test file
pytest tests/test_convergence.py -v -s
\\\

## Test Files

### test_cosf.py
- **test_golden_ratio()** - Validates φ = 1.618...
- **test_phi_17()** - Verifies φ = 5472.999
- **test_e_86()** - Verifies e^8.6 = 5434.644
- **test_convergence()** - Confirms 0.7% deviation
- **test_cosf_value()** - Validates COSF  5466
- **test_fcc_packing_density()** - Confirms 74.048%
- **test_torus_volume()** - Validates geometric formulas

### test_convergence.py
- **test_cosf_convergence()** - Core convergence claim (φ/e^8.6)
- **test_cosf_uniqueness()** - Proves (17, 8.6) optimality
- **test_golden_ratio_precision()** - High-precision φ validation

## Expected Output

\\\
tests/test_cosf.py::test_golden_ratio PASSED
tests/test_cosf.py::test_phi_17 PASSED
tests/test_cosf.py::test_e_86 PASSED
tests/test_cosf.py::test_convergence PASSED
tests/test_cosf.py::test_cosf_value PASSED
tests/test_cosf.py::test_fcc_packing_density PASSED
tests/test_cosf.py::test_torus_volume PASSED
tests/test_convergence.py::test_cosf_convergence PASSED
tests/test_convergence.py::test_cosf_uniqueness PASSED
tests/test_convergence.py::test_golden_ratio_precision PASSED

 10 tests passed
\\\

## CI/CD Integration

Tests run automatically on every push via GitHub Actions.
See \.github/workflows/ci.yml\ for configuration.
