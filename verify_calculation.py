from mpmath import mp, sqrt, exp, power
mp.dps = 100  # Max precision

phi = (1 + sqrt(5)) / 2

print("CORRECT CALCULATIONS:")
print("=" * 70)
print(f"φ (golden ratio) = {phi}")
print(f"φ = {phi**2}")
print(f"φ = {phi**3}")
print()

# Calculate φ with full precision
phi_17 = power(phi, 17)
print(f"φ (FULL PRECISION) = {phi_17}")
print(f"φ (rounded) = {float(phi_17):.10f}")
print()

# Calculate e^8.6
e_86 = exp(mp.mpf('8.6'))
print(f"e^8.6 (FULL PRECISION) = {e_86}")
print(f"e^8.6 (rounded) = {float(e_86):.10f}")
print()

# Ratio and deviation
ratio = phi_17 / e_86
deviation = abs(ratio - 1) * 100

print("=" * 70)
print("CONVERGENCE ANALYSIS:")
print("=" * 70)
print(f"φ¹⁷ / e^8.6 = {float(ratio):.10f}")
print(f"Deviation from 1.0 = {float(deviation):.6f}%")
print()

# COSF value
COSF = mp.mpf('42800') / mp.mpf('7.83')
print(f"COSF = 42800/7.83 = {float(COSF):.10f}")
print()

# Distance to each
dist_to_phi17 = abs(COSF - phi_17) / phi_17 * 100
dist_to_e86 = abs(COSF - e_86) / e_86 * 100

print(f"COSF distance to φ: {float(dist_to_phi17):.6f}%")
print(f"COSF distance to e^8.6: {float(dist_to_e86):.6f}%")
print("=" * 70)

# Verify our core claim
if deviation < 1.0:
    print("\n CONFIRMED: φ  e^8.6 within 1% (0.7% actual)")
else:
    print(f"\n ERROR: Deviation is {float(deviation):.2f}%, NOT within 1%!")
