from mpmath import mp, sqrt, exp
mp.dps = 50

phi = (1 + sqrt(5)) / 2
phi_17 = phi ** 17

print("Checking n=17 across different m values:")
print("=" * 60)

for m_tenths in range(80, 90):
    m = m_tenths / 10.0
    e_m = exp(m)
    ratio = phi_17 / e_m
    deviation = abs(ratio - 1)
    
    print(f"n=17, m={m:.1f}: ratio={float(ratio):.6f}, dev={float(deviation)*100:.4f}%")

print("=" * 60)
print(f"\nφ¹⁷ = {float(phi_17):.2f}")
print(f"e^8.6 = {float(exp(8.6)):.2f}")
print(f"COSF = 42800/7.83  {42800/7.83:.2f}")
