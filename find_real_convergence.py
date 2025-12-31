from mpmath import mp, sqrt, exp, power, log
mp.dps = 100

phi = (1 + sqrt(5)) / 2

print("SEARCHING FOR ACTUAL CONVERGENCE...")
print("=" * 80)

# Check what n gives us ~5473
print("\nFinding which n gives φⁿ ≈ 5000-6000:")
print("-" * 80)

for n in range(1, 30):
    phi_n = power(phi, n)
    
    if 3000 < float(phi_n) < 10000:
        # Find best m for this n
        m_target = float(log(phi_n))
        e_m = exp(m_target)
        
        ratio = phi_n / e_m
        deviation = abs(ratio - 1) * 100
        
        print(f"n={n:2d}: φⁿ={float(phi_n):10.2f}, best m={float(m_target):.2f}, "
              f"ratio={float(ratio):.6f}, dev={float(deviation):.4f}%")

print("\n" + "=" * 80)
print("CHECKING CONVERGENCES WHERE φⁿ/eᵐ  1 (searching m around n/2):")
print("=" * 80)

convergences = []

for n in range(5, 25):
    phi_n = power(phi, n)
    
    # Try m values around n/2
    for m_tenths in range(int(n*4), int(n*6)):
        m = m_tenths / 10.0
        e_m = exp(m)
        ratio = phi_n / e_m
        deviation = abs(ratio - 1) * 100
        
        if deviation < 1.0:
            convergences.append({
                'n': n,
                'm': m,
                'phi_n': float(phi_n),
                'e_m': float(e_m),
                'ratio': float(ratio),
                'deviation': float(deviation)
            })

# Sort by deviation
convergences.sort(key=lambda x: x['deviation'])

print("\nFOUND CONVERGENCES (φⁿ/eᵐ within 1%):")
print("-" * 80)
for c in convergences[:10]:
    print(f"n={c['n']:2d}, m={c['m']:5.1f}: φⁿ={c['phi_n']:12.2f}, "
          f"eᵐ={c['e_m']:12.2f}, dev={c['deviation']:.4f}%")

print("\n" + "=" * 80)
print("CHECKING OUR ORIGINAL CLAIM (n=17, m=8.6):")
print("=" * 80)

phi_17 = power(phi, 17)
e_86 = exp(mp.mpf('8.6'))

print(f"φ = {float(phi_17):.10f}")
print(f"e^8.6 = {float(e_86):.10f}")
print(f"φ/e^8.6 = {float(phi_17/e_86):.10f}")
print(f"Deviation = {float(abs(phi_17/e_86 - 1)*100):.6f}%")

print("\n" + "=" * 80)
print("CRITICAL INSIGHT:")
print("=" * 80)

# The issue: we need φⁿ/eᵐ  1, not φⁿ  eᵐ!
# So if φ  3571 and we want ratio  1, we need e^m  3571
# That means m  ln(3571)  8.18

correct_m = float(log(phi_17))
e_correct = exp(correct_m)
correct_ratio = phi_17 / e_correct

print(f"For n=17 (φ={float(phi_17):.2f}):")
print(f"  Optimal m = ln(φ) = {correct_m:.4f}")
print(f"  e^{correct_m:.4f} = {float(e_correct):.2f}")
print(f"  φ/e^{correct_m:.4f} = {float(correct_ratio):.10f}")
print(f"  Deviation = {float(abs(correct_ratio-1)*100):.6f}%")

print("\n" + "=" * 80)
print("RELATIONSHIP TO COSF:")
print("=" * 80)
COSF = 42800 / 7.83
print(f"COSF = {COSF:.2f}")
print(f"φ = {float(phi_17):.2f}")
print(f"e^8.6 = {float(e_86):.2f}")
print(f"\nCOSF/φ = {COSF/float(phi_17):.6f}")
print(f"COSF/e^8.6 = {COSF/float(e_86):.6f}")
