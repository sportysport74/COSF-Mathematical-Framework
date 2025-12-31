import sys

with open("code/gpu/cuda_convergence_search.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find and fix the m_range calculation (around line 100-101)
for i, line in enumerate(lines):
    if "m_min = max(m_range[0], m_optimal - 2.0)" in line:
        lines[i] = "                m_min = max(m_range[0], m_optimal - 1.0)\n"
    if "m_max = min(m_range[1], m_optimal + 2.0)" in line:
        lines[i] = "                m_max = min(m_range[1], m_optimal + 1.0)\n"

with open("code/gpu/cuda_convergence_search.py", "w", encoding="utf-8") as f:
    f.writelines(lines)

print("Fixed m_range to 1.0 instead of 2.0")
