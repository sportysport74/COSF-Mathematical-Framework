# COSF Quick Start Guide

Get up and running with the COSF Mathematical Framework in 5 minutes!

## 1. Clone the Repository

\\\ash
git clone https://github.com/sportysport74/COSF-Mathematical-Framework.git
cd COSF-Mathematical-Framework
\\\

## 2. Install Dependencies

\\\ash
pip install -r requirements.txt
\\\

## 3. Run the Tests

\\\ash
pytest tests/ -v
\\\

Expected output:
\\\
 10 tests passed in 0.5s
\\\

## 4. Run Example Scripts

### Verify Convergence
\\\ash
python code/examples/cosf_verification.py
\\\

Output:
\\\
φ¹⁷ = 5472.9992880337
e^8.6 = 5434.6440642301
φ¹⁷/e^8.6 = 1.0070549839
Deviation: 0.71%
✓ Convergence verified!
\\\

### Generate Visualizations
\\\ash
python code/examples/convergence_analysis.py
\\\

Saves comprehensive 6-panel plot to \igures/convergence_plots/\

## 5. Explore Interactive Notebooks

\\\ash
jupyter notebook notebooks/cosf_convergence_demo.ipynb
\\\

Or the full analysis:
\\\ash
jupyter notebook code/examples/COSF_Analysis.ipynb
\\\

## 6. Read the Paper

- **Quick version:** [README.md](README.md)
- **Complete paper:** [COSF_Mathematical_Framework.md](COSF_Mathematical_Framework.md)
- **LaTeX source:** [latex/COSF_Mathematical_Framework.tex](latex/COSF_Mathematical_Framework.tex)

## 7. Explore Documentation

- [Experimental Protocols](docs/EXPERIMENTAL_PROTOCOLS.md) - Lab procedures
- [Mathematical Proofs](docs/MATHEMATICAL_PROOFS.md) - Extended derivations
- [Project Summary](PROJECT_SUMMARY.md) - Complete overview

## Need Help?

-  [Full Documentation](docs/)
-  [Report Issues](https://github.com/sportysport74/COSF-Mathematical-Framework/issues)
-  [Discussions](https://github.com/sportysport74/COSF-Mathematical-Framework/issues)

---

** Star the repo if you find it useful!**
