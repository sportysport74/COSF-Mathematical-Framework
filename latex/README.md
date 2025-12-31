# LaTeX Source

Complete LaTeX source for the COSF Mathematical Framework paper.

## Compilation

\\\ash
cd latex
make
\\\

Or manually:
\\\ash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
\\\

## Requirements

- pdflatex
- bibtex
- Standard LaTeX packages (amsmath, graphicx, etc.)

## Output

Generates \main.pdf\ - complete publication-ready paper.
