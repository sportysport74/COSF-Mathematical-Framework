# Makefile for COSF LaTeX compilation

.PHONY: all clean

all: main.pdf

main.pdf: main.tex ../supplementary/REFERENCES.bib
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

clean:
rm -f *.aux *.bbl *.blg *.log *.out *.toc *.synctex.gz

view: main.pdf
open main.pdf
