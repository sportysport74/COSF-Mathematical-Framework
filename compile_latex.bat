@echo off
cd latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
cd ..
echo Compilation complete!
