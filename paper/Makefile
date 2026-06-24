paper.pdf: paper.tex body.tex reference.bib figures
	pdflatex -interaction=nonstopmode paper.tex >/dev/null || true
	bibtex paper >/dev/null || true
	pdflatex -interaction=nonstopmode paper.tex >/dev/null || true
	pdflatex -interaction=nonstopmode paper.tex
figures:
	cd figures && python3 generate_figures.py
.PHONY: figures
