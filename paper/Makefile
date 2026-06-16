paper.pdf: paper.tex body.tex reference.bib figures/*.pdf
	pdflatex -interaction=nonstopmode paper.tex
	bibtex paper
	pdflatex -interaction=nonstopmode paper.tex
	pdflatex -interaction=nonstopmode paper.tex
figures:
	cd figures && python3 consolidate.py && python3 generate_figures.py
clean:
	rm -f *.aux *.bbl *.blg *.log *.out
