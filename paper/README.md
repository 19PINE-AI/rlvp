# Paper source

LaTeX source for **RLVP: Penalize the Path, Reward the Outcome**.

- **Published version:** https://arxiv.org/abs/2607.07435 (arXiv:2607.07435)
- **PDF:** https://arxiv.org/pdf/2607.07435

The compiled `paper.pdf` is not committed — build it from source or read it on
arXiv.

## Build

```bash
make figures   # regenerate figures/ from the result dumps (needs matplotlib)
make           # pdflatex + bibtex -> paper.pdf
```

## Layout

- `paper.tex` — top-level document (loads `arxiv.sty`)
- `body.tex`, `appendix.tex` — main text and appendix
- `table_*.tex` — generated result tables
- `reference.bib` — bibliography
- `figures/` — figure PDFs plus the scripts that generate them
  (`generate_figures.py`, `consolidate.py`) from `../results`
