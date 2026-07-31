# PACLIC 40 Multilingual RAG Paper Skeleton

This folder contains an anonymous ACL-style LaTeX skeleton for the proposed paper:

**From Articles to Sentences: Retrieval Granularity and Distractor Control for Multilingual RAG**

## Before compiling

PACLIC 40 requires the ACL 2026 two-column format. Download the current official ACL style files from the ACL style-file repository and place these two files in this folder:

- `acl.sty`
- `acl_natbib.bst`

Then compile:

```bash
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## Important

- Keep `\usepackage[review]{acl}` for the anonymous review submission.
- The current generation results are preliminary (`n=10`) and are explicitly labeled as such.
- Replace every bold `[TODO: ...]` marker before submission.
- Omit acknowledgments in the review version.
- Run ACL Pubcheck on the final PDF.
- Confirm the PACLIC 40 CFP immediately before submission in case requirements change.
