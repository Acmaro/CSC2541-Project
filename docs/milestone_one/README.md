# Milestone One: Expanding Synthesizable Chemical Space

Benchmarking Post-Generation Molecular Modification

**Authors:** Tyler Biswurm, Jingxin Huang, Yuma Nakamura, Ben Luo
**Date:** March 2026
**Deadline:** March 8, 2026

## Quick Start

### View the PDF
```bash
open milestone_one.pdf
```

### Edit a section
1. Open a section file in `sections/` (e.g., `sections/02-methods.tex`)
2. Make your changes
3. Recompile: `latexmk -pdf milestone_one.tex`

### Add/update references
1. Edit `references.bib` in BibTeX format
2. Use `\cite{bibkey}` in section files
3. Recompile—bibliography updates automatically

## Directory Structure

```
milestone_one/
├── milestone_one.tex          # Main LaTeX file (uses \input{} to include sections)
├── milestone_one.pdf          # Compiled output (CURRENT)
│
├── sections/                  # Modular section files
│   ├── 01-intro.tex          # Research question & significance, context
│   ├── 02-methods.tex        # Proposed methods
│   ├── 03-preprocessing.tex  # Data & preprocessing
│   ├── 04-diagram.tex        # Information flow diagram
│   ├── 05-setup.tex          # Experimental setup & baselines
│   └── 06-timeline.tex       # Timeline to Milestone 2
│
├── assets/                    # Figures and images
│   └── info-flow-diagram.jpg
│
├── references.bib            # Bibliography (BibTeX format)
├── sn-jnl.cls               # Springer Nature journal class
├── bst/                       # Bibliography style files (for bibtex)
│   ├── sn-mathphys-num.bst   # Current style
│   └── [other styles]
│
├── BUILD.md                  # Compilation and workflow guide
├── README.md                 # This file
└── .gitignore               # Git ignores for artifacts
```

## Workflow

### For collaborative editing:
- **Each team member** edits their section independently
- **Git workflow**:
  ```bash
  git checkout -b milestone-section-<name>
  # Edit sections/<number>-<name>.tex or references.bib
  git add sections/<number>-<name>.tex
  git commit -m "feat: expand <section> with <detail>"
  git push origin milestone-section-<name>
  # Create PR for review
  ```

### For LLM assistance:
- Share the specific `.tex` file you want edited
- The LLM can edit directly and suggest changes
- You review and approve before merging

### For compilation:
- **One-liner:** `latexmk -pdf milestone_one.tex`
- **Manual:** See `BUILD.md`

## Style Guidelines

- **Section headings:** Use `\section*{Title}` (no numbering)
- **Subsections:** Use `\subsection*{Title}` if needed
- **Citations:** Use `\cite{bibkey}` (e.g., `\cite{bib1}`)
- **Emphasis:** Use `\textit{...}` (italic) or `\textbf{...}` (bold)
- **Math:** Inline with `$...$`, display with `\[...\]`
- **Line breaks:** Use `\\` sparingly; let LaTeX handle flow

## Page Limits

Per guidelines:
- **Part I (Intro & Background):** 1 page max
- **Part II (Methods & Setup):** 2 pages max
- **References:** Unlimited (extra pages)
- **Font:** Times New Roman 11pt, single-space
- **Margins:** 2 cm (handled by class file)

## Citation Style

All references use numbered citations (`[1]`, `[2]`, etc.) with DOI URLs.

**Example:**
```latex
Some fact about synthesis-aware generation \cite{bib2}, and another claim \cite{bib7}.
```

Becomes:
> Some fact about synthesis-aware generation [2], and another claim [7].

## Submission Prep

Before final submission, you may need to **flatten** the `.tex` file (remove `\input{}` statements) depending on journal requirements.

See `BUILD.md` for details on converting to a single-file submission.

---

**For detailed compilation & workflow info, see** `BUILD.md`
