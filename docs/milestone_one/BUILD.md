# Milestone One Typesetting Guide

## Structure

```
milestone_one/
├── milestone_one.tex          # Main file (uses \input{} for modular editing)
├── sections/                  # Modular section files
│   ├── 01-intro.tex          # Research question & significance, context
│   ├── 02-methods.tex        # Proposed methods
│   ├── 03-preprocessing.tex  # Data & preprocessing
│   ├── 04-diagram.tex        # Information flow diagram
│   ├── 05-setup.tex          # Experimental setup & baselines
│   └── 06-timeline.tex       # Timeline to Milestone 2
├── assets/
│   └── info-flow-diagram.jpg
├── references.bib            # Bibliography (BibTeX format)
├── sn-jnl.cls               # Springer Nature journal class
└── BUILD.md                 # This file
```

## Compilation

### Using `latexmk` (Recommended)

```bash
cd docs/milestone_one
latexmk -pdf -pdflatex='pdflatex -interaction=nonstopmode' milestone_one.tex
```

This automatically handles multiple passes and bibliography generation.

### Manual compilation

```bash
cd docs/milestone_one

# First compilation
pdflatex milestone_one.tex

# Generate bibliography
bibtex milestone_one

# Two more LaTeX passes for correct citations and formatting
pdflatex milestone_one.tex
pdflatex milestone_one.tex
```

The output PDF will be: `milestone_one.pdf`

**Note:** The `.bst` bibliography style files are in `bst/`. If `bibtex` complains about missing style files, copy them to the root directory:
```bash
cp bst/*.bst .
```

## Editing Workflow

### For Collaborative Writing

1. **Edit section files** (e.g., `sections/02-methods.tex`) directly
   - Each team member can edit a different section without conflicts
   - Use meaningful LaTeX section markers (`\section*{}`, `\subsection*{}`)

2. **Update references** in `references.bib` using BibTeX format
   - Use citation keys like `\cite{bib1}` in the text
   - Compile to auto-generate formatted bibliography

3. **Recompile** after changes:
   ```bash
   latexmk -pdf milestone_one.tex
   ```

### For LLM Assistance

- Share specific section files (e.g., `sections/02-methods.tex`) for editing
- Use consistent LaTeX formatting and citation style
- The modular structure makes it easy to refactor sections independently

## Submission (Flattening)

For final submission to journals that require a single `.tex` file:

```bash
cd docs/milestone_one
latexmk -c  # Clean intermediate files
```

Then copy all `.tex` content into a single file, removing the `\input{}` statements and replacing them with the actual section content. The `.bib` entries can be converted to `.bbl` format using:

```bash
bibtex milestone_one
```

And then embedded directly in the final document.

## Style Notes

- **Document Class**: `sn-jnl` with `sn-mathphys-num` style (numbered references)
- **Page Limits**: 3 pages for main content + unlimited references
- **Margins**: Handled by class file (2 cm)
- **Font**: Times New Roman 11pt (default for class)
- **Formatting**: Use `\textit{}` for italic, `\textbf{}` for bold, `\cite{}` for citations

## Common Commands

| Purpose | Command |
|---------|---------|
| Section (no number) | `\section*{Title}` |
| Subsection (no number) | `\subsection*{Title}` |
| Italic text | `\textit{...}` |
| Bold text | `\textbf{...}` |
| Citation | `\cite{bib1}` |
| Footnote | `\footnote{...}` |
| Math inline | `$...$` |
| Math display | `\[...\]` |
| Figure | `\begin{figure}...\end{figure}` |

## Tips

1. **Keep sections modular**: Don't worry about page breaks in individual sections; `latexmk` handles the full layout automatically.

2. **Use section markers sparingly**: The milestone uses `\section*{}` (no numbering) per the guidelines.

3. **Images**: Store in `assets/` and reference with relative paths:
   ```latex
   \includegraphics[width=0.78\textwidth]{assets/info-flow-diagram.jpg}
   ```

4. **Collaborative Git workflow**:
   ```bash
   git checkout -b milestone-one-edits
   # Edit sections/02-methods.tex
   git add sections/02-methods.tex
   git commit -m "feat: expand methods section with CReM details"
   git push origin milestone-one-edits
   # Create PR for review
   ```

5. **Version control**: Commit `.tex` and `.bib` files; ignore `.pdf`, `.aux`, `.log`, `.fls`, `.fdb_latexmk` files (add to `.gitignore` if desired).

## Troubleshooting

### `! Package babel Error: Unknown language 'english'`
→ Install TeX Live or update packages

### Bibliography not appearing
→ Run `bibtex milestone_one` after `pdflatex`, then `pdflatex` again

### Image not found
→ Check path is relative from `milestone_one.tex` location. Use `assets/info-flow-diagram.jpg`

### Package not found
→ Try `pdflatex` instead of `xelatex`; Springer template is optimized for `pdflatex`

---

**Last updated**: March 2026
