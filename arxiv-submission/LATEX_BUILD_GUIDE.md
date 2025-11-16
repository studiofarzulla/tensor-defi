# LaTeX Build Guide: Tensor Decomposition Paper

**Last Updated:** November 15, 2025
**Author:** Murad Farzulla
**Purpose:** Comprehensive documentation of pure LaTeX build system for arxiv-submission

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Build Instructions](#3-build-instructions)
4. [Document Structure](#4-document-structure)
5. [Critical Quirks & Gotchas](#5-critical-quirks--gotchas)
6. [File Structure](#6-file-structure)
7. [Compilation Errors & Solutions](#7-compilation-errors--solutions)
8. [Differences from Pandoc Version](#8-differences-from-pandoc-version)
9. [Advanced Topics](#9-advanced-topics)

---

## 1. Overview

### Why We Switched from Pandoc to Pure LaTeX

**Previous workflow (Pandoc/MyST):**
- Authored in Markdown with MyST syntax
- Used `pandoc-crossref` for figure/table references
- Compiled via Pandoc → LaTeX → PDF
- Single-column layout

**New workflow (Pure LaTeX):**
- Authored directly in LaTeX
- Native LaTeX cross-referencing (`\ref{}`, `\label{}`)
- Compiled via `pdflatex` + `bibtex`
- **Two-column layout for main body** (academic paper standard)

### Benefits of Two-Column Layout

1. **Standard format:** Most academic journals and conferences use two-column
2. **Better readability:** Shorter line lengths (50-75 chars) improve reading speed
3. **Space efficiency:** Fits more content per page
4. **Professional appearance:** Looks like a real published paper
5. **Figure spanning:** Can use `figure*` for full-width figures when needed

### Comparison with Previous Approach

| Feature | Pandoc/MyST | Pure LaTeX |
|---------|-------------|------------|
| Authoring | Markdown (easier) | LaTeX (more verbose) |
| Cross-refs | `@fig:label` | `\ref{fig:label}` |
| Citations | `[@key]` | `\citep{key}` |
| Layout control | Limited | Full control |
| Two-column | Hard to configure | Native support |
| Build time | Slower (2 tools) | Faster (direct) |
| Debugging | Harder (2 layers) | Easier (1 layer) |

**Verdict:** LaTeX is more verbose but gives complete control and matches academic publishing standards.

---

## 2. Prerequisites

### Required Packages (Arch Linux)

```bash
# Core LaTeX distribution
sudo pacman -S texlive-core texlive-latexextra texlive-bibtexextra

# Additional utilities (optional but recommended)
sudo pacman -S texlive-fontsextra texlive-science

# Build tools
sudo pacman -S make

# Optional: Live rebuild on file changes
sudo pacman -S inotify-tools
```

### For Other Distributions

**Ubuntu/Debian:**
```bash
sudo apt install texlive-latex-base texlive-latex-extra texlive-bibtex-extra
sudo apt install make
```

**macOS (via Homebrew):**
```bash
brew install --cask mactex
brew install make
```

**Verify Installation:**
```bash
pdflatex --version    # Should show pdfTeX
bibtex --version      # Should show BibTeX
make --version        # Should show GNU Make
```

### Package Overview

Our LaTeX preamble uses these packages (all included in texlive-latexextra):

- **inputenc/fontenc:** UTF-8 support, proper symbol rendering
- **geometry:** Page margins (1 inch all sides)
- **mathptmx:** Times New Roman font (standard for academic papers)
- **hyperref:** Clickable links and cross-references
- **amsmath/amssymb:** Math symbols and environments
- **natbib:** Citation management (author-year style)
- **graphicx:** Include figures
- **booktabs:** Professional tables
- **setspace:** Line spacing control
- **enumitem:** Customizable lists

---

## 3. Build Instructions

### Standard Workflow (4 Compilation Passes)

```bash
cd /home/kawaiikali/Resurrexi/projects/need-work/tensor-defi/arxiv-submission/

# Option 1: Use Makefile (recommended)
make

# Option 2: Manual commands
pdflatex Farzulla_2025_Tensor_Decomposition
bibtex Farzulla_2025_Tensor_Decomposition
pdflatex Farzulla_2025_Tensor_Decomposition
pdflatex Farzulla_2025_Tensor_Decomposition
```

### Why Multiple Passes Are Needed

LaTeX compilation requires **4 passes** to fully resolve all references:

**Pass 1: First pdflatex**
- Compiles document structure
- Creates `.aux` file with citation keys and label references
- Bibliography appears as `[?]` (unknown references)
- Cross-references show as `??` (undefined labels)
- **Output:** `Farzulla_2025_Tensor_Decomposition.aux`

**Pass 2: bibtex**
- Reads `.aux` file to find cited keys
- Looks up citations in `references.bib`
- Formats bibliography according to `\bibliographystyle{apalike}`
- **Output:** `Farzulla_2025_Tensor_Decomposition.bbl` (formatted bibliography)

**Pass 3: Second pdflatex**
- Incorporates `.bbl` file into document
- Bibliography now appears with actual entries
- Cross-references still show `??` (one-pass lag)
- **Output:** Updated `.aux` with resolved citations

**Pass 4: Third pdflatex**
- Resolves all cross-references
- Table of contents finalized
- All `\ref{}` commands show correct numbers
- **Output:** Final PDF with everything resolved

**Shortcut for quick iteration:**
```bash
make quick  # Single pass only - fast but broken refs
```

### Makefile Targets

```bash
make          # Full 4-pass build (default)
make pdf      # Same as make
make clean    # Remove auxiliary files (.aux, .log, etc.)
make distclean # Remove all generated files including PDF
make quick    # Single-pass compile (debugging LaTeX errors)
make watch    # Auto-rebuild on file changes (requires inotify-tools)
make wordcount # Estimate word count
make spell    # Interactive spell check (requires aspell)
make help     # Show all targets
```

### Troubleshooting Common Errors

**Error: "LaTeX Error: File `natbib.sty' not found"**
```bash
# Install missing LaTeX packages
sudo pacman -S texlive-latexextra
```

**Error: "! Undefined control sequence"**
- Check for typos in LaTeX commands
- Run `make quick` to isolate syntax errors
- Look at line number in error message

**Warning: "Citation X undefined"**
- Normal on first pass
- If persists after 4 passes: check `references.bib` has the key
- Verify BibTeX ran successfully (check for `.bbl` file)

**Warning: "Reference Y on page Z undefined"**
- Normal on passes 1-3
- If persists after pass 4: check `\label{Y}` exists in document

**Figures not appearing:**
- Check `\graphicspath{{figures/}}` matches your directory
- Verify figure files exist: `ls figures/`
- Use `.png` extension explicitly in `\includegraphics{file.png}`

---

## 4. Document Structure

### Complete Structure Overview

```latex
\documentclass{article}

% ==================== PREAMBLE ====================
% Package imports (inputenc, geometry, mathptmx, etc.)
% Graphics path: \graphicspath{{figures/}}
% Bibliography style: \bibliographystyle{apalike}
% Column gap: \setlength{\columnsep}{0.5in}
% Hyperref configuration

% ==================== METADATA ====================
\title{...}
\author{...}
\date{November 2025}

\begin{document}

% ==================== FRONT MATTER (SINGLE-COLUMN) ====================
\setstretch{1.5}  % Line spacing
\maketitle
\begin{abstract}...\end{abstract}
\tableofcontents

% ==================== SWITCH TO TWO-COLUMN ====================
\clearpage              % CRITICAL: Finish single-column page
\twocolumn              % Switch to two-column mode
\setstretch{1.5}        % CRITICAL: Re-apply line spacing!

% ==================== MAIN BODY (TWO-COLUMN) ====================
\section{Introduction}
...
\section{Related Work}
...
% Figures use figure* for full-width spanning
\begin{figure*}[htbp]...\end{figure*}

% ==================== ACKNOWLEDGMENTS ====================
\section*{Acknowledgments}  % Unnumbered section
...

% ==================== SWITCH BACK TO SINGLE-COLUMN ====================
\clearpage              % CRITICAL: Finish two-column page
\onecolumn              % Switch back to single-column
\setstretch{1.5}        % Re-apply line spacing

% ==================== BIBLIOGRAPHY (SINGLE-COLUMN) ====================
\bibliography{references}

\end{document}
```

### Key Structural Elements

**Preamble (Lines 1-50):**
- Package imports
- Settings (margins, fonts, spacing)
- Graphics path configuration
- Must come BEFORE `\begin{document}`

**Front Matter (Lines 51-100):**
- Title, author, date
- Abstract (full-width)
- Keywords
- Table of contents
- **Single-column layout** for better readability

**Two-Column Switch (Line ~100):**
```latex
\clearpage    % Finish current page
\twocolumn    % Activate two-column mode
\setstretch{1.5}  % MUST re-apply!
```

**Main Body (Lines 100-400):**
- All sections (Introduction → Conclusion)
- **Two-column layout**
- Use `figure*` and `table*` for full-width floats

**References Return (Line ~400):**
```latex
\clearpage    % Finish two-column content
\onecolumn    % Return to single-column
\setstretch{1.5}  % Re-apply spacing
\bibliography{references}
```

---

## 5. Critical Quirks & Gotchas

### ⚠️ QUIRK 1: Line Spacing Reset After Column Switches

**Problem:**
```latex
\setstretch{1.5}  % Set at document start
\twocolumn        % WIPES OUT LINE SPACING!
% Document now has single spacing (1.0)
```

**Solution:**
```latex
\twocolumn
\setstretch{1.5}  % MUST re-apply after \twocolumn!
```

**Why it happens:** LaTeX resets many formatting parameters when switching column modes. Always re-apply `\setstretch{}` after `\twocolumn` or `\onecolumn`.

**Learned from:** Event-study paper debugging (spent 2 hours figuring this out)

---

### ⚠️ QUIRK 2: Figure/Table Spanning in Two-Column

**Problem:**
```latex
\begin{figure}[htbp]  % Regular figure
  \includegraphics[width=0.95\textwidth]{wide_plot.png}
  % Image overflows column!
\end{figure}
```

**Solution:**
```latex
\begin{figure*}[htbp]  % Note the asterisk!
  \centering
  \includegraphics[width=0.95\textwidth]{wide_plot.png}
  \caption{Full-width figure spanning both columns}
  \label{fig:wide}
\end{figure*}
```

**Key differences:**
- `figure` → spans one column only (width = `\columnwidth`)
- `figure*` → spans both columns (width = `\textwidth`)
- Same for tables: `table` vs `table*`

**Width guidelines:**
- Single-column figure: `width=0.95\columnwidth`
- Two-column figure: `width=0.95\textwidth`

---

### ⚠️ QUIRK 3: Column Gap Must Be Set Before \twocolumn

**Problem:**
```latex
\twocolumn
\setlength{\columnsep}{0.5in}  % TOO LATE! Doesn't work
```

**Solution:**
```latex
% In preamble (before \begin{document})
\setlength{\columnsep}{0.5in}  % Set gap BEFORE \twocolumn
```

**Why:** Column gap is fixed when `\twocolumn` is invoked. Setting it afterward has no effect.

**Recommendation:** Always set in preamble with other geometry settings.

---

### ⚠️ QUIRK 4: Clearing Pages Before Column Switches

**Problem:**
```latex
\section{Conclusion}  % Last section in two-column
\onecolumn            % Switch immediately
\bibliography{references}
% Bibliography appears mid-page, looks terrible
```

**Solution:**
```latex
\section{Conclusion}
\clearpage            % FINISH current two-column page
\onecolumn            % Then switch
\bibliography{references}
```

**Why:** `\clearpage` flushes all pending floats and forces a page break. Without it, column switch happens mid-page, causing layout chaos.

**Rule:** ALWAYS use `\clearpage` before `\twocolumn` or `\onecolumn`.

---

### ⚠️ QUIRK 5: Graphics Path Requires Trailing Slash

**Problem:**
```latex
\graphicspath{{figures}}  % Missing trailing slash
\includegraphics{plot.png}  % File not found!
```

**Solution:**
```latex
\graphicspath{{figures/}}  % Trailing slash required
\includegraphics{plot.png}  % Now works
```

**Why:** LaTeX literally prepends the path. Without `/`, it looks for `figuresplot.png` instead of `figures/plot.png`.

---

### ⚠️ QUIRK 6: Citation Commands (natbib)

**Available commands:**

```latex
% Author-year citations
\citet{Kolda2009}         % → Kolda and Bader (2009)
\citep{Kolda2009}         % → (Kolda and Bader, 2009)
\citet*{Kolda2009}        % → Kolda et al. (2009)  [3+ authors]
\citep*{Kolda2009}        % → (Kolda et al., 2009)

% Multiple citations
\citep{Kolda2009,Han2022} % → (Kolda and Bader, 2009; Han et al., 2022)

% Textual citations
\citeauthor{Kolda2009}    % → Kolda and Bader
\citeyear{Kolda2009}      % → 2009

% Numerical citations (if using \bibliographystyle{plain})
\cite{Kolda2009}          % → [1]
```

**Conversion from Pandoc:**
- `[@Kolda2009]` → `\citep{Kolda2009}`
- `@Kolda2009` → `\citet{Kolda2009}`
- `[@Kolda2009; @Han2022]` → `\citep{Kolda2009,Han2022}`

---

### ⚠️ QUIRK 7: Bibliography Placement and Style

**Our configuration:**
```latex
% In preamble
\bibliographystyle{apalike}  % APA-style author-year

% At end of document
\clearpage
\onecolumn
\setstretch{1.5}
\bibliography{references}  % No .bib extension!
```

**Common styles:**
- `apalike` - APA-like author-year (we use this)
- `plainnat` - Plain author-year
- `abbrvnat` - Abbreviated author-year
- `plain` - Numerical [1]
- `alpha` - Alphabetic [Kol09]

**File naming:** LaTeX automatically appends `.bib`, so use `\bibliography{references}` not `\bibliography{references.bib}`.

---

### ⚠️ QUIRK 8: Float Placement Options

**Placement specifiers:**
```latex
\begin{figure}[htbp]
% h = here (try to place at current position)
% t = top of page
% b = bottom of page
% p = separate page for floats
% ! = override LaTeX's restrictions
% H = EXACTLY here (requires \usepackage{float})
```

**Recommended:** `[htbp]` for most cases, `[!htbp]` if LaTeX is being stubborn.

**If figures drift too much:**
```latex
\usepackage{placeins}  % Add to preamble
...
\section{Results}
\FloatBarrier  % Forces all pending floats to appear before this point
```

---

### ⚠️ QUIRK 9: Math in Two-Column Layout

**Problem:** Display equations can be cramped in narrow columns.

**Solutions:**
```latex
% Small inline equations - no problem
The variance is $\sigma^2 = \text{Var}(X)$.

% Medium equations - use equation environment
\begin{equation}
  \mathcal{X} \approx \sum_{r=1}^{R} \lambda_r \cdot (\mathbf{a}_r \otimes \mathbf{b}_r)
\end{equation}

% Long equations - may overflow column
% Solution 1: Use smaller font
\begin{equation}
  \small
  \mathcal{X} \approx \sum_{r=1}^{R} \lambda_r \cdot (\mathbf{a}_r \otimes \mathbf{b}_r \otimes \mathbf{c}_r \otimes \mathbf{d}_r)
\end{equation}

% Solution 2: Multi-line (align environment)
\begin{align}
  \mathcal{X} &\approx \sum_{r=1}^{R} \lambda_r \cdot (\mathbf{a}_r \otimes \mathbf{b}_r \\
              &\quad \otimes \mathbf{c}_r \otimes \mathbf{d}_r)
\end{align}

% Solution 3: Use figure* for very wide equations
\begin{figure*}[htbp]
  \[
  \text{Very long equation spanning both columns}
  \]
\end{figure*}
```

---

### ⚠️ QUIRK 10: Cross-Reference Labels

**LaTeX conventions:**
```latex
% Figures
\label{fig:rank-selection}  % Prefix with fig:
\ref{fig:rank-selection}    % Reference
Figure~\ref{fig:rank-selection}  % With text

% Tables
\label{tbl:reconstruction}  % Prefix with tbl:
Table~\ref{tbl:reconstruction}

% Equations
\label{eq:cp}              % Prefix with eq:
Equation~\ref{eq:cp}

% Sections
\label{sec:methodology}    % Prefix with sec:
Section~\ref{sec:methodology}
```

**Tilde (`~`) creates non-breaking space:** `Figure~\ref{}` prevents line break between "Figure" and number.

**Conversion from Pandoc/MyST:**
- `@fig:rank-selection` → `Figure~\ref{fig:rank-selection}`
- `@tbl:reconstruction` → `Table~\ref{tbl:reconstruction}`
- `@eq:cp` → `Equation~\ref{eq:cp}`

---

## 6. File Structure

```
arxiv-submission/
├── Farzulla_2025_Tensor_Decomposition.tex  # Main LaTeX source
├── references.bib                          # Bibliography database
├── figures/                                # All figure files
│   ├── rank_selection.png
│   ├── cp_factors.png
│   ├── method_comparison.png
│   ├── reconstruction_quality.png
│   ├── temporal_evolution.png
│   └── asset_factor_space.png
├── Makefile                                # Build automation
├── LATEX_BUILD_GUIDE.md                   # This documentation
└── README.md                              # Project overview

Generated files (after compilation):
├── Farzulla_2025_Tensor_Decomposition.pdf  # Final output
├── Farzulla_2025_Tensor_Decomposition.aux  # Auxiliary references
├── Farzulla_2025_Tensor_Decomposition.log  # Compilation log
├── Farzulla_2025_Tensor_Decomposition.out  # Hyperref output
├── Farzulla_2025_Tensor_Decomposition.toc  # Table of contents
├── Farzulla_2025_Tensor_Decomposition.bbl  # Formatted bibliography
└── Farzulla_2025_Tensor_Decomposition.blg  # BibTeX log
```

**Cleaning up:**
```bash
make clean     # Remove generated files except PDF
make distclean # Remove everything including PDF
```

---

## 7. Compilation Errors & Solutions

### Common LaTeX Errors

**1. Undefined control sequence**

```
! Undefined control sequence.
l.123 \myfakecommand
```

**Cause:** Typo in command name or missing package.

**Solution:**
- Check line 123 for typos
- If using custom command, verify it's defined
- If using package-specific command, verify package is loaded

---

**2. Missing $ inserted**

```
! Missing $ inserted.
<inserted text>
                $
l.145 The value of x_i is important
```

**Cause:** Math mode character (`_`, `^`, etc.) used in text mode.

**Solution:**
```latex
% Wrong
The value of x_i is important

% Right
The value of $x_i$ is important
```

---

**3. File not found**

```
! LaTeX Error: File `natbib.sty' not found.
```

**Cause:** Missing LaTeX package.

**Solution:**
```bash
sudo pacman -S texlive-latexextra  # Arch
sudo apt install texlive-latex-extra  # Ubuntu
```

---

**4. Bibliography not appearing**

**Symptoms:** `\bibliography{references}` produces no output, or citations show as `[?]`.

**Debugging checklist:**
```bash
# 1. Verify .bib file exists
ls references.bib

# 2. Check BibTeX ran successfully
ls Farzulla_2025_Tensor_Decomposition.bbl

# 3. Check BibTeX log for errors
cat Farzulla_2025_Tensor_Decomposition.blg

# 4. Verify citations in .aux file
grep citation Farzulla_2025_Tensor_Decomposition.aux

# 5. Rerun full 4-pass workflow
make distclean && make
```

**Common causes:**
- Forgot to run `bibtex` (pass 2)
- `.bib` file has syntax errors
- Citation keys don't match (case-sensitive!)
- No `\cite{}` commands in document (BibTeX skips uncited entries)

---

**5. Figures not appearing**

**Symptoms:** `! File `plot.png' not found.`

**Debugging:**
```bash
# 1. Check figure exists
ls figures/plot.png

# 2. Verify graphics path in preamble
grep graphicspath Farzulla_2025_Tensor_Decomposition.tex
# Should show: \graphicspath{{figures/}}

# 3. Check for typos in filename (case-sensitive!)
# 4. Try absolute path as test
\includegraphics{/full/path/to/figure.png}
```

---

### Common Warnings (Usually Safe to Ignore)

```
Overfull \hbox (2.34pt too wide) in paragraph at lines 145--147
```
**Meaning:** Text slightly overflows margin (by 2.34 points).
**Action:** Usually cosmetic, can ignore unless severe (>10pt).

```
LaTeX Warning: Reference `fig:xyz' on page 3 undefined on input line 123.
```
**Meaning:** Cross-reference not resolved (normal on passes 1-3).
**Action:** Ignore if disappears after pass 4. If persists, check `\label{fig:xyz}` exists.

```
Package natbib Warning: Citation `Kolda2009' on page 5 undefined on input line 200.
```
**Meaning:** Citation not found (normal on passes 1-2).
**Action:** Ignore if disappears after pass 4. If persists, check `references.bib` has the key.

---

## 8. Differences from Pandoc Version

### What We Gained

**1. Two-column layout**
- Professional academic paper appearance
- Better space utilization
- Standard format for journals/conferences

**2. Better formatting control**
- Direct control over spacing, margins, fonts
- Can tweak any aspect without fighting Pandoc
- Native LaTeX figure placement options

**3. Simpler build system**
- One tool (pdflatex) instead of two (pandoc + pdflatex)
- Easier debugging (single layer, clearer error messages)
- Faster compilation (no Pandoc overhead)

**4. Standard academic workflow**
- Most journals provide LaTeX templates
- Easier to submit to arXiv, journals
- Collaborators familiar with LaTeX can contribute directly

---

### What We Lost

**1. Markdown authoring simplicity**

**Pandoc:**
```markdown
# Introduction

Cryptocurrency markets are cool [@Kolda2009].

![Rank selection](figures/rank_selection.png){#fig:rank width=80%}

See @fig:rank for details.
```

**LaTeX:**
```latex
\section{Introduction}

Cryptocurrency markets are cool \citep{Kolda2009}.

\begin{figure*}[htbp]
  \centering
  \includegraphics[width=0.8\textwidth]{rank_selection.png}
  \caption{Rank selection}
  \label{fig:rank}
\end{figure*}

See Figure~\ref{fig:rank} for details.
```

**Verdict:** LaTeX is more verbose, but you get used to it. The control is worth it.

---

**2. Pandoc-crossref automation**

**Pandoc:** Auto-numbered figures, tables, equations with `@fig:`, `@tbl:`, `@eq:`
**LaTeX:** Manual `\ref{}` but more flexible and standard

**Migration note:** Search/replace makes conversion easy:
```bash
# Convert figure references
sed -i 's/@fig:\([a-z-]*\)/Figure~\\ref{fig:\1}/g' paper.tex

# Convert citations
sed -i 's/\[@\([A-Za-z0-9]*\)\]/\\citep{\1}/g' paper.tex
```

---

**3. Easier table authoring**

**Pandoc:** Markdown tables are simpler:
```markdown
| Asset | Price | Return |
|-------|-------|--------|
| BTC   | 66K   | +88%   |
```

**LaTeX:** More verbose but more control:
```latex
\begin{table}[h]
\centering
\begin{tabular}{lcc}
\toprule
Asset & Price & Return \\
\midrule
BTC & 66K & +88\% \\
\bottomrule
\end{tabular}
\end{table}
```

**Tool:** Can use [Tables Generator](https://www.tablesgenerator.com/) to convert Markdown → LaTeX tables.

---

### Migration Path (If Needed)

**Going back to Pandoc:**

1. Keep `.bib` file (compatible)
2. Convert LaTeX → Markdown:
   ```bash
   pandoc paper.tex -o paper.md
   ```
3. Fix figure references: `\ref{fig:x}` → `@fig:x`
4. Fix citations: `\citep{x}` → `[@x]`

**Hybrid approach:**

- Write in Markdown
- Convert to LaTeX for final formatting
- Best of both worlds (authoring simplicity + layout control)

---

## 9. Advanced Topics

### Custom Figure Widths

```latex
% Single-column figures
\begin{figure}[htbp]
  \centering
  \includegraphics[width=0.9\columnwidth]{narrow_plot.png}
  \caption{Narrow plot}
\end{figure}

% Two-column spanning figures
\begin{figure*}[htbp]
  \centering
  \includegraphics[width=0.95\textwidth]{wide_plot.png}
  \caption{Wide plot}
\end{figure*}

% Fixed width (in inches)
\includegraphics[width=3in]{plot.png}

% Fixed height
\includegraphics[height=2in]{plot.png}

% Scale factor
\includegraphics[scale=0.5]{plot.png}
```

---

### Side-by-Side Figures

```latex
\begin{figure*}[htbp]
  \centering
  \begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{plot1.png}
    \caption{First plot}
    \label{fig:plot1}
  \end{minipage}
  \hfill
  \begin{minipage}{0.48\textwidth}
    \centering
    \includegraphics[width=\textwidth]{plot2.png}
    \caption{Second plot}
    \label{fig:plot2}
  \end{minipage}
\end{figure*}
```

---

### Subfigures (Requires subcaption Package)

```latex
% Add to preamble
\usepackage{subcaption}

% In document
\begin{figure*}[htbp]
  \centering
  \begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{plot1.png}
    \caption{First subplot}
    \label{fig:sub1}
  \end{subfigure}
  \hfill
  \begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{plot2.png}
    \caption{Second subplot}
    \label{fig:sub2}
  \end{subfigure}
  \caption{Overall caption for both subfigures}
  \label{fig:combined}
\end{figure*}

% Reference: Figure~\ref{fig:combined} shows... Subplot~\ref{fig:sub1}...
```

---

### Custom Section Numbering

```latex
% Unnumbered section (doesn't appear in TOC)
\section*{Acknowledgments}

% Unnumbered but in TOC
\section*{Acknowledgments}
\addcontentsline{toc}{section}{Acknowledgments}

% Change numbering depth (show subsections in TOC)
\setcounter{tocdepth}{2}  % 1=sections, 2=subsections, 3=subsubsections

% Restart numbering
\setcounter{section}{0}
```

---

### Custom List Formatting

```latex
% Compact lists (no spacing between items)
\begin{itemize}[noitemsep]
  \item First item
  \item Second item
\end{itemize}

% Custom bullets
\begin{itemize}[label=$\triangleright$]
  \item Triangle bullet
\end{itemize}

% Custom left margin
\begin{itemize}[leftmargin=*]  % Align with text
  \item Item
\end{itemize}
```

---

### Page Layout Tweaks

```latex
% Adjust column separation mid-document
\setlength{\columnsep}{0.3in}  % Narrower gap
\twocolumn  % Apply new setting

% Force column break
\columnbreak  % Inside two-column mode

% Balance columns on last page
\usepackage{balance}  % Add to preamble
\balance  % At end of document before \onecolumn
```

---

### Bibliography Customization

```latex
% Compress year ranges (2020a, 2020b → 2020a,b)
\usepackage[authoryear,round,compress]{natbib}

% Sort citations chronologically
\bibliographystyle{unsrtnat}

% Custom cite separator
\setcitestyle{citesep={;}}  % Use semicolon instead of comma
```

---

### PDF Metadata

```latex
\hypersetup{
    pdftitle={Tensor Decomposition for Cryptocurrency Markets},
    pdfauthor={Murad Farzulla},
    pdfsubject={Quantitative Finance},
    pdfkeywords={tensor decomposition, cryptocurrency, market microstructure},
}
```

Metadata appears in PDF properties (useful for arXiv submission).

---

## Summary Checklist

**Before first compilation:**
- [ ] Verify all packages installed (`texlive-latexextra`)
- [ ] Check `\graphicspath{{figures/}}` matches your directory
- [ ] Verify all figure files exist: `ls figures/*.png`
- [ ] Check `references.bib` has all cited keys

**Common first-time issues:**
- [ ] Line spacing reset after `\twocolumn` → Add `\setstretch{1.5}`
- [ ] Figures not spanning columns → Use `figure*` not `figure`
- [ ] Bibliography empty → Run full 4-pass workflow
- [ ] Cross-refs showing `??` → Need 4 passes to resolve

**Build workflow:**
```bash
make          # Full 4-pass build
make clean    # Remove auxiliary files
make distclean # Start fresh
```

**For help:**
- LaTeX errors: Check section 7 above
- Quirks/gotchas: See section 5
- Advanced formatting: See section 9

---

**Last Updated:** November 15, 2025
**Contact:** contact@farzulla.org
**Repository:** Will be made public upon publication
