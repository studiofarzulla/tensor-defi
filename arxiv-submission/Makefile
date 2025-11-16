# =============================================================================
# Makefile for Tensor Decomposition Paper (Pure LaTeX Build)
# Author: Murad Farzulla
# Date: November 2025
# =============================================================================

MAIN = Farzulla_2025_Tensor_Decomposition
TEX = $(MAIN).tex
PDF = $(MAIN).pdf
BIB = references.bib

# Default target
.PHONY: all
all: pdf

# Build PDF with full bibliography workflow
# Why 4 passes?
#   1. pdflatex: Generate .aux file with citations and references
#   2. bibtex: Process bibliography, create .bbl file
#   3. pdflatex: Incorporate bibliography into document
#   4. pdflatex: Resolve all cross-references and citations
.PHONY: pdf
pdf: $(PDF)

$(PDF): $(TEX) $(BIB)
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

# Clean auxiliary files (keep PDF)
.PHONY: clean
clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.lof *.lot

# Clean everything including PDF
.PHONY: distclean
distclean: clean
	rm -f $(PDF)

# Quick single-pass compile (for debugging LaTeX errors)
# WARNING: Citations and cross-references will be broken
.PHONY: quick
quick:
	pdflatex $(MAIN)

# Watch mode: rebuild on file changes (requires inotify-tools)
# Install: sudo pacman -S inotify-tools
.PHONY: watch
watch:
	while inotifywait -e modify $(TEX) $(BIB); do make pdf; done

# Word count (rough estimate via detex)
.PHONY: wordcount
wordcount:
	detex $(TEX) | wc -w

# Spell check (requires aspell)
.PHONY: spell
spell:
	aspell -t -c $(TEX)

# Help
.PHONY: help
help:
	@echo "Tensor Decomposition Paper Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make          - Build PDF (full 4-pass workflow)"
	@echo "  make pdf      - Same as 'make'"
	@echo "  make clean    - Remove auxiliary files (.aux, .log, etc.)"
	@echo "  make distclean - Remove all generated files including PDF"
	@echo "  make quick    - Single-pass compile (fast, broken refs)"
	@echo "  make watch    - Auto-rebuild on file changes (requires inotify-tools)"
	@echo "  make wordcount - Estimate word count"
	@echo "  make spell    - Interactive spell check (requires aspell)"
	@echo ""
	@echo "Files:"
	@echo "  Input:  $(TEX) + $(BIB)"
	@echo "  Output: $(PDF)"
