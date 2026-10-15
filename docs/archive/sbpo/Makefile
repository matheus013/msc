MAIN := main
BUILD := build
LATEXMK := latexmk
LATEXMK_FLAGS := -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=$(BUILD)

PDF := $(MAIN).pdf
BUILD_PDF := $(BUILD)/$(MAIN).pdf

.PHONY: all clean

all: $(PDF)

$(PDF): $(MAIN).tex exemplo-latex.bib
	@mkdir -p $(BUILD)
	$(LATEXMK) $(LATEXMK_FLAGS) $(MAIN).tex
	@cp $(BUILD_PDF) $(PDF)

clean:
	@mkdir -p $(BUILD)
	-$(LATEXMK) -C -outdir=$(BUILD) $(MAIN).tex
	rm -rf $(BUILD)
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).fdb_latexmk $(MAIN).fls \
	      $(MAIN).log $(MAIN).lot $(MAIN).lof $(MAIN).out $(MAIN).toc \
	      $(MAIN).upa $(MAIN).xdv $(MAIN).nav $(MAIN).snm $(MAIN).vrb
	@echo "Limpeza concluida; PDF preservado: $(PDF)"