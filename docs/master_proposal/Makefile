PDFLATEX := /Library/TeX/texbin/pdflatex
BIBTEX   := /Library/TeX/texbin/bibtex
BUILD    := build
MAIN     := tcc

FLAGS := -interaction=nonstopmode -output-directory=$(BUILD)

.PHONY: all clean

all: $(BUILD)/$(MAIN).pdf

$(BUILD)/$(MAIN).pdf: $(MAIN).tex referencias.bib
	@mkdir -p $(BUILD)/capitulos
	$(PDFLATEX) $(FLAGS) $(MAIN).tex
	$(BIBTEX) $(BUILD)/$(MAIN)
	$(PDFLATEX) $(FLAGS) $(MAIN).tex
	$(PDFLATEX) $(FLAGS) $(MAIN).tex
	$(PDFLATEX) $(FLAGS) $(MAIN).tex

clean:
	rm -f $(BUILD)/$(MAIN).pdf  $(BUILD)/$(MAIN).aux  $(BUILD)/$(MAIN).bbl \
	      $(BUILD)/$(MAIN).blg  $(BUILD)/$(MAIN).hd   $(BUILD)/$(MAIN).lof \
	      $(BUILD)/$(MAIN).log  $(BUILD)/$(MAIN).lot  $(BUILD)/$(MAIN).out \
	      $(BUILD)/$(MAIN).toc  $(BUILD)/$(MAIN).upa  $(BUILD)/$(MAIN).fls \
	      $(BUILD)/$(MAIN).fdb_latexmk $(BUILD)/$(MAIN).xdv \
	      $(BUILD)/capitulos/*.aux IC-eps-converted-to.pdf
