MAIN := main
LATEXMK := latexmk -xelatex -interaction=nonstopmode -halt-on-error -file-line-error

.PHONY: all clean

all:
	$(LATEXMK) $(MAIN).tex

clean:
	$(LATEXMK) -C $(MAIN).tex
	rm -f $(MAIN).nav $(MAIN).snm $(MAIN).vrb
