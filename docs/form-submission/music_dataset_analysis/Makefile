.PHONY: pdf clean

pdf: main.pdf

main.pdf: main.tex references.bib
	tectonic main.tex

clean:
	rm -f main.pdf *.aux *.bbl *.blg *.log *.out
