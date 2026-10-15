.PHONY: pdf clean

pdf: report.pdf

report.pdf: report.md references.bib ieee-template.tex ieee.csl
	pandoc report.md \
	  --from markdown \
	  --pdf-engine=tectonic \
	  --template=ieee-template.tex \
	  --citeproc \
	  --bibliography=references.bib \
	  --csl=ieee.csl \
	  -o report.pdf

clean:
	rm -f report.pdf
