.PHONY: pdf clean

pdf: report.pdf

report.pdf: report.md references.bib ieee-template.tex
	pandoc report.md \
	  --from markdown-smart \
	  --pdf-engine=tectonic \
	  --template=ieee-template.tex \
	  --natbib \
	  -o report.pdf

clean:
	rm -f report.pdf
