.PHONY: pdf tex clean

pdf: report.pdf

tex: report.tex

report.tex: report.md references.bib ieee-template.tex
	pandoc report.md \
	  --from markdown-smart \
	  --template=ieee-template.tex \
	  --natbib \
	  -o report.tex

report.pdf: report.tex references.bib
	tectonic report.tex

clean:
	rm -f report.pdf report.tex *.aux *.bbl *.blg *.log
