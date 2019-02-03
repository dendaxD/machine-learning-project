all: lampasova-project.pdf

lampasova-project.pdf: lampasova-project.tex *.tex
	pdflatex lampasova-project
	pdflatex lampasova-project

clean:
	@rm -f *.{log,aux,pdf,toc,bbl,lof,blg,out}

cleanaux:
	@rm -f *.{log,aux,toc,bbl,lof,blg,out}