.PHONY: help clean html dirhtml

help:
	@echo "Please use 'make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  clean      to remove build artifacts"

clean:
	rm -rf _build

html: clean
	sphinx-build -b html . _build/html

dirhtml: clean
	sphinx-build -b dirhtml . _build/dirhtml

.DEFAULT_GOAL := help
