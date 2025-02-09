#!/usr/bin/env bash

cd docs
rm -f api/_styles-quartodoc.css api/_sidebar.yml *.qmd
quartodoc build && quartodoc interlinks && quarto render
cd ..
