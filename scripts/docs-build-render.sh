#!/usr/bin/env bash

pip install --quiet --no-deps --force-reinstall -e . \
    && cd docs \
    && rm -f api/_styles-quartodoc.css api/_sidebar.yml *.qmd \
    && quartodoc build && quartodoc interlinks && quarto render \
    && cd .. \
    && pip uninstall --quiet -y munch-group-library
