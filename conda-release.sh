#!/usr/bin/env bash

gh release create --latest "v$(python setup.py --version)" --title "v$(python setup.py --version)" --notes ""