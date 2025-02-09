#!/usr/bin/env bash

BLUE='\033[0;34m'
NC='\033[0m' # No Color

for FILE in docs/pages/*.ipynb ; do
    echo -e "${BLUE}Rendering ${FILE}${NC}"
    CMD="jupyter nbconvert --Application.log_level=50 --to notebook --execute --inplace $FILE"
    echo $CMD
    PYDEVD_DISABLE_FILE_VALIDATION=1 $CMD || exit 1 ;
 done