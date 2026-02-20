#!/usr/bin/env bash

BLUE='\033[0;34m'
NC='\033[0m' # No Color

DIR=./docs/pages

if test -d $DIR; then
  for FILE in $(find $DIR -name '*.ipynb') ; do
    echo -e "${BLUE}Rendering ${FILE}${NC}"
    # CMD="jupyter nbconvert --Application.log_level=50 --to notebook --execute --inplace $FILE"
    CMD="jupyter nbconvert --log-level=WARN --to notebook --execute --inplace \
      --TagRemovePreprocessor.enabled=True \
      --TagRemovePreprocessor.remove_cell_tags='{"skip-execution"}' $FILE"
    echo $CMD
    NOTEBOOK_THEME=light PYDEVD_DISABLE_FILE_VALIDATION=1 $CMD || exit 1 ;
  done
else
  echo "docs/pages directory does not exist"
  exit 1
fi

for FILE in $(find $DIR -type f -size +50M); do
  echo "$FILE too large. Clearing outputs."
  jupyter nbconvert --clear-output --inplace $FILE
done
