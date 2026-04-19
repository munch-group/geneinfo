#!/usr/bin/env bash

BLUE='\033[0;34m'
NC='\033[0m' # No Color

cd docs
DIR=pages

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

NBCONVERT_BGCOLOR='white'

if test -d $DIR; then
  for FILE in $(find $DIR -name '*.ipynb' | sort) ; do
    if [[  "$(grep $FILE _quarto.yml | grep -v '#')" ]]; then
      echo -e "${BLUE}Rendering ${FILE}${NC}"
      # CMD="jupyter nbconvert --Application.log_level=50 --to notebook --execute --inplace $FILE"
      CMD="jupyter nbconvert --log-level=WARN --to notebook --execute --inplace \
        --TagRemovePreprocessor.enabled=True \
        --TagRemovePreprocessor.remove_cell_tags='{"skip-execution"}' $FILE"
      set -x
    #  NBCONVERT=1 PYDEVD_DISABLE_FILE_VALIDATION=1 $CMD || exit 1 ;
     NBCONVERT_BGCOLOR="$NBCONVERT_BGCOLOR" PYDEVD_DISABLE_FILE_VALIDATION=1 $CMD > /dev/null 2>&1 && {
      set +x
      echo -e "${GREEN}Successfully executed ${FILE}${NC}"
     } || {
        set +x
        echo -e "${RED}Error executing ${FILE}${NC}"
        errors=true
      }
      echo
    fi
  done
else
  echo "directory $DIR does not exist"
  exit 1
fi

for FILE in $(find $DIR -type f -size +50M); do
  echo "$FILE too large. Clearing outputs."
  jupyter nbconvert --clear-output --inplace $FILE
done

# if [ "$errors" = true ] ; then
#   echo "One or more notebooks failed to execute"
#   exit 1
# fi