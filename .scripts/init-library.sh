
FILES="index.qmd custom.scss global_params.* workflow.py _quarto.yml references.* data_table.csv input_file1.txt input_file2.txt filters thesis manuscript slides"

y_flag=false
verbose=false

while getopts 'yv' flag; do
  case "${flag}" in
    y) y_flag=true ;;
    v) verbose=true ;;
    *) echo "Usage: $0 [-y] [-v]"
       exit 1 ;;
  esac
done

if [ "$verbose" = true ] ; then
  set -x
fi

if [ "$y_flag" = true ] ; then
  for FILE in $FILES; do
    if [ -e $FILE ]; then
      git rm -r $FILE 2>/dev/null
    fi
  done
else
  for FILE in $FILES; do
    if [ -e $FILE ]; then
      echo "remove $FILE?"
      read -p "Continue (y/n)?" choice
      case "$choice" in 
        y|Y ) git rm -r $FILE 2>/dev/null ;;
        n|N ) echo "Aborted"; exit 0;;
        * ) echo "invalid"; exit 1;;
      esac
    fi
  done
fi
