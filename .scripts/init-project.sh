
FILES="src docs"

y_flag='false'
verbose='false'

while getopts 'yv' flag; do
  case "${flag}" in
    y) y_flag='true' ;;
    v) verbose='true' ;;
    *) echo "Usage: $0 [-y] [-v]"
       exit 1 ;;
  esac
done

if [ "$verbose" = true ] ; then
  set -x
fi

if [ "$y_flag" = true ] ; then
  git rm -r $FILES 2>/dev/null || echo "Some files didn't exist, continuing..."
else
  git rm --dry-run -r $FILES 2>/dev/null || echo "Some files don't exist"
  read -p "Continue (y/n)?" choice
  case "$choice" in 
    y|Y ) git rm -r $FILES 2>/dev/null || echo "Some files didn't exist, continuing...";;
    n|N ) echo "Aborted"; exit 0;;
    * ) echo "invalid"; exit 1;;
  esac
fi
