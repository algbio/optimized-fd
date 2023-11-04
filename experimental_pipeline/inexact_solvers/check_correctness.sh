awk 'FNR==NR {a[$3]=$1; next} $3 in a {if (a[$3] != $1) print a[$3], $1, $3}' $1 $2
