@@ -0,0 +1,27 @@
awk '
BEGIN {
    found_difference = 0
}
FNR==NR {

    if (match($3, /datasets\/.*/)) {
        key = substr($3, RSTART, RLENGTH)
        a[key] = $1 
    }
    next
}
{
    if (match($3, /datasets\/.*/)) {
        key = substr($3, RSTART, RLENGTH)

        if (key in a && a[key] != $1) {
            print a[key], $1, key >> "dif.txt"
            found_difference = 1  
        }
    }
}
END {
    if (found_difference == 0) {
        print "nothing is different"
    }
}' $1 $2