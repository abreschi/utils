#!/usr/bin/env bash

# Select rows of a matrix based on a list of rownames
# If you want to replace the existing rownames 
# the selection of rownames can be passed as standard input


if [ $# == 0 ]; then
echo "USAGE: $0 <file.txt> <file.tsv> [file.txt]"
exit 1
fi

rownames="$1"
matrix="$2"
newrownames="$3"


#if [[ $newrownames != "" ]]; then
#if [[ $(wc -l < $newrownames) != $(wc -l < $rownames) ]]; then
#printf "\nERROR: the length of the new rownames is different from the length of the old ones!\n\n"
#exit 1
#fi
#fi

awk -v file="$matrix" -v file2="$newrownames" '\
BEGIN{FS="\t"; OFS="\t"; 
while(getline<file>0){i++; if(i==1){print}else{gene=$1;$1="place_holder";array[gene]=$0}}
if (file2 != "") {while(getline<file2>0){j++; newRownames[j]=$1}}
}
{
if (!($1 in array)) {print "Warning: "$1" not found in matrix!" | "cat 1>&2"; next}
if (file2=="") {print $1, array[$1]} else {print newRownames[NR], array[$1]}
}END{
# Check this error
#if(j!=NR){print "ERROR" | "cat 1>&2"}
}' "$rownames" | sed 's/place_holder\t//g'
