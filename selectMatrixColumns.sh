#!/usr/bin/env bash

if [[ $# != 2 ]]; then
	echo "USAGE: $0 selected_columns matrix.file.tsv"
	echo "The selected_columns are \":\" separated"
	exit 1
fi

f="$1"
m="$2"

awk -v f="$f" '
BEGIN {
	FS=OFS="\t";
}

NR == 1 {
	n = NF
	for (i=1;i<=NF;i++) {
		header[$(i)] = i	
	}
	split(f, a, ":");
	c = 1
	for (i=1;i<=length(a);i++) {
		if (a[i] in header) {
			sel[a[i]] = c++
		}
	}
	for (k=1;k<=n;k++) {
		if ($(k) in sel) {
			indices[sel[$(k)]] = k
 		}
	}
}
{
	offset = (n == NF ? 0 : 1)
	split($0, line, FS)
	for(j in indices) {
		$(j+offset) = line[indices[j]+offset]
	} 
	NF=length(sel)+offset
}
1
' "$m"


