#!/usr/bin/env Rscript

cat("Calculate pearson correlation coefficient between two vectors of values passed as columns in standard input\n\nNOTE: columns must have no header\n\n")


#showConnections()
m = read.table(file("stdin"),h=F)

#head(m)
cc = cor(m[1], m[2], use='p', method='s')

cat("Spearman cc = ", cc, "\n\n")

q(save='no')
