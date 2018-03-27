#!/usr/bin/env Rscript

cat("Calculate pearson correlation coefficient between two vectors of values passed as columns in standard input\n\nNOTE: columns must have no header\n\n", file=stderr())


#showConnections()
m = read.table(file("stdin"),h=F)


c = cor.test(m[,1], m[,2], use='p', method='p')

cat(sprintf("%s\t%s\n", c$estimate, c$p.value))

q(save='no')
