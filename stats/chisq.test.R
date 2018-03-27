#!/usr/bin/env Rscript

cat("USAGE: script.R setAB setA setB total\n")

args = commandArgs(TRUE)
args = as.list(as.double(args))

# Given the contingency matrix
#         _
#   | A | A
# --|---------
# B | a | b
# _ |---------
# B | c | d
#   

a=args[[1]]
b=args[[2]]-a
c=args[[3]]-a
d=args[[4]]-a-b-c

M = matrix(c(a,b,c,d), byrow=T, nrow=2)

print(M)

print(chisq.test(M)$expected)

cat("\np.value=",chisq.test(M)$p.value,"\n")
