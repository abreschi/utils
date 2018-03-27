#!/usr/bin/env Rscript


suppressPackageStartupMessages(library("optparse"))

option_list <- list(
	make_option(c("--header"), action="store_true", default=FALSE,
		help="Use this if the input has a header [default=%default]")
)

parser <- OptionParser(
        usage = "%prog [options] file", 
        option_list=option_list,
        description = "Reads the values on the first column and outputs a histogram"
)
arguments <- parse_args(parser, positional_arguments = TRUE)
opt <- arguments$options



m = read.table(file("stdin"),h=opt$header, sep="\t", quote="", comment.char="")

#head(m)
print(summary(m))

q(save='no')
