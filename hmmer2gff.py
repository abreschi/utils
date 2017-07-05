#!/usr/bin/env python

import argparse, sys

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("-i", "--input", type=str,
	help="HMMER tabular output for domains (--domtblout)")
parser.add_argument("-O", "--output-type", dest="output_type", type=str, default="gff",
	help="<gff> <gff3> [default=%(default)s]")
args = parser.parse_args()


def read_hmmer(f, output_type):
	openf = sys.stdin if f == "stdin" else open(f)
	for line in openf:
		if line.startswith("#"):
			continue
		lsp = line.split()
		chr = lsp[0]
		gn = lsp[3]
		acc = lsp[4]
		start = lsp[17]
		end = lsp[18]

		# GFF3 output
		if output_type == "gff3":
			gfftags = (
				"Name=%s;" %(gn),
				"ID=%s;" %(acc),
			)
			gfftags = "".join(gfftags)
		# GFF output
		if output_type == "gff":
			gfftags = (
				'gene_id "%s";' %(gn),
				'transcript_id "%s";' %(gn),
			)
			gfftags = " ".join(gfftags)
		gffl = (
			chr,
			"HMMER",
			"DOMAIN",
			start,
			end,
			".",
			".",
			".",
			gfftags,
		)
		print "\t".join(gffl)
	return

if __name__ == "__main__":
	read_hmmer(args.input, args.output_type)

