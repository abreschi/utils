#!/usr/bin/env python

import argparse, sys

decor_dict = {
	"keep" : ">>>>>",
	"exact" : "]]=[[",
	"one" : "]]=|[",
	"two" : "]|=[[",
}


def options():
	parser = argparse.ArgumentParser()
	parser = argparse.ArgumentParser(description='Add exon boundaries to MSA in fasta format')
	parser.add_argument("-i", "--input", type=str,
		help="Alignment in fasta format")
	parser.add_argument("-b", "--bed12", type=str,
		help="bed12 file with CDS. THe 4th column corresponds to the id in the alignment (e.g. >ID description)")
	parser.add_argument("-O", "--output-type", dest="output_type", type=str, default="gff",
		help="<gff> <gff3> [default=%(default)s]")
	args = parser.parse_args()
	return args


def read_bed12(bed12):
	ref = {}
	for line in open(bed12, 'r'):
#		chr, start, end, id, score, strand, Tstart, Tend, itemRgb, blockCount, blockSizes, blockStarts 
		line_el = line.strip().split("\t")
		chr, start, end, id, score, strand = line_el[0:6]
		Tstart, Tend = line_el[6:8]
		# No CDS in the transcript
		if Tstart == Tend:
			continue
		start, end, Tstart, Tend = map(int, (start, end, Tstart, Tend))
		blockCount, blockSizes, blockStarts = line_el[9:12]
		blockSizes = map(int, blockSizes.strip(",").split(","))
		blockStarts = map(int, blockStarts.strip(",").split(","))
		for i in range(int(blockCount)):
			blockStart = start + blockStarts[i]
			blockEnd = start + blockStarts[i] + blockSizes[i]
			if blockStart <= Tend and blockEnd >= Tstart:
				CDS = [max(Tstart, blockStart), min(Tend, blockEnd)]
				ref.setdefault(id, []).append((CDS[0], CDS[1], strand))
	# Return a dict with key: transcript_id; value: list of CDSs
	return ref


def decorate_seq(seqs, ref, decor_dict):
	# Initialize dictionaries with indeces
	s_indeces = dict.fromkeys(seqs.keys(), 0)
	new_seqs = dict.fromkeys(seqs.keys(), "")
	exon_indeces = dict.fromkeys(seqs.keys(), 0)
	tx_lens = {}
	for id in seqs:
		exons = sorted(ref[id])
		ref[id] = exons
		exon_i = exon_indeces[id]
		tx_lens[id] = exons[exon_i][1] - exons[exon_i][0]

	align_len = len(seqs.values()[0])
	# Iterate over alignment length
	for i in range(align_len):
		decor = {}
		# Iterate over aligned sequences
		for id in seqs:
			seq = seqs[id]
			s = seq[i]
			tx_len = tx_lens[id]
			if s != "-":
				s_indeces[id] += 1
				s_index = s_indeces[id]
				# Check if the aa is overlapping a splice site
				if s_index * 3 < tx_len:
					decor[id] = "keep"    # no overlap with splice site
					continue
				elif s_index * 3 == tx_len:
					decor[id] = "exact"   # exact overlap with splice site
				elif s_index * 3 == tx_len +1:
					decor[id] = "one"     # overlap with splice site +1
				elif s_index * 3 == tx_len +2:
					decor[id] = "two"     # overlap with splice site +2
				exon_indeces[id] += 1
				exons = ref[id]
				exon_i = exon_indeces[id]
				if exon_i < len(exons):
					tx_lens[id] += exons[exon_i][1] - exons[exon_i][0]
			if s == "-":                  # alignment gap
				decor[id] = "keep"

		# Build the new sequence with splice site annotation
		add_decor = any(map(lambda x: x != "keep", decor.values()))
		exon_decor = ""
		for id in seqs:
			if add_decor:
				exon_decor = decor_dict[decor[id]]
			new_seqs[id] += seqs[id][i] + exon_decor
	return new_seqs


def read_fa(f, ref):
	openf = sys.stdin if f == "stdin" else open(f)
	seqs = dict()
	for line in openf:
		if line.startswith(">"):
			id = line.lstrip(">").split()[0].strip()
			seqs[id] = ""
			continue
		seqs[id] += line.strip()
	return seqs


def print_seqs(seqs):
	for id, seq in seqs.iteritems():
		print ">%s" %(id)
		print seq
	return


if __name__ == "__main__":

	args = options()

	ref = read_bed12(args.bed12)

	seqs = read_fa(args.input, ref)

	new_seqs = decorate_seq(seqs, ref, decor_dict)
	print_seqs(new_seqs)
