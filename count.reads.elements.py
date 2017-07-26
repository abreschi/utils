#!/usr/bin/env python

import argparse, sys
import subprocess as sp

parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser(description='Quantify elements from bam file')
parser.add_argument("--abam", type=str, help="Bam file")
parser.add_argument("-b", type=str, help="Annotation file with elements to quantify (.gtf)")
parser.add_argument("-o", "--output", type=str, default="stdout", help="output file name. [default=stdout]")
parser.add_argument("-u", "--umi", action="store_true", help="Count UMIs instead of reads [default=%(default)]")
args = parser.parse_args()

c = dict()
# read all gene ids from gtf
for line in open(args.b,"r"):
	cols = line.strip().strip(";").split("\t")
	tags = cols[9-1].split("; ")
	d = dict((tag.split(" ")[0],tag.split(" ")[1].strip("\"")) for tag in tags)
	if args.umi:
		c.setdefault(d["gene_id"], set())
		continue
	c.setdefault(d["gene_id"], 0)

p1 = sp.Popen("intersectBed -abam %s -b %s -f 1 -s -wb -bed" %(args.abam,args.b), stdout=sp.PIPE, shell=True)
#p2 = sp.Popen("head", stdin=p1.stdout, stdout=sp.PIPE)
#p1.stdout.close()
#for line in p2.communicate()[0]:
for line in p1.stdout:
	cols = line.strip().strip(";").split("\t")
	tags = cols[12+9-1].split("; ")
	d = dict((tag.split(" ")[0],tag.split(" ")[1].strip("\"")) for tag in tags)
	gene_id = d["gene_id"]
	sep = ":"
	if args.umi:
		u = cols[4-1].strip().split("/")[0].split(":")[-1]
		c[gene_id].add(u)
		continue
	c[gene_id] = c[gene_id] + 1

p1.stdout.close()

outF = sys.stdout if args.output == "stdout" else open(args.output,"w")
for k,v in c.iteritems():
	count = v
	if args.umi:
		count = len(v)
	outF.write("%s\t%s\n" %(k,count))
outF.close()

exit()
