#!/usr/bin/env python

import argparse, ast
import datetime as dt
import pandas
import numpy as np
from dateutil.parser import *


def arguments():

	# top-level parser
	parser = argparse.ArgumentParser()
	parser.add_argument('-a', dest='dates_a', 
		type=argparse.FileType('r'), 
		help='File in dates format (columns are <dateStart> <dateEnd> <value> <...>)')
	parser.add_argument('-o', dest='output', 
		type=argparse.FileType('w'), default="-", 
		help='Output file name')
	parser.add_argument('-F','--floor', type=str, default=None, 
		help='''Rounding interval. Accepted units: years, months, days, 
		hours, minutes, seconds [default: %(default)s]''')
	parser.add_argument('--before', 
		help="How much extend the time interval before [default: %(default)s]")
	parser.add_argument('--after', 
		help="How much extend the time interval after [default: %(default)s]")
	subparsers = parser.add_subparsers(metavar="List of utils")

	# create the parser for the 'intersect' command
	parser_intersect = subparsers.add_parser('intersect', 
		help='Intersect date intervals')
	parser_intersect.add_argument('-b', dest='dates_b',
		type=argparse.FileType('r'), 
		help='File in dates format (interval)')
	parser_intersect.set_defaults(func=intersect)
	
	# create the parser for the 'closest' command
	parser_closest = subparsers.add_parser('closest', 
		help='Find closest interval')
	parser_closest.add_argument('-b', dest='dates_b',
		type=argparse.FileType('r'), 
		help='File in dates format (interval)')
	parser_closest.add_argument('-d', dest='direction',
		choices=['d','u','b'], default='d', 
		help='''For each date in A find closest date in B. 
		Restrict to downstream dates only (d), upstream (u)
		or both (b)	[default=%(default)s]''')
	parser_closest.set_defaults(func=closest)
	
	# create the parser for the 'merge' command
	parser_merge = subparsers.add_parser('merge',
		help='Merge date intervals')
	parser_merge.add_argument('-T', dest='no_touching',
		default=False, action="store_true",
		help='''Do not merge if dates are touching 
			but not overlapping [default=Merge touching dates]''')
	parser_merge.set_defaults(func=merge)
	
	# create the parser for the 'format' command
	parser_format = subparsers.add_parser('format', 
		help='Format date intervals')
	parser_format.set_defaults(func=format)
	
	return parser
	

def floor_date(date, interval):
	''' Round date to the closest interval before '''
	units = ['years', 'months', 'days', 'hours', 
		'minutes', 'seconds']
	# Approximate to closest value
	def round(date, unit, n):
		n = int(n)
		return date.__getattribute__(unit) / n * n
	n, unit = interval.strip().split()
	# Supported units
	if unit not in units:
		print 'ERROR: unit not supported'
		exit()
	# remove plural
	unit = unit.rstrip('s')
	rounded = round(date, unit, n)
	date = eval('date.replace(%s = %s)' %(unit, rounded))
	return date
	

def extend(date, before=None, after=None):
	if before:
		before = before.strip().split()[1] + "=" + before.strip().split()[0]
		extended = eval('date - dt.timedelta(%s)' %before)
	if after:
		after = after.strip().split()[1] + "=" + after.strip().split()[0]
		extended = eval('date + dt.timedelta(%s)' %after)
	return extended


def extend_dates(df, before, after):
	if before:
		df[0] = extend(df[0], before=before)
	if after:
		df[1] = extend(df[1], after=after)
	return df


def read_dates(f, interval=None):
	#if interval:
	#	dateparse = lambda x: floor_date(parse(x), interval)
	#else:
	#	dateparse = lambda x: parse(x)	
	#df = pandas.read_csv(f, sep='\t', header=None, parse_dates=[0,1], date_parser=dateparse)
	df = pandas.read_csv(f, sep='\t', header=None, parse_dates=[0,1])
	if interval:
		df[0] = map(lambda x: floor_date(x, interval), df[0])
		df[1] = map(lambda x: floor_date(x, interval), df[1])
	return df


def read_dates_np(f, interval):
	df = np.genfromtxt(f, delimiter='\t', dtype=None, missing_values="NA")
	return df


def read_dates_a(args):
	# Read data with date intervals
	df_a = read_dates(args.dates_a, args.floor)
	if args.before or args.after:
		df_a = extend_dates(df_a, args.before, args.after)
	return df_a


def dates_overlap(dates1, dates2, touching=True):
	dates1_start, dates1_end = dates1
	dates2_start, dates2_end = dates2
	if touching:
		return dates1_start <= dates2_end and dates1_end >= dates2_start
	else:
		return dates1_start < dates2_end and dates1_end > dates2_start


def format_row(df_row):
	return "\t".join(map(str, df_row.values.tolist()[0]))


def closest_dates_by_group(df_a, df_b, direction, group_ix=3, max_dist=2.5*3600):
	''' For each date in A find the closest 
	date in B by the column specified in group '''
	# Sort dataframes
	df_a.sort_values([group_ix,0], 0, inplace=True)
	df_b.sort_values([group_ix,0], 0, inplace=True)
	# Initialize counter for rows in B
	j = 0
	df_a_nrows = df_a.count()[0]
	df_b_nrows = df_b.count()[0]
	# Traverse rows in A
	for i in xrange(df_a_nrows):
		b_row = df_b.iloc[[j]]
		a_row = df_a.iloc[[i]]
		b_group = b_row.iloc[0][group_ix]
		a_group = a_row.iloc[0][group_ix]
		# Check for group matching
		while b_group < a_group and j < df_b_nrows - 1:
			j += 1
			b_row = df_b.iloc[[j]]
			b_group = b_row.iloc[0][group_ix]
		if b_group > a_group:
			continue
		# Initialize variables for minimum
		min_diff = 1e+10
		min_b_row = None
		same_group = (a_group == b_group)
		while same_group:
			a_start = a_row.iloc[0][0]
			b_start = b_row.iloc[0][0]
			# Difference between date starts (consider other borders?)
			diff = abs((b_start - a_start).total_seconds())
			# Keep scanning B
			while a_start > b_start and j < df_b_nrows -1:
				if diff < min_diff and direction != "d":
					min_diff = diff
					min_b_row = b_row
				j += 1
				b_row = df_b.iloc[[j]]
				b_start = b_row.iloc[0][0]
			# Stop and print
			if a_start <= b_start:
				if diff < min_diff and direction != "u":
					min_diff = diff
					min_b_row = b_row
				if min_diff <= max_dist:
					# Exit the while loop
					same_group = False
					# Return both A and B rows
					yield format_row(a_row) + "\t" + format_row(b_row) + "\n"
				continue


def format_row_np(row):
	return "\t".join(map(str, row.tolist()))

def merge_dates(df_a, touching=True, group_ix=None):
	''' Merge date intervals '''
	if group_ix:
		df_a.sort_values([group_ix,0,1], 0, inplace=True)
	else:
		df_a.sort_values([0,1], 0, inplace=True)
	df_a_nrows = df_a.count()[0]
	a_row = df_a.iloc[[0]].values[0]
	for i in xrange(df_a_nrows):
		curr_row = df_a.iloc[[i]].values[0]
		if group_ix and a_row[group_ix] != curr_row[group_ix]:
			yield_row = format_row_np(a_row) + "\n"
			a_row = curr_row
			yield yield_row
		if dates_overlap(a_row[:2], curr_row[:2]):
			a_row[1] = curr_row[1]
			continue
		yield_row = format_row_np(a_row) + "\n"
		a_row = curr_row
		yield yield_row
	yield format_row_np(a_row) + "\n"
		

#def intersect_dates_by_group(df_a, df_b, group_ix=3):
#	''' Intersect dates when a group is specified '''
#	groups = set(df_a[group_ix].values.tolist())
#	df_a.sort_values([group_ix,0], 0, inplace=True)
#	df_b.sort_values([group_ix,0], 0, inplace=True)
#	return
#	j = 0
#	for i in range(df_b.count()[0]):
#		b_row = df_b.iloc[[i]]
#		a_row = df_a.iloc[[j]]
#		if b_row.iloc[0][group_ix] < a_row.iloc[0][group_ix]:
#			continue
#		while b_row.iloc[0][group_ix] > a_row.iloc[0][group_ix] and j < df_a.count()[0] - 1:
#			j += 1
#			a_row = df_a.iloc[[j]]
#		if b_row.iloc[0][group_ix] == a_row.iloc[0][group_ix]:
#			if b_row.iloc[0][1] < a_row.iloc[0][0]:
#				continue
#			while b_row.iloc[0][0] > a_row.iloc[0][1] and j < df_a.count()[0] - 1:
#				j += 1
#				a_row = df_a.iloc[[j]]
#			if dates_overlap(list(b_row.iloc[0][[0,1]]), list(a_row.iloc[0][[0,1]])):
#				yield format_row(a_row) + format_row(b_row) + "\n"

def intersect_dates(df_a, df_b):
	''' Intersect dates without specifying group '''
	df_a_nrows = df_a.count()[0]
	df_b_nrows = df_b.count()[0]
	j = 0
	for i in xrange(df_b_nrows):
		b_row = df_b.iloc[[i]].values[0]
		a_row = df_a.iloc[[j]].values[0]
		if b_row[1] < a_row[0]:
			continue
		while b_row[0] > a_row[1] and j < df_a_nrows - 1:
			j += 1
			a_row = df_a.iloc[[j]].values[0]
		if dates_overlap(b_row[:2], a_row[:2]):
			yield format_row_np(a_row) + "\t" + format_row_np(b_row) + "\n"


def intersect_dates_by_group(df_a, df_b, group_ix=3):
	''' Intersect dates when a group is specified '''
	groups = set(df_a[group_ix].values.tolist())
	for group in sorted(groups):
		print group
		subdf_a = df_a[df_a[group_ix] == group].sort_values([0], 0)
		subdf_b = df_b[df_b[group_ix] == group].sort_values([0], 0)
		for date in intersect_dates(subdf_a, subdf_b):
			yield date


def intersect(args):
	''' Intersect dates in b with dates in a '''
	# Read data with date intervals
	df_a = read_dates_a(args)
	df_b = read_dates(args.dates_b, args.floor)
	map(args.output.write, intersect_dates_by_group(df_a, df_b))
	return


def format(args):
	''' Print dates file after formatting dates '''
	df_a = read_dates_a(args)
	df_a.to_csv(args.output, sep='\t', header=False, index=False)
	return


def merge(args):
	''' Merged overlapping dates '''
	df_a = read_dates_a(args)
	map(args.output.write, merge_dates(df_a, not args.no_touching, group_ix=3))
	return


def closest(args):
	''' For each date in A, find the closest 
	feature in B '''
	df_a = read_dates_a(args)
	df_b = read_dates(args.dates_b, args.floor)
	map(args.output.write, closest_dates_by_group(df_a, df_b, 
		direction=args.direction))
	return



if __name__ == '__main__':
	# Argument parsing
	parser = arguments()
	args = parser.parse_args()
	# Call function
	args.func(args)
	exit()
