#!/usr/bin/env python

import argparse, ast, re
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
	parser.add_argument('-g', dest='group_by', type=int, default=None, 
		help='''Perform operation by group. Specify column index 
		with the group factor [default=%(default)s]''')
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
	parser_merge.add_argument('-m', dest='max_dist',
		help='''Merge intervals not overlapping yet
		closer than this distance. Format is "<duration> <unit>",
		eg "10 minutes" [default: %(default)s]''')
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


def dates_overlap(dates1, dates2, touching=True, dist=0):
	dates1_start, dates1_end = dates1
	dates2_start, dates2_end = dates2
	# add distance to dates1_end
	dates1_end = dates1_end + dt.timedelta(0, dist)
	if touching:
		return dates1_start <= dates2_end and dates1_end >= dates2_start
	else:
		return dates1_start < dates2_end and dates1_end > dates2_start


def format_row(df_row):
	return "\t".join(map(str, df_row.values.tolist()[0]))


def closest_dates(df_b, df_a, direction, tol):
	"""
	http://code.activestate.com/recipes/335390-closest-elements-in-a-target-array-for-a-given-inp/
	Find the set of elements in input_array that are closest to
	elements in target_array.  Record the indices of the elements in
	target_array that are within tolerance, tol, of their closest
	match. Also record the indices of the elements in target_array
	that are outside tolerance, tol, of their match.

	For example, given an array of observations with irregular
	observation times along with an array of times of interest, this
	routine can be used to find those observations that are closest to
	the times of interest that are within a given time tolerance.

	NOTE: input_array must be sorted! The array, target_array, does not have to be sorted.

	Inputs:
	  input_array:  a sorted Float64 numarray
	  target_array: a Float64 numarray
	  tol:		  a tolerance

	Returns:
	  closest_indices:  the array of indices of elements in input_array that are closest to elements in target_array
	  accept_indices:  the indices of elements in target_array that have a match in input_array within tolerance
	  reject_indices:  the indices of elements in target_array that do not have a match in input_array within tolerance
	"""

	df_b = df_b.sort_values([0], 0).as_matrix()
	df_a = df_a.as_matrix()

	# Extract date columns
	input_array = df_a[:, 0]
	target_array = df_b[:, 0]
	# 
	input_array_len = len(input_array)
	acc_rej_indices = [-1] * len(target_array)
	curr_tol = [tol] * len(target_array)
	# Determine the locations of target_array in input_array
	closest_indices = np.searchsorted(input_array, target_array)

	est_tol = 0.0
	for i in xrange(len(target_array)):
		best_off = 0		  # used to adjust closest_indices[i] for best approximating element in input_array
		closest_index = closest_indices[i]

		if closest_index >= input_array_len:
			# the value target_array[i] is >= all elements 
			# in input_array so check whether it is within 
			# tolerance of the last element
			closest_indices[i] = input_array_len - 1
			closest_index = closest_indices[i]
			est_tol = (target_array[i] - input_array[closest_index]).total_seconds()
			if est_tol < curr_tol[i] and direction != 'd':
				curr_tol[i] = est_tol
				acc_rej_indices[i] = i
		elif target_array[i] == input_array[closest_index]:
			# target_array[i] is in input_array
			est_tol = 0.0
			curr_tol[i] = 0.0
			acc_rej_indices[i] = i
		elif closest_index == 0 and direction != 'u':
			# target_array[i] is <= all elements in input_array
			est_tol = (input_array[0] - target_array[i]).total_seconds()
			if est_tol < curr_tol[i]:
				curr_tol[i] = est_tol
				acc_rej_indices[i] = i
		else:
			# target_array[i] is between input_array[closest_indices[i]-1] and input_array[closest_indices[i]]
			# and closest_indices[i] must be > 0
			top_tol = (input_array[closest_index] - target_array[i]).total_seconds()
			bot_tol = (target_array[i] - input_array[closest_index-1]).total_seconds()
#			print target_array[i], input_array[closest_index], input_array[closest_index-1]
			if direction == 'u':
				est_tol = bot_tol
				best_off = -1
			elif direction == 'd':
				est_tol = top_tol
			else: # if both directions are allowed
				if bot_tol <= top_tol:
					est_tol = bot_tol
					best_off = -1		   # this is the only place where best_off != 0
				else:
					est_tol = top_tol

			if est_tol < curr_tol[i]:
				curr_tol[i] = est_tol
				acc_rej_indices[i] = i

		if est_tol <= tol:
			closest_indices[i] += best_off

	curr_tol = np.asarray(curr_tol)
	accept_indices = np.compress(np.greater(acc_rej_indices, -1), acc_rej_indices)
	reject_indices = np.compress(np.equal(acc_rej_indices, -1), np.arange(len(acc_rej_indices)))
	accept_df =  np.hstack((df_b[accept_indices], 
		df_a[closest_indices[accept_indices]]))
	accept_df = np.concatenate( (accept_df, curr_tol[accept_indices, np.newaxis] ), 1)
	reject_df = np.hstack((df_b[reject_indices], 
		np.full((len(reject_indices), df_a.shape[1]), np.nan))) 
	reject_df = np.concatenate( (reject_df, curr_tol[reject_indices, np.newaxis] ), 1)
	out = np.vstack((accept_df, reject_df))
#	return (closest_indices, accept_indices, reject_indices)
	return out


def closest_dates_by_group(df_a, df_b, direction, 
		group_ix, max_dist=float('inf')):
	''' For each date in A find the closest 
	date in B by the column specified in group '''
	try:
		groups = set(df_a[group_ix].values)
	except:
		print "Incorrect column index for group factor"
		exit() 
	for group in sorted(groups):
	# Sort dataframes
		subdf_a = df_a[df_a[group_ix] == group]
		subdf_b = df_b[df_b[group_ix] == group]
		if len(subdf_a) == 0 or len(subdf_b) == 0:
			continue
		subdf_out = closest_dates(subdf_a, subdf_b, 
			direction=direction, tol=max_dist)
		try:
			np.hstack((out, subdf_out))
		except:
		#except UnboundLocalError:
			out = subdf_out
	return out


def format_row_np(row):
	return "\t".join(map(str, row.tolist()))

def merge_dates(df_a, touching=True, group_ix=None, dist=None):
	''' Merge date intervals '''
	# Parse distance if provided
	dist = parse_time_interval(dist) if dist else 0
	# Check column group
	if group_ix:
		df_a.sort_values([group_ix,0,1], 0, inplace=True)
	else:
		df_a.sort_values([0,1], 0, inplace=True)
	df_a = df_a.as_matrix()
	df_a_nrows, df_a_ncols = df_a.shape
	a_row = df_a[0,]
	for i in xrange(df_a_nrows):
		curr_row = df_a[i,]
		if group_ix and a_row[group_ix] != curr_row[group_ix]:
			yield_row = format_row_np(a_row) + "\n"
			a_row = curr_row
			yield yield_row
		if dates_overlap(a_row[:2], curr_row[:2], dist=dist):
			a_row[1] = max(curr_row[1], a_row[1])
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
	groups = set(df_a[group_ix].values)
	for group in sorted(groups):
		subdf_a = df_a[df_a[group_ix] == group].sort_values([0], 0)
		subdf_b = df_b[df_b[group_ix] == group].sort_values([0], 0)
		for date in intersect_dates(subdf_a, subdf_b):
			yield date



def parse_time_interval(t):
	t = t.strip()
	count, unit = re.match('(\d+\.?\d*)\ ?([aA-zZ]*)', t).group(1,2)
	if unit not in ["years", "months", "weeks", 
		"days", "hours", "minutes", "seconds"]:
		print "ERROR: Time unit not supported"
		exit(1)
	sec = dt.timedelta(**{unit: float(count)}).seconds
	return sec


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
	map(args.output.write, merge_dates(df_a, 
		not args.no_touching, group_ix = args.group_by, 
		dist = args.max_dist)
	)
	return


def closest(args):
	''' For each date in A, find the closest 
	feature in B '''
	df_a = read_dates_a(args)
	df_b = read_dates(args.dates_b, args.floor)
	max_dist = float('inf')
	if args.max_dist:
		max_dist = parse_time_interval(args.max_dist) # seconds
	if args.group_by:
		out = closest_dates_by_group( df_a, df_b, 
			direction=args.direction, 
			group_ix=args.group_by,
			max_dist=max_dist,
			)
	else:
		out = closest_dates(df_a, df_b, 
			direction = args.direction, 
			tol=max_dist,
			)
	np.savetxt(args.output, out, fmt='%s', delimiter='\t')
	return



if __name__ == '__main__':
	# Argument parsing
	parser = arguments()
	args = parser.parse_args()
	# Call function
	args.func(args)
	exit()