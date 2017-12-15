#!/usr/bin/env python

import argparse, ast
import datetime as dt
import pandas
from dateutil.parser import *


def arguments():

	# top-level parser
	parser = argparse.ArgumentParser()
	parser.add_argument('-a', dest='dates_a', 
		type=argparse.FileType('r'), 
		help='File in dates format (interval)')
	parser.add_argument('-o', dest='output', 
		type=argparse.FileType('w'), 
		help='Output file name')
	parser.add_argument('-F','--floor', type=str, default="5 minutes", 
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
	

def read_dates(f, interval):
	dateparse = lambda x: floor_date(parse(x), interval)
	df = pandas.read_csv(f, sep='\t', header=None, parse_dates=[0,1], date_parser=dateparse)
	return df


def read_dates_a(args):
	# Read data with date intervals
	df_a = read_dates(args.dates_a, args.floor)
	if args.before or args.after:
		df_a = extend_dates(df_a, args.before, args.after)
	return df_a


def intersect(args):
	# Read data with date intervals
	df_a = read_dates_a(args.dates_a, args.floor)
	df_b = read_dates(args.dates_b, args.floor)
	if args.before or args.after:
		df_a = extend_dates(df_a, args.before, args.after)
	return


def format(args):
	df_a = read_dates_a(args)
	df_a.to_csv(args.output, sep='\t', header=False, index=False)
	return

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


if __name__ == '__main__':
	# Argument parsing
	parser = arguments()
	args = parser.parse_args()
	print args
	args.func(args)
	exit()
