#!/soft/bin/python

from optparse import OptionParser
import itertools
from scipy.stats import wilcoxon

parser = OptionParser()
parser.add_option('-X','--X',dest='X',help='first distribution no header, <tag> <value>',metavar='file.tab')
parser.add_option('-Y','--Y',dest='Y',help='second distribution no header, <tag> <value>',metavar='file.tab')
options,args = parser.parse_args()

pv05 = [0,2,4,6,8,11,14,17,21,25,30,35,40,46,52,59,66,73,81,89]
pv02 = ['NA',0,2,3,5,7,10,13,16,20,24,28,33,38,43,49,56,62,69,77]
pv01 = ['NA','NA',0,2,3,5,7,10,13,16,20,23,28,32,38,43,49,55,61,68]
N = range(6,26)

def read_experiment(fname):
	return dict(line.strip().split('\t') for line in open(fname))

dict1 = read_experiment(options.X)
dict2 = read_experiment(options.Y)

experiments = dict((k,(float(dict1.get(k,0)),float(dict2.get(k,0)))) for k in set(itertools.chain(dict1.iterkeys(),dict2.iterkeys())))
n = len(experiments)

diff = [(x-y) for x,y in experiments.itervalues()]
signs = [(x-y)/abs(x-y) for x,y in experiments.itervalues()]

W,approx_pv = wilcoxon(diff)

w_05 = pv05[N.index(n)]
w_02 = pv02[N.index(n)]
w_01 = pv01[N.index(n)]

print 'Wilcoxon=', W

if w_01 > W and w_01 != 'NA':
	print w_01, '0.01%'
else:
	if w_02 > W and w_02 != 'NA':
		print w_02, '0.02%'
	else:
		if w_05 > W:	
			print w_05, '0.05%'
		else:
			print w_05, 'samples not significantly different'
