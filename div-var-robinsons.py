from netCDF4 import Dataset
import sys
import os

if len(sys.argv) < 2:
    #python div-var-robinsons.py 2
    raise Exception('1 command line arguments required: <steps: (a/1/2)>')
step = sys.argv[1]
root = '/glade/derecho/scratch/nlbills/all-ice-core-data'
dirs = ['loadbc', 'mmrbc', 'sootsn', 'wet-dry']

if step == '1' or step == 'a': #combine nc files
    for dir in dirs:
        p_i = os.path.join(root, dir, 'CESM2_pi.nc')
        p_d = os.path.join(root, dir, 'CESM2_pd.nc')
        if os.path.isfile(p_i) and os.path.isfile(p_d):
            print(dir, ': good')
        else:
            print(dir, ': bad')
if step == '2' or step == 'a': #plot
    pass