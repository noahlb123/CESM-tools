import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import scipy
import tools

#read index file
t = tools.ToolBox()
p = pd.read_csv('data/standardized-ice-cores/index.csv')
p = p.reset_index()

#read each ice core file
exclude = set(['mcconnell-2017-1.csv', 'brugger-2021-1.csv'])
windows = [1, 3, 5, 11]
prei = {}
pred = {}
pi_indexes = set([])
pd_indexes = set([])
ya = {}
for key in windows:
    ya[key] = []
x = []
annotations = []

def add_pi_pd(a1, y1, a2, y2):
    def helper(a, y):
        x.append(y)
        for key in windows:
            ya[key].append(a[key])
        annotations.append(filename)
    def valid_yr(y):
        return y != None and abs(y - 1850.5) < 30
    if (valid_yr(y1)):
        #pi
        helper(a1, y1)
        pi_indexes.add(len(x) - 1)
        prei[filename] = a1
        #pd
        helper(a2, y2)
        pd_indexes.add(len(x) - 1)
        pred[filename] = a2

for index, row in p.iterrows():
    for i in range(row['n_cores']):
        filename = row['First Author'].lower() + '-' + str(row['Year']) + '-' + str(i + 1) + '.csv'
        if (filename in exclude):
            continue
        d = pd.read_csv('data/standardized-ice-cores/' + filename)
        #must be flipped bec they are in decending order
        BC = np.flip(d['BC'].to_numpy())
        Yr = np.flip(d['Yr'].to_numpy())
        a1, y1 = t.get_avgs(Yr, BC, 1850.5, windows)
        a2, y2 = t.get_avgs(Yr, BC, 9999, windows)
        add_pi_pd(a1, y1, a2, y2)
        

#setup pi and pd colors
colors = []
c = ''
for i in range(len(x)):
    if i in pd_indexes:
        c = "#4d4564"
    elif i in pi_indexes:
        c = "#da6032"
    else:
        c = "#FF0000"
    colors.append(c)

#plot
for target_w in windows:
    y = ya[target_w]
    #calculate ratio
    s = 0
    n = 0
    for key in pred.keys():
        s += pred[key][target_w] / prei[key][target_w]
        n += 1
    fig,ax = plt.subplots()
    sc = plt.scatter(x, y, s=5, c=colors)
    coef = np.polyfit(x,y,1)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    poly1d_fn = np.poly1d(coef)
    main_line = plt.plot(x, poly1d_fn(x), '-', c="#000000", label='m='+str(round(slope, 4)))
    plt.title("PD/PI_BC= " + str(round(s/n, 4)) + ", window=" + str(target_w) + ", n=" + str(len(x)/2))
    plt.xlabel("Year (CE)")
    plt.ylabel("BC (ng/g)")
    legend_elements = [
        Line2D([0], [0], marker='o', color="#da6032", label='PI', markersize=5),
        Line2D([0], [0], marker='o', color="#4d4564", label='PD', markersize=5),
        Line2D([0], [0], color="#000000", label='m='+str(round(slope, 4))),
        ]

    t.matplot_tooltips(ax, fig, sc, annotations)
    #ax.legend(handles=legend_elements, loc='upper left')
    plt.legend(handles=legend_elements)
    #plt.savefig("figures/ice-cores/" + str(target_w))
    plt.show()
    plt.close()