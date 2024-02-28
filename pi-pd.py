import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import cartopy
import scipy
import tools
import math

#read index file
t = tools.ToolBox()
p = pd.read_csv('data/standardized-ice-cores/index.csv')
p = p.reset_index()

#setup vars
exclude = set(['mcconnell-2017-1.csv', 'brugger-2021-1.csv'])
windows = [1, 3, 5, 11]
full_data = {}
for key in windows:
    full_data[key] = [] #filename, lat, lon, PI averge, PD averge, ratio, PI year, PD year, 3 averege, 3 year
for_cartopy = {}

#read each ice core file
for index, row in p.iterrows():
    for i in range(row['n_cores']):
        filename = row['First Author'].lower() + '-' + str(row['Year']) + '-' + str(i + 1) + '.csv'
        lat = row['N']
        lon = row['E']
        if (filename in exclude):
            continue
        d = pd.read_csv('data/standardized-ice-cores/' + filename)
        #must be flipped bec they are in decending order
        BC = np.flip(d['BC'].to_numpy())
        Yr = np.flip(d['Yr'].to_numpy())
        a1, y1 = t.get_avgs(Yr, BC, 1850.49, windows)
        a2, y2 = t.get_avgs(Yr, BC, 9999, windows)
        a3, y3 = t.get_avgs(Yr, BC, 1980, windows)
        #add data to datasets
        if (y1 != None and y2 != None and abs(y1 - y2) >= 100 and y1 < 1900):
            for key in windows:
                full_data[key].append([filename, lat, lon, a1[key], a2[key], a2[key]/a1[key], y1, y2, a3[key], y3])
            #cartopy
            for_cartopy[filename] = {'lat': lat, 'lon': lon, 'ratio': a3[5] / a1[5]}

#plot
inp = input("Matplot or Cartopy? (m/c): ")
def format_column(c):
    return np.transpose(c).astype('float64').tolist()[0]
if (inp == "m"):#Raw Matplot
    for target_w in windows:
        #setup data
        data = np.matrix(full_data[target_w])
        filenames = np.transpose(data[:,0]).tolist()[0]
        lat = data[:,1]
        lon = data[:,2]
        pia = data[:,3]
        pda = data[:,4]
        ratios = format_column(data[:,5])
        piy = data[:,6]
        pdy = data[:,7]
        a3 = data[:,8]
        y3 = data[:,9]
        y = format_column(np.concatenate((pia, pda)))
        x = format_column(np.concatenate((piy, pdy)))
        colors = ["#da6032"] * len(data) + ["#4d4564"] * len(data)
        annotations = filenames * 2
        #plot
        fig,ax = plt.subplots()
        sc = plt.scatter(x, y, s=5, c=colors)
        coef = np.polyfit(x,y,1)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
        poly1d_fn = np.poly1d(coef)
        main_line = plt.plot(x, poly1d_fn(x), '-', c="#000000", label='m='+str(round(slope, 4)))
        plt.title("PD/PI_BC= " + str(round(np.mean(ratios), 4)) + ", window=" + str(target_w) + ", n=" + str(len(x)/2))
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
        plt.savefig("figures/ice-cores/" + str(target_w))
        #plt.show()
        plt.close()
elif (inp == "c"):#Cartopy
    #Matplot
    #f, ax = plt.subplots(projection=cartopy.crs.Robinson())
    ax = plt.axes(projection=cartopy.crs.Robinson())
    ax.set_extent([-180, 180, -90, 90], crs=cartopy.crs.PlateCarree())

    #get this from https://www.naturalearthdata.com/features/
    glaciers = cartopy.feature.NaturalEarthFeature(
        category='physical',
        name='glaciated_areas',
        scale='10m',
        facecolor='#00A6E3')
    ax.add_feature(cartopy.feature.COASTLINE, edgecolor='grey')
    ax.add_feature(glaciers)

    #get duplicate pud core lat lons
    dup_index_map = {}
    p_dup = pd.read_csv('data/standardized-ice-cores/index-dup-cores.csv')
    for index, r in p_dup.iterrows():
        dup_index_map[r['Filename']] = [r['Lat'], r['Lon']]

    #setup color scale
    max_ratio = 0
    for key in for_cartopy.keys():
        #r = math.log(for_cartopy[key]['ratio'], 10)
        r = for_cartopy[key]['ratio']
        max_ratio = r if r > max_ratio else max_ratio
    #norm = Normalize(vmin=-max_ratio, vmax=max_ratio)
    norm = Normalize(vmin=0, vmax=2)
    cmap = colormaps['PRGn']#inferno
    sm = ScalarMappable(cmap=cmap, norm=norm)

    #plot
    for key in for_cartopy.keys():
        obj = for_cartopy[key]
        [lat, lon] = [obj['lat'], obj['lon']]
        if math.isnan(lat) or math.isnan(lon):
            lat, lon = dup_index_map[key]
        plt.plot(lon, lat, c=cmap(norm(obj['ratio'])), markeredgecolor='black', marker='^', markersize=7, transform=cartopy.crs.PlateCarree())
        #plt.plot(lon, lat, c=cmap(norm(math.log(obj['ratio'], 10))), markeredgecolor='black', marker='^', markersize=6, transform=cartopy.crs.PlateCarree())
    plt.colorbar(mappable=sm, label="PD/PI BC Conc.", orientation="horizontal")
    
    plt.savefig('figures/ice-cores/diverging-global-ratios-1980.png', dpi=300)
    #plt.show()

#index does not have lat lons of multiple ice cores from the same pub, plots them all at 0, 0