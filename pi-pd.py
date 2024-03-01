import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from matplotlib import colormaps
import plotly.express as px
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
                if filename != "legrand-2023-1.csv":
                    full_data[key].append([filename, lat, lon, a1[key], a2[key], a3[key]/a1[key], y1, y2, a3[key], y3])
            #cartopy
            for_cartopy[filename] = {'lat': lat, 'lon': lon, 'ratio': a3[5] / a1[5]}

#plot
inp = input("Matplot or Cartopy? (m/c): ")
def format_column(c):
    return np.transpose(c).astype('float64').tolist()[0]
if (inp == "m"): #Raw Matplot
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
        #plt.savefig("figures/ice-cores/" + str(target_w))
        plt.show()
        plt.close()
elif (inp == 'b'): #box plot
    for target_w in [5]:
        data = np.matrix(full_data[target_w])
        my_pi = [math.log(x, 10) for x in format_column(data[:,3])]
        my_pd = [math.log(x, 10) for x in format_column(data[:,4])]
        #my_pi = format_column(data[:,3])
        #my_pd = format_column(data[:,4])
        # rectangular box plot
        fig, ax = plt.subplots()
        bplot1 = ax.boxplot([my_pi, my_pd], vert=True, labels=[1850, 1980])
        #ax.set_title('Rectangular box plot')
        ax.yaxis.grid(True)
        ax.set_xlabel('Year')
        ax.set_ylabel('log10(PD/PI BC Conc.)')
        #ax.set_ylabel('PD/PI BC Conc.')
        #plt.ylim(0, 25)
        plt.savefig('figures/ice-cores/test-box-1.png', dpi=300)
        plt.show()
elif (inp == 'p'): #Plotly
    fig = px.scatter_geo(p, lat='N', lon='E', hover_name='First Author', title='PD/PI Ratios')
    fig.show()
elif (inp == "c"): #Cartopy
    #Matplot
    #f, ax = plt.subplots(projection=cartopy.crs.Robinson())
    #ax = plt.axes(projection=cartopy.crs.Robinson())
    ax = plt.axes(projection=cartopy.crs.RotatedPole())
    #ax.set_extent([-180, 180, -90, 90], crs=cartopy.crs.PlateCarree())

    #get this from https://www.naturalearthdata.com/features/
    glaciers = cartopy.feature.NaturalEarthFeature(
        category='physical',
        name='glaciated_areas',
        scale='110m',
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
    cmap = colormaps['BrBG_r']#inferno
    # extract all colors from the map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the last color entry to be red
    #cmaplist[-1] = (1, 0, 0, 1.0)
    # create the new map
    cmap = LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    # define the bins and normalize
    bounds = [round(x, 1) for x in np.linspace(0, 2, 8)]
    norm = BoundaryNorm(bounds, cmap.N)
    sm = ScalarMappable(cmap=cmap, norm=norm)

    #plot
    for key in for_cartopy.keys():
        obj = for_cartopy[key]
        [lat, lon] = [obj['lat'], obj['lon']]
        if math.isnan(lat) or math.isnan(lon):
            lat, lon = dup_index_map[key]
        temp = key.split('-')
        print(temp[0].capitalize() + " et al. " + temp[1], lat, lon, round(obj['ratio'], 3), temp[2])
        plt.plot(lon, lat, c=cmap(norm(obj['ratio'])), markeredgecolor='black', marker='.', markersize=9, transform=cartopy.crs.PlateCarree())
        #plt.plot(lon, lat, c=cmap(norm(math.log(obj['ratio'], 10))), markeredgecolor='black', marker='.', markersize=6, transform=cartopy.crs.PlateCarree())
    plt.colorbar(mappable=sm, label="PD/PI BC Conc.", orientation="horizontal")
    
    #plt.savefig('figures/ice-cores/discrete-global-ratios-1980.png', dpi=300)
    plt.show()

print("n=" + str(len(for_cartopy)))