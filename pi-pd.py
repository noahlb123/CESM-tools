import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
from matplotlib import colormaps
import matplotlib.pyplot as plt
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
exclude = set([])#set(['mcconnell-2017-1.csv', 'brugger-2021-1.csv'])
windows = [1, 3, 5, 11]
full_data = {}
for key in windows:
    full_data[key] = [] #filename, lat, lon, PI averge, PD averge, ratio, PI year, PD year, 3 averege, 3 year
for_cartopy = {}

#fix duplicate pud core lat lons
dup_index_map = {}
p_dup = pd.read_csv('data/standardized-ice-cores/index-dup-cores.csv')
for index, r in p_dup.iterrows():
    dup_index_map[r['Filename']] = [r['Lat'], r['Lon'], r['Abbreviation']]

#read each ice core file
for index, row in p.iterrows():
    for i in range(row['n_cores']):
        filename = row['First Author'].lower() + '-' + str(row['Year']) + '-' + str(i + 1) + '.csv'
        lat = row['N']
        lon = row['E']
        abbr = row['Abbreviation']
        if (filename in exclude):
            continue
        d = pd.read_csv('data/standardized-ice-cores/' + filename)
        #must be flipped bec they are in decending order
        BC = np.flip(d['BC'].to_numpy())
        Yr = np.flip(d['Yr'].to_numpy())
        a1, y1 = t.get_avgs(Yr, BC, 1850.49, windows)
        #a1, y1 = t.get_avgs(Yr, BC, 1925.49, windows)
        a2, y2 = t.get_avgs(Yr, BC, 9999, windows)
        a3, y3 = t.get_avgs(Yr, BC, 1980, windows)
        #add data to datasets
        if (y1 != None and y2 != None and abs(y1 - y2) >= 100 and y1 < 1900):
            for key in windows:
                if math.isnan(lat) or math.isnan(lon):
                    lat, lon, abbr = dup_index_map[filename]
                if True:#filename != "legrand-2023-1.csv":
                    full_data[key].append([filename, lat, lon, a1[key], a2[key], a3[key]/a1[key], y1, y2, a3[key], y3])
            #cartopy
            for_cartopy[filename] = {'lat': lat, 'lon': lon, 'ratio': a3[5] / a1[5], 'abbr': abbr}
final_pd = pd.DataFrame.from_records(for_cartopy).T

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
        #my_pi = [math.log(x, 10) for x in format_column(data[:,3])]
        #my_pd = [math.log(x, 10) for x in format_column(data[:,4])]
        my_pi = format_column(data[:,3])
        my_pd = format_column(data[:,4])
        # rectangular box plot
        fig, ax = plt.subplots()
        bplot1 = ax.boxplot([my_pi, my_pd], vert=True, labels=[1850, 1980])
        #ax.set_title('Northern Hemisphere')
        ax.yaxis.grid(True)
        ax.set_xlabel('Year')
        #ax.set_ylabel('PD/PI BC Conc.')
        #ax.set_yscale('log')
        #ax.set_yticks([0, 1, 2, 3, 4, 10, 20, 30, 40, 50, 60])
        #ax.get_xaxis().set_major_formatter(ScalarFormatter())
        plt.yscale("log")
        ax.set_ylabel('PD/PI BC Conc.')
        #plt.ylim(0, 25)
        #plt.savefig('figures/ice-cores/southern-box.png', dpi=300)
        plt.show()
elif (inp == 'p'): #Plotly
    fig = px.scatter_geo(final_pd, lat='lat', lon='lon', hover_name='abbr', title='PD/PI Ratios')
    fig.show()
elif (inp == "c"): #Cartopy
    #Matplot
    #for globe
    #ax = plt.axes(projection=cartopy.crs.Robinson())
    #ax.set_extent([-180, 180, -90, 90], crs=cartopy.crs.PlateCarree())
    #for rotated pole
    rp = cartopy.crs.RotatedPole(pole_longitude=180.0, pole_latitude=36.0, central_rotated_longitude=-40)#-106
    ax = plt.axes(projection=rp)
    ax.set_extent((-180.0, 180.0, -78.0, 73.0), crs=rp)
    #xs, ys, zs = rp.transform_points(cartopy.crs.PlateCarree(), np.array([-180, 180]), np.array([-90, 90])).T
    #ax.set_xlim(xs)
    #ax.set_ylim(ys)
    #for antartica
    #ax = plt.axes(projection=cartopy.crs.NearsidePerspective(central_longitude=0.0, central_latitude=-90))
    #ax.set_extent([-2536032.75925479, 2536640.3242591335, -2*1045053.5124408401, 2*1878973.7356212165], crs=cartopy.crs.NearsidePerspective(central_longitude=0.0, central_latitude=-90))

    #get this from https://www.naturalearthdata.com/features/
    glaciers = cartopy.feature.NaturalEarthFeature(
        category='physical',
        name='glaciated_areas',
        scale='110m',
        facecolor='#00A6E3')
    ax.add_feature(cartopy.feature.COASTLINE, edgecolor='grey')
    ax.add_feature(glaciers)

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
    bounds = [round(x, 1) for x in np.linspace(0, 2, 10)]
    norm = BoundaryNorm(bounds, cmap.N)
    sm = ScalarMappable(cmap=cmap, norm=norm)

    #plot
    for key in for_cartopy.keys():
        obj = for_cartopy[key]
        [lat, lon] = [obj['lat'], obj['lon']]
        #if (lat < -61):
        temp = key.split('-')
        #print(temp[0].capitalize() + " et al. " + temp[1], lat, lon, round(obj['ratio'], 3), temp[2])
        plt.plot(lon, lat, c=cmap(norm(obj['ratio'])), markeredgecolor='black', marker='.', markersize=9, transform=cartopy.crs.PlateCarree())
        #plt.plot(lon, lat, c=cmap(norm(math.log(obj['ratio'], 10))), markeredgecolor='black', marker='.', markersize=6, transform=cartopy.crs.PlateCarree())
    plt.colorbar(mappable=sm, label="PD/PI BC Conc.", orientation="horizontal")
    
    #plt.savefig('figures/ice-cores/rotated-pole.png', dpi=300)
    plt.show()
elif (inp == "l"): #Lens data
    #lens are in 5 year avereges so comparing like to like
    #setup data:
    lens_pi = pd.read_csv('data/lens/pi.csv')
    lens_pd = pd.read_csv('data/lens/pd.csv')
    dont_use = {'model number', 'BC_vars', 'year', 'mcconnell-2021-6.csv', 'ming-2008-1.csv', 'sierra-hernández-2022-1.csv', 'wolff-2012-1.csv', 'mcconnell-2021-2.csv', 'mcconnell-2017-1.csv', 'xu-2009-1.csv'}
    bar_lables = []
    bar_means = {'LENS Models': [], 'Ice Core': []}
    distribution_map = {}
    ice_based_dists = {}
    ice_based_labels = []
    lens_stds = []
    #get distributions by model
    #lens_pd = lens_pd.drop(['chellman-2017-1.csv', 'xu-2009-1.csv'], axis=1)#drop high value sites
    for index, row in lens_pd.iterrows():
        model_index = int(row['model number'].split('-')[0])
        ice_based_labels.append(model_index)
        ratio_dists = row.iloc[3:len(row)] / lens_pi.iloc[index].iloc[3:len(row)]
        distribution_map[model_index] = ratio_dists
    distribution_lables = row.iloc[3:len(row)].index
    #get model means
    for col_name in lens_pd.columns:
        if col_name in dont_use:
            continue
        model_ratios = lens_pd[col_name] / lens_pi[col_name]
        ice_based_dists[col_name] = {'PD': lens_pd[col_name], 'PI': lens_pi[col_name]}
        model_mean = np.mean(model_ratios)
        model_std = np.std(model_ratios)
        ice_mean = for_cartopy[col_name]['ratio']
        if True: #for_cartopy[col_name]['lat'] > 0: #if in given hemisphere
            bar_lables.append(col_name)
            lens_stds.append(model_std)
            bar_means['LENS Models'].append(model_mean)
            bar_means['Ice Core'].append(ice_mean)
    #plot mean bars
    '''x = np.arange(len(bar_lables))  # the label locations
    width = 0.4  # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    for attribute, measurement in bar_means.items():
        color = "#000000" if attribute == "LENS Models" else "#00A6E3"
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=color)
        multiplier += 1
    plt.errorbar(bar_lables, bar_means['LENS Models'], yerr=lens_stds, fmt="o", color="r")
    ax.set_ylabel('1980/1925 BC Ratio')
    ax.set_title('N-Hemisphere LENS vs Ice Core BC Change')
    ax.set_xticks(x + width, bar_lables)
    plt.xticks(rotation=90)
    ax.legend()
    plt.show()
    #plt.savefig('figures/ice-cores/lens-std.png', dpi=300)'''
    #plot distribution bars by model
    '''x = np.arange(len(distribution_lables))
    fig, axes = plt.subplots(len(distribution_map), dpi=300, figsize=(4*1000/300, 4*1000/300))
    fig.suptitle('LENS BC Distributions')
    i = 0
    for key, value in distribution_map.items():
        ax = axes[i]
        ax.bar(x, value, label=distribution_lables)
        ax.set_title('LENS Model ' + str(key))
        #ax.yaxis.set_major_locator(ticker.NullLocator())
        ax.xaxis.set_major_locator(ticker.NullLocator())
        i += 1
    axes[-1].set_xticks(x, distribution_lables)
    plt.xticks(rotation=90)
    #plt.show()
    plt.subplots_adjust(hspace=0.7) #add spacing between subplots
    plt.savefig('figures/ice-cores/test.png', dpi=300)'''
    #plot distribution bars by ice core
    x = np.arange(len(ice_based_labels))
    width = 0.4
    multiplier = 0
    #get max and min bc
    max_bc = 0
    min_bc = 10
    for ice_core_name, distribution in ice_based_dists.items():
        for key, value in distribution.items():
            max_bc = max(value) if max(value) > max_bc else max_bc
            min_bc = min(value) if min(value) < min_bc else min_bc
    #actually plot
    #bins = [1/np.power(10, i * -1) for i in t.get_ticks(-16, -9, 4)]
    powers = (-16, -13.67, -12.67, -12.33, -11.33, -9)
    #powers =  t.get_ticks(-16, -9, 4)
    bins = [1/np.power(10, i * -1) for i in powers]
    distribution_labels = t.bins2labels(["E" + str(i) for i in powers])
    #distribution_labels = t.bins2labels(bins)
    x = np.arange(len(distribution_labels))
    all_lens_data = []
    #plot all togethere
    for ice_core_name, distribution in ice_based_dists.items():
        for key, value in distribution.items():
            if True:#key == "PI":
                color = "#00A6E3"
                offset = width * multiplier
                all_lens_data = np.concatenate((all_lens_data, value.to_list()))
                multiplier += 1
        i += 1
    fig, ax = plt.subplots(layout='constrained')
    ax.bar(x, np.histogram(all_lens_data, bins=bins, range=(min_bc, max_bc))[0], width, label=distribution_labels, color=color)
    ax.set_xticks(x, distribution_labels, fontsize=8)
    plt.savefig('figures/ice-cores/test1.png', dpi=300)
    #plot all sepratley
    plt.clf()
    i = 0
    fig, axes = plt.subplots(len(ice_based_dists), dpi=300, figsize=(4*1000/300, 4*1000/300))
    fig.suptitle('LENS BC Histograms')
    for ice_core_name, distribution in ice_based_dists.items():
        ax = axes[i]
        for key, value in distribution.items():
            if True:#key == "PI":
                color = "#000000" if key == "PI" else "#00A6E3"
                offset = width * multiplier
                ax.bar(x + offset, np.histogram(value, bins=bins, range=(min_bc, max_bc))[0], width, label=key, color=color)
                multiplier += 1
        ax.set_title(ice_core_name)
        ax.xaxis.set_major_locator(ticker.NullLocator())
        i += 1
    axes[-1].set_xticks(x + 82.5 * width, distribution_labels)
    axes[0].legend()
    plt.subplots_adjust(hspace=0.9, top=0.95, bottom=0.02) #add spacing between subplots
    plt.savefig('figures/ice-cores/test2.png', dpi=300)
elif (inp == "t"): #testing
    print(t.get_ticks(-13, -12, 4))


print("n=" + str(len(for_cartopy)))