import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.nonparametric.smoothers_lowess import lowess
from itertools import chain, combinations
from matplotlib.patches import Rectangle
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Wedge
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
from matplotlib import colormaps
import matplotlib.pyplot as plt
from matplotlib import rcParams
#from scipy.integrate import quad
#from scipy.stats import lognorm
from scipy.stats import norm
#import statsmodels.api as sm
#import plotly.express as px
#from netCDF4 import Dataset
import numpy as np
import platform
#import cartopy
import scipy
import tools
import math
import json
import csv
import sys
import os

#read index file
t = tools.ToolBox()
p = pd.read_csv('data/standardized-ice-cores/index.csv')
p = p.reset_index()

#setup vars
exclude = set([])#set(['mcconnell-2017-1.csv', 'brugger-2021-1.csv'])
windows = [11]#[1, 3, 5, 11]
system = platform.system()
full_data = {}
for key in windows:
    full_data[key] = [] #filename, lat, lon, PI averge, PD averge, ratio, PI year, PD year, 3 averege, 3 year
main_dict = {}
filename_region = {}
filename_index = {}
pd_recent = []
pd_1980 = []
name_bc = {}
name_yr = {}
dont_use = set()
a_p = 66.566667
m_g = 71.5 #midpoint between lowest greenland (60) and highest (83)
s_g = 60
patches = { #Okabe and Ito colorblind pallet
    'Arctic': (-15, a_p, 315, 90 - a_p, '#6CB3E4'),
    'South Greenland': (-55, s_g, 35, m_g - s_g, '#880D1E'),
    'North Greenland': (-60, m_g, 45, 90 - m_g, '#DDA138'),
    'Antarctica': (-180, -60, 360, -30, '#2C72AD'),
    'South ZAmerica': (-90, 15, 70, -71, '#EFE362'),
    'North America': (-170, 15, 115, a_p - 15, '#C17EA5'),
    'Europe': (-20, 23.5, 80, s_g - 23.5, '#C86526'),
    #'Middle east': (30, 23.5, 30, s_g - 23.5, '#DDA138'),
    'Africa': (-20, 23.5, 80, -58.5, '#000000'),
    'Asia': (60, 5, 90, a_p - 5, '#459B76')
}
'''
    'Greenland': (-55, s_g, 35, 90 - s_g, '#880D1E'),
    'East Antarctica': (-180, -60, 180, -30, '#000000'),
    'West Antarctica': (0, -60, 180, -30, '#6CB3E4'),'''
model_colors = {'CESM': '#EE692C', 'CMIP6': '#CC397C', 'Ice Core': '#6C62E7', 'CESM-SOOTSN': '#638FF6', 'LENS': '#F5B341', 'loadbc': '#CC397C'}

def within_patch(lat, lon, patch, name):
    lat_min, lat_max, lon_min, lon_max = t.patch_min_max(patch)
    if name == 'Arctic':
        #return lat_min <= lat <= lat_max and not within_patch(lat, lon, patches['Greenland'], 'Greenland')
        return lat_min <= lat <= lat_max and not within_patch(lat, lon, patches['South Greenland'], 'South Greenland') and not within_patch(lat, lon, patches['North Greenland'], 'North Greenland')
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max

def powerset(iterable):
    #"powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))

def divide(p_d, p_i):
    columns = set(p_d.columns)
    non_numeric = set(['model', 'n ensemble members', 'window'])
    bad = list(columns.intersection(non_numeric))
    return p_d.drop(bad, axis=1).div(p_i.drop(bad, axis=1))

def divide_pd_pi(p_d, p_i):
    output_columns = list(p_i.columns)
    models = list(set(p_d['model']).union(set(p_i['model'])))
    p_d = p_d.set_index('model')
    p_i = p_i.set_index('model')
    init_dict = {key: [] for key in output_columns}
    df = pd.DataFrame(data=init_dict)
    for i in range(len(models)):
        model = models[i]
        if model in p_d.index and model in p_i.index:
            df.loc[i] = [model] + list(p_d.loc[model].div(p_i.loc[model]))
    return df.drop(['model'], axis=1)

#alternative workflow cmip6 data:
if system == 'Linux':
    alt_df = pd.DataFrame()
    f = Dataset('data/model-ice-depo/cmip6/cmip6-bc-depo.nc')
    temp = f['new_var'][:]
    lats = f['lat'][:]
    lons = f['lon'][:]
    for filename, coords in t.get_ice_coords('data/standardized-ice-cores/index.csv', 'data/standardized-ice-cores/index-dup-cores.csv').items():
        lat, lon = coords
        alt_df[filename] = [temp[0][t.nearest_search(lats, lat)][t.nearest_search(lons, lon)]]
    f.close()
    alt_df.to_csv(os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'cmip6', 'alt-method.csv'))

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
        if filename == 'thompson-2002-1.csv':
            BC = np.flip(lowess(BC, Yr, frac=0.2, is_sorted=True, return_sorted=False))
        a1, y1 = t.get_avgs(Yr, BC, 1850.49, windows)
        #a1, y1 = t.get_avgs(Yr, BC, 1925.49, windows)
        #a2, y2 = t.get_avgs(Yr, BC, 9999, windows)
        a3, y3 = t.get_avgs(Yr, BC, 1980, windows)
        #a4, y4 = t.get_avgs(Yr, BC, 1750, windows)
        #a1800, y5 = t.get_avgs(Yr, BC, 1800, windows)
        #a1900, y5 = t.get_avgs(Yr, BC, 1800, windows)
        #add data to datasets
        if (y1 != None and y3 != None and abs(y1 - y3) >= 100):
            for key in windows:
                if math.isnan(lat) or math.isnan(lon):
                    lat, lon, abbr = dup_index_map[filename]
                #full_data[key].append([filename, lat, lon, a1[key], a3[key], a3[key]/a1[key], y1, y3, a3[key], y3])
                #pd_recent.append(y2)
                #pd_1980.append(y3)
                for region, patch in patches.items():
                    if within_patch(lat, lon, patch, region):
                        filename_region[filename] = region
            main_dict[filename] = {'lat': lat, 'lon': lon, 'ratio': a3[11] / a1[11], 'abbr': abbr, 'filename': filename}
            name_bc[filename] = BC
            name_yr[filename] = Yr
        else:
            dont_use.add(filename)

main_dict = {k: v for k, v in sorted(main_dict.items(), key=lambda item: item[1]['ratio'])} #sort by ratio
for i in range(len(main_dict.keys())):
    filename_index[list(main_dict.keys())[i]] = i + 1
final_pd = pd.DataFrame.from_records(main_dict).T

#plot
inp = 'z' if len(sys.argv) < 2 else sys.argv[1]
def format_column(c):
    return np.transpose(c).astype('float64').tolist()[0]
if (inp == 'm'): #Raw Matplot
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
        y3 = data[:,9].astype(float) 
        y2 = data[:,7].astype(float)
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
elif (inp == 'd'): #pd hists
    plt.hist(pd_1980)
    plt.title('PD = Nearest sample time to 1980 CE')
    plt.savefig('figures/ice-cores/pd-1980.png', dpi=200)
    plt.close()
    plt.hist(pd_recent)
    plt.title('PD = Most Recent Value')
    plt.savefig('figures/ice-cores/pd-recent.png', dpi=200)
    plt.close()
elif (inp == 'n'): #table of ice core numbers and filenames
    #setup data
    filenames = [x['filename'] for x in main_dict.values()]
    ratios = [x['ratio'] for x in main_dict.values()]
    index = [i + 1 for i in range(len(filenames))]
    df_n = pd.DataFrame({'core index': pd.Series(index, index=filenames), 'filename': pd.Series(filenames, index=filenames), 'ratio': pd.Series(ratios, index=filenames)}, index=filenames)
    df_n = df_n.drop(['filename'], axis=1)
    #setup 1750 data
    alt_ratios = pd.Series([x['1750ratio'] for x in main_dict.values()], index=filenames)
    ratio_1800 = pd.Series([x['1800ratio'] for x in main_dict.values()], index=filenames)
    ratio_1900 = pd.Series([x['1900ratio'] for x in main_dict.values()], index=filenames)
    diff = alt_ratios.sub(df_n['ratio'])
    df_n.insert(2, 'pd=1750', alt_ratios)
    df_n.insert(3, 'pd=1800', ratio_1800)
    df_n.insert(4, 'pd=1900', ratio_1900)
    df_n.insert(5, '1750-1850', diff)
    #setup color scale
    cmap = colormaps['BrBG_r']
    c_norm = Normalize(vmin=0, vmax=2)
    #plot with color
    vals = np.vectorize(lambda a : round(a, 2))(df_n.to_numpy())
    fix, ax = plt.subplots(figsize=(4, 2), dpi=300)
    ax.axis('off')
    colors = cmap(c_norm(vals))
    colors[:,0,:] = [1, 1, 1, 1] #make first column white
    #give 5th column different color scale
    for i in range(len(colors)):
        colors[i][5] = cmap(Normalize(vmin=-1, vmax=1)(diff.iloc[i]))
    table = plt.table(cellText=vals, colLabels=df_n.columns, loc='center', cellColours=colors)
    table.auto_set_font_size(False)
    table.set_fontsize(3)
    table.scale(0.5, 0.5)
    plt.savefig('figures/ice-cores/test-big-table-pi-comparison.png', bbox_inches='tight', pad_inches=0.0, dpi=300)
elif (inp == 'big-table'): #make table comparing individual models
    #setup cmip6 data
    filenames = [x['filename'] for x in main_dict.values()]
    ratios = [x['ratio'] for x in main_dict.values()]
    index = [i + 1 for i in range(len(filenames))]
    df = pd.DataFrame({'Index': pd.Series(index, index=filenames), 'filename': pd.Series(filenames, index=filenames), 'Ice Core': pd.Series(ratios, index=filenames)}, index=filenames)
    df = df.drop(['filename'], axis=1)
    cmip_binned = divide_pd_pi(pd.read_csv('data/model-ice-depo/cmip6/binned-pd.csv'), pd.read_csv('data/model-ice-depo/cmip6/binned-pi.csv')).T
    cmip_binned.columns = cmip_binned.loc['model']
    cmip_binned = cmip_binned.drop(['model'])
    cmip_binned = cmip_binned.join(cmip_binned.mean(axis=1).rename('CMIP6'))
    lens = pd.read_csv('data/model-ice-depo/lens/a10lv30.csv')
    lens = lens.rename(columns={"Unnamed: 0": "Restart"}).T
    lens.columns = lens.loc['Restart']
    lens = lens.drop(['Restart'])
    lens = lens.join(lens.mean(axis=1).rename('LENS'))
    min_restart = 9999999
    min_name = ''
    for column in lens.columns:
        temp_df = pd.DataFrame(df['Ice Core']).join(lens[column])
        diff = (temp_df['Ice Core'] - temp_df[column]).abs().sum()
        if diff < min_restart:
            min_restart = diff
            min_name = column
    #print(min_restart, min_name)
    df = df.join(cmip_binned, how='outer')
    df = df.join(lens, how='outer')
    df = df[df['Index'].notna()]
    df = df.sort_values('Index')
    #setup color scale
    cmap = colormaps['BrBG_r']
    c_norm = Normalize(vmin=0, vmax=2)
    #plot with color
    vals = np.vectorize(lambda a : round(a, 3))(df.to_numpy())
    red_mask = np.zeros(np.shape(df))
    for column in [n + 18 for n in range(18)] + ['LENS']:
        for i in range(len(df[column])):
            if np.abs(df['Ice Core'].iloc[i] - df[column].iloc[i]) < 0.1:
                red_mask[i][df.columns.get_loc(column)] = 1
    fix, ax = plt.subplots(figsize=(4, 2), dpi=300)
    ax.axis('off')
    colors = cmap(c_norm(vals))
    colors[:,0,:] = [1, 1, 1, 1] #make first column white
    for il in range(len(colors)):
        for ic in range(len(colors[il])):
            if red_mask[il][ic]:
                colors[il][ic] = [1, 0, 0, 1]
    table = plt.table(cellText=vals, colLabels=df.columns, loc='center', cellColours=colors, colWidths=[0.1] * len(df.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(5)
    plt.savefig('figures/ice-cores/test-big-table-cmip-models.png', bbox_inches='tight', pad_inches=0.0, dpi=300)
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
    fig = px.scatter_geo(final_pd, lat='lat', lon='lon', hover_name='filename', title='PD/PI Ratios')
    fig.show()
elif (inp == 'c'): #Cartopy
    projections = {
        'rotated-pole': {
            'projection': cartopy.crs.RotatedPole(pole_longitude=180.0, pole_latitude=36.0, central_rotated_longitude=-40),
            'extent': (-180.0, 180.0, -78.0, 73.0),
            'crs': cartopy.crs.RotatedPole(pole_longitude=180.0, pole_latitude=36.0, central_rotated_longitude=-40)
            },
        'north-pole': {
            'projection': cartopy.crs.NearsidePerspective(central_longitude=0, central_latitude=90),
            'extent': (180, -180, 65, 65),
            'crs': cartopy.crs.PlateCarree()
            },
        'antartica': {
            'projection': cartopy.crs.NearsidePerspective(central_longitude=0, central_latitude=-90),
            'extent': (180, -180, -65, -65),
            'crs': cartopy.crs.PlateCarree()
            }
        }
    #for globe
    #ax = plt.axes(projection=cartopy.crs.Robinson())
    #ax.set_extent([-180, 180, -90, 90], crs=cartopy.crs.PlateCarree())

    index_name_map = {}
    offsets = json.load(open('data/offsets.json'))
    for projection, params in projections.items():
        plt.clf()
        dpi = 300
        #figsize=(740/dpi, 740/dpi)
        fig, ax = plt.subplots(dpi=dpi, subplot_kw={'projection': params['projection']})
        ax.set_extent(params['extent'], crs=params['crs'])
        ax.add_feature(cartopy.feature.COASTLINE, edgecolor='grey')

        #elevation
        elev = Dataset('data/elevation-land-only.nc') #from https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2/ETOPO2v2-2006/ETOPO2v2c/netCDF/
        elev_lon = elev['lon'][:]
        elev_lat = elev['lat'][:]
        elev_z = np.transpose(elev['land_elev'][:])
        mesh = plt.pcolormesh(elev_lon, elev_lat, elev_z, cmap=colormaps['Greys'], vmin=0, transform=cartopy.crs.PlateCarree())

        #setup color scale
        cmap = colormaps['BrBG_r']#inferno
        # extract all colors from the map
        cmaplist = [cmap(i) for i in range(cmap.N)]
        #force middle color to be white
        cmaplist[len(cmaplist) // 2] = (1, 1.0, 1.0, 1.0)
        # force the last color entry to be red
        #cmaplist[-1] = (1, 0, 0, 1.0)
        # create the new map
        cmap = LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
        # define the bins and normalize
        bounds = [round(x, 1) for x in np.linspace(0, 2, 10)]
        c_norm = BoundaryNorm(bounds, cmap.N)
        sm = ScalarMappable(cmap=cmap, norm=c_norm)

        #plot
        i = 1
        for key in main_dict.keys():
            obj = main_dict[key]
            [lat, lon] = [obj['lat'], obj['lon']]
            scale = 0.7468 if projection == 'rotated-pole' else 1
            color = cmap(c_norm(obj['ratio']))
            stroke_color = "black" if color == (0.32941176470588235, 0.18823529411764706, 0.0196078431372549, 1.0) else "black"
            #temp = key.split('-')
            #print(temp[0].capitalize() + " et al. " + temp[1], lat, lon, round(obj['ratio'], 3), temp[2])
            modification = ""
            if key == 'sigl-2018-1.csv':
                modification = ",36"
            if key not in ('eichler-2023-1.csv'):
                if key != 'sigl-2018-1.csv':
                    plt.plot(lon, lat, c=color, markeredgecolor='black', marker='.', markersize=16*scale, transform=cartopy.crs.PlateCarree())
                else:
                    ax.add_patch(Wedge((-34, -8), 4 * scale, -90, 90, fc=cmap(c_norm(main_dict['eichler-2023-1.csv']['ratio'])), ec='black', zorder=999999))
                    ax.add_patch(Wedge((-34, -8), 4 * scale, 90, -90, fc=color, ec='black', zorder=999999))
                rcParams.update({'font.size': 12 * scale})
                if (projection == 'north-pole' and lat >= 60) or (projection == 'antartica' and lat <= -60) or (projection == 'rotated-pole' and -60 <= lat <= 60):
                    plt.text(lon + offsets[key][0], lat + offsets[key][1], " " + str(i) + modification, c="white", transform=cartopy.crs.PlateCarree(), path_effects=[pe.withStroke(linewidth=2*scale, foreground=stroke_color)])
            #plt.plot(lon, lat, c=cmap(norm(math.log(obj['ratio'], 10))), markeredgecolor='black', marker='.', markersize=6, transform=cartopy.crs.PlateCarree())
            index_name_map[i] = key
            i += 1
        if not projection in ('antartica', 'north-pole'):
            rcParams.update({'font.size': 10})
            plt.colorbar(mappable=sm, label="PD/PI BC Conc.", orientation="horizontal")
            #plt.colorbar(mappable=mesh.colorbar, label="Elevation (m)", orientation="horizontal")
        
        #patches
        '''for patch in patches.values():
            ax.add_patch(Rectangle(xy=[patch[0], patch[1]], width=patch[2], height=patch[3], facecolor=patch[4] + '50', edgecolor=patch[4],transform=cartopy.crs.PlateCarree()))'''

        plt.savefig('figures/ice-cores/testmap-' + projection + '.png', bbox_inches='tight', pad_inches=0.0)
        #plt.show()
    s = set()
    for pub in str(index_name_map).replace('}', '').replace('{', '').split(','):
        s.add(pub[t.find_nth(pub, " ", 2) + 2:t.find_nth(pub, "-", 2)])
    print(len(s), "unique ice core pubs")
    #print(index_name_map)
elif (inp == 'l'): #Lens data
    #setup data:
    models = {
        'LENS': {
            'dataset': pd.read_csv('data/model-ice-depo/lens/a10lv30.csv'),
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': model_colors['LENS'],#IBM Design library's colorblind pallete
            },
        'CESM': {
            'dataset': divide_pd_pi(pd.read_csv('data/model-ice-depo/cesm-wetdry/pd.csv'), pd.read_csv('data/model-ice-depo/cesm-wetdry/pi.csv')).mean(axis=0),
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': model_colors['CESM'],
            },
        'CMIP6': {
            'dataset': divide_pd_pi(pd.read_csv('data/model-ice-depo/cmip6/binned-pd.csv'), pd.read_csv('data/model-ice-depo/cmip6/binned-pi.csv')).mean(axis=0), #pd.read_csv(os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'cmip6', 'alt-method.csv')),
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': model_colors['CMIP6'],
            },
        'Ice Core': {
            'dataset': main_dict,
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': model_colors['Ice Core'],
            },
        'CESM-SOOTSN': {
            'dataset': divide_pd_pi(pd.read_csv('data/model-ice-depo/cesm-sootsn/pd.csv'), pd.read_csv('data/model-ice-depo/cesm-sootsn/pi.csv')).mean(axis=0),
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': model_colors['CESM-SOOTSN'],
            },
        'loadbc': {
            'dataset': divide_pd_pi(pd.read_csv('data/model-ice-depo/loadbc/binned-pd.csv'), pd.read_csv('data/model-ice-depo/loadbc/binned-pi.csv')).mean(axis=0),
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': model_colors['CMIP6'],
            }
    }
    '''models = {
        'LENS-S1': {
            'dataset': pd.read_csv('data/model-ice-depo/lens/a10lv30.csv'),
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': '#F5B341',#IBM Design library's colorblind pallete
            },
        'LENS-S4': {
            'dataset': pd.read_csv('data/model-ice-depo/lens/a10lv30s4.csv'),
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': '#CC397C',#IBM Design library's colorblind pallete
            },
        'LENS-S8': {
            'dataset': pd.read_csv('data/model-ice-depo/lens/a10lv30s8.csv'),
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': '#638FF6',#IBM Design library's colorblind pallete
            },
        'Ice Core': {
            'dataset': main_dict,
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': '#6C62E7',
            }
    }'''
    '''models = {
        'LENS-LV30': {
            'dataset': pd.read_csv('data/model-ice-depo/lens/a10lv30.csv'),
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': '#F5B341',#IBM Design library's colorblind pallete
            },
        'LENS-LV29': {
            'dataset': pd.read_csv('data/model-ice-depo/lens/a10lv29.csv'),
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': '#CC397C',#IBM Design library's colorblind pallete
            },
        'LENS-LV28': {
            'dataset': pd.read_csv('data/model-ice-depo/lens/a10lv28.csv'),
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': '#638FF6',#IBM Design library's colorblind pallete
            },
        'Ice Core': {
            'dataset': main_dict,
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': '#6C62E7',
            }
    }'''
    '''models = {
        'CMIP6 PD': {
            'dataset': pd.read_csv('data/model-ice-depo/cmip6/pd.csv'),
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': '#EE692C',
            },
        'CMIP6 PI': {
            'dataset': pd.read_csv('data/model-ice-depo/cmip6/pi.csv'),
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': '#CC397C',
            },
        'Ice Core': {
            'dataset': main_dict,
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': '#6C62E7',
            }
    }'''
    lens_pi = pd.read_csv('data/model-ice-depo/lens/pi.csv')
    lens_avg = models['LENS']['dataset']
    models_datasets = {key: value['dataset'] for key, value in models.items()}
    models_colors = {key: value['color'] for key, value in models.items()}
    n_models = len(list(models))
    dont_use = dont_use.union({'model number', 'Unnamed: 0', 'BC_vars', 'year'})
    bar_labels = []
    bar_means = {key: [] for key in models.keys()}
    bar_stds = {key: [] for key in models.keys()}
    distribution_map = {}
    ice_based_dists = {}
    ice_based_labels = []
    filenames = []
    lens_stds = []
    background_colors = []
    #get model means
    counter = 0
    for col_name in lens_avg.columns:
        if col_name in dont_use:
            continue
        counter += 1
        background_colors.append(patches[filename_region[col_name]][-1] + '30')
        for model_key in models.keys():
            model_data = models[model_key]['data']
            ds = models_datasets[model_key]
            if model_key == 'Ice Core':
                ice_mean = main_dict[col_name]['ratio']
                bar_means[model_key].append(ice_mean)
                #ice_based_dists[col_name] = {'PD': lens_pd[col_name], 'PI': lens_pi[col_name]}
                bar_labels.append(filename_region[col_name] + '-' + str(filename_index[col_name]).zfill(2))
                bar_stds[model_key].append(0)
            else:
                model_ratios = ds[col_name]
                model_mean = np.mean(model_ratios)
                #model_std = np.std(model_ratios.dropna())#scipy.stats.gstd(model_ratios.dropna())
                model_std = np.std(model_ratios)
                bar_stds[model_key].append(0)#(model_std)
                bar_means[model_key].append(model_mean)
        filenames.append(col_name)
    #resort everything by region
    print('ice core mean', np.mean(bar_means['Ice Core']))
    print(np.mean([np.mean(bar_means['CMIP6']), np.mean(bar_means['LENS']), np.mean(bar_means['CESM'])]))
    bar_labels, filenames, bar_means['LENS'], bar_means['Ice Core'], bar_means['CESM'], bar_means['CMIP6'], bar_stds['LENS'], bar_stds['Ice Core'], bar_stds['CESM'], bar_stds['CMIP6'], bar_means['CESM-SOOTSN'], bar_stds['CESM-SOOTSN'], background_colors = zip(*sorted(list(zip(bar_labels, filenames, bar_means['LENS'], bar_means['Ice Core'], bar_means['CESM'], bar_means['CMIP6'], bar_stds['LENS'], bar_stds['Ice Core'], bar_stds['CESM'], bar_stds['CMIP6'], bar_means['CESM-SOOTSN'], bar_stds['CESM-SOOTSN'], background_colors))))
    #bar_labels, bar_means['Ice Core'], bar_stds['Ice Core'], bar_means['LENS-S1'], bar_stds['LENS-S1'], bar_means['LENS-S4'], bar_stds['LENS-S4'], bar_means['LENS-S8'], bar_stds['LENS-S8'], background_colors = zip(*sorted(list(zip(bar_labels, bar_means['Ice Core'], bar_stds['Ice Core'], bar_means['LENS-S1'], bar_stds['LENS-S1'], bar_means['LENS-S4'], bar_stds['LENS-S4'], bar_means['LENS-S8'], bar_stds['LENS-S8'], background_colors))))
    #bar_labels, bar_means['Ice Core'], bar_stds['Ice Core'], bar_means['LENS-LV30'], bar_stds['LENS-LV30'], bar_means['LENS-LV29'], bar_stds['LENS-LV29'], bar_means['LENS-LV28'], bar_stds['LENS-LV28'], background_colors = zip(*sorted(list(zip(bar_labels, bar_means['Ice Core'], bar_stds['Ice Core'], bar_means['LENS-LV30'], bar_stds['LENS-LV30'], bar_means['LENS-LV29'], bar_stds['LENS-LV29'], bar_means['LENS-LV28'], bar_stds['LENS-LV28'], background_colors))))
    #bar_labels, bar_means['CMIP6 PD'], bar_stds['CMIP6 PD'], bar_means['CMIP6 PI'], bar_stds['CMIP6 PI'], background_colors = zip(*sorted(list(zip(bar_labels, bar_means['CMIP6 PD'], bar_stds['CMIP6 PD'], bar_means['CMIP6 PI'], bar_stds['CMIP6 PI'], background_colors))))
    #Remove Duplicate Region Labels
    region_lables = list(map(lambda x: x.split('-')[0], bar_labels))
    if len(sys.argv) == 2 or sys.argv[2] == 'var':
        transition_indexes = []
        csv_dict = []
        models_in_csv = set(['Ice Core'])
        old = ''
        for i in range(len(region_lables)):
            region = region_lables[i]
            if region == old:
                region_lables[i] = ''
            else:
                old = region
                transition_indexes.append(i)
        transition_indexes.append(transition_indexes[-1] + 1)
        #plot mean bars
        x = np.arange(len(bar_labels))  # the label locations
        rcParams.update({'font.size': 9})
        #divide data for each subfigure
        #sub_figures = [dict([['Ice Core', bar_means['Ice Core']], ['loadbc', bar_means['loadbc']], ['CESM', bar_means['CESM']], ['CESM-SOOTSN', bar_means['CESM-SOOTSN']], ['+2', 0]])]
        sub_figures = [
            dict(
                [['Ice Core', bar_means['Ice Core']], ['LENS', bar_means['LENS']], ['+1', 0], ['+2', 0], ['+3', 0]],
                ),
            dict(
                [['Ice Core', bar_means['Ice Core']], ['CMIP6', bar_means['CMIP6']], ['+1', 0], ['+2', 0], ['+3', 0]],
                ),
            dict(
                [['Ice Core', bar_means['Ice Core']], ['CESM', bar_means['CESM']], ['CESM-SOOTSN', bar_means['CESM-SOOTSN']], ['+1', 0], ['+2', 0]],
                ),
            dict(
                [['Ice Core', bar_means['Ice Core']], ['loadbc', bar_means['loadbc']], ['CESM', bar_means['CESM']], ['CESM-SOOTSN', bar_means['CESM-SOOTSN']], ['+2', 0]],
                )
        ]
        #plot rest of data
        for sub in sub_figures:
            print(list(sub.keys())[1])
            fig, ax = plt.subplots(layout='constrained')
            multiplier = 0
            width = 0.18
            #calcualte max height
            all_h = []
            means = []
            for key in sub.keys():
                if '+' not in key:
                    means.append(bar_means[key])
            for value in means:
                all_h.append(value)
            max_lens_h = np.max(all_h)
            min_h = np.min(all_h)
            #plot line at y=1
            ax.axhline(1, c='black', xmin=0, xmax=1, linewidth=0.75)
            for model_key in sub.keys():
                offset = width * multiplier
                bars = ax.bar(x + offset, max_lens_h, width, label=model_key, color=background_colors)
                if '+' in model_key:
                    multiplier += 1
                    continue
                measurement = bar_means[model_key]
                color = models_colors[model_key]
                #plot mean bars
                row = {'Model': model_key}
                for i in range(len(transition_indexes) - 1):
                    trans_i = transition_indexes[i]
                    next_i = transition_indexes[i + 1]
                    region = bar_labels[trans_i].split('-')[0]
                    x_start = (plt.getp(bars[trans_i], 'x') - offset + width + 0.07 - (trans_i / len(bar_labels)) * (1 - width * n_models)) / len(bar_labels)
                    x_end   = x_start + ((next_i - trans_i) - width - 0.07) / len(bar_labels) #(x_start + width * 4 * (next_i - trans_i)) / 36
                    plt.axhline(np.mean(bar_means[model_key][trans_i:next_i]), c=color, xmin=x_start, xmax=x_end, linewidth=0.75)
                    row[region] = str(round(np.mean(bar_means[model_key][trans_i:next_i]) - np.mean(bar_means['Ice Core'][trans_i:next_i]), 2))
                if model_key not in models_in_csv:
                    csv_dict.append(row)
                    models_in_csv.add(model_key)
                multiplier += 1
                plt.errorbar(x + offset, bar_means[model_key], yerr=bar_stds[model_key], fmt=".", color=color, elinewidth=0.5, capthick=0.5)
            ax.set_ylabel('1980/1850 BC Ratio')
            ax.set_xlabel('Region')
            ax.set_title('Modeled and Observed BC Deposition Change')
            ax.set_xticks(x + width, list(map(lambda x: x.replace('ZAmerica', 'America'), region_lables)))
            plt.xticks(rotation=90)
            plt.xlim([-width, x[-1] + width * n_models])
            plt.ylim([min_h - 0.01, max_lens_h])
            ax.get_xaxis().get_major_formatter().labelOnlyBase = False
            ax.set_yscale('log')
            ax.set_yticks([0.3, 0.5, 1, 2, 4])
            ax.get_yaxis().set_major_formatter(ScalarFormatter())
            ax2 = ax.twiny()
            plt.xlim([-width, x[-1] + width * n_models])
            plt.ylim([min_h - 0.01, max_lens_h])
            ax2.set_xticks(x + width, [int(label[len(label)-2:len(label)]) for label in bar_labels], fontsize=10)
            plt.xticks(rotation=90)
            ax2.set_xlabel('Ice Core Number')
            ax2.set_yscale('log')
            ax2.set_yticks([0.3, 0.5, 1, 2, 4])
            ax2.get_yaxis().set_major_formatter(ScalarFormatter())
            legend_handels = []
            for label in sub.keys():
                if '+' not in label:
                    legend_handels.append(Patch(label=label))
            ax.legend(handles=legend_handels)
            #manualy change legend colors
            leg = ax.get_legend()
            new_model_colors = {k: models_colors[k] if not '+' in k else 0 for k in list(sub.keys())}
            for bad_key in ('+1', '+2', '+3'):
                if bad_key in new_model_colors:
                    del new_model_colors[bad_key]
            for i in range(len(list(new_model_colors.items()))):
                key, color = list(new_model_colors.items())[i]
                leg.legend_handles[i].set_color(color)
            for a in plt.gcf().get_axes():
                for i in range(len(bar_labels)):
                    filename = bar_labels[i].split('-')[0]
                    color = patches[filename][-1]
                    a.get_xticklabels()[i].set_color(color)
            plt.savefig('figures/ice-cores/test4' + list(sub.keys())[1] + '.png', dpi=300)
        plt.close()
    if sys.argv[2] == 'test-table-color':
        #Model - Ice Core PD/PI Mean Regional Difference
        fields = [region for region in csv_dict[0].keys()]
        write_path = 'data/model-ice-depo/regional-model-diffs.csv'
        with open(write_path, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fields)
            writer.writeheader()
            writer.writerows(csv_dict)
        #setup table data
        rd_df = pd.read_csv(write_path)
        col_index = rd_df['Model']
        rd_df = rd_df.drop(['Model'], axis=1).T
        vals = rd_df.values
        #setup color scale
        cmap = t.custom_cmap([(0.5, 0.5, 1), (0.75, 0.75, 1), 0.25, (0.75, 0.75, 1), (1, 1, 1), 0.5, (1, 1, 1), (1, 0.75, 0.75,), 0.75, (1, 0.75, 0.75,), (1, 0.5, 0.5)])
        c_norm = Normalize(vmin=-2, vmax=2)
        colours = cmap(c_norm(vals))
        #colours[:,:,3] = 0.5
        sm = ScalarMappable(cmap=cmap, norm=c_norm)
        #plot with color
        fix, ax = plt.subplots(figsize=(14, 4), dpi=300)
        ax.axis('off')
        table = plt.table(cellText=vals, rowLabels=list(map(lambda x: x.replace('ZAmerica', 'America'), rd_df.index)), colLabels=col_index, loc='center', cellColours=colours)
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        plt.savefig('figures/ice-cores/test-table-color.png', bbox_inches='tight', pad_inches=0.0, dpi=300)
        plt.close()
    #plot variable figure
    if sys.argv[2] == 'var':
        print('var comparison boxplot')
        region2region = {'North Pole': 'Arctic', 'South Pole': 'Antarctica', 'Rest': 'Africa'}
        sub = dict(
            [['Ice Core', bar_means['Ice Core']],
                ['loadbc', bar_means['loadbc']],
                ['CESM', bar_means['CESM']],
                ['CESM-SOOTSN', bar_means['CESM-SOOTSN']]]
            )
        #get regional means
        new_regions = {
            'North Pole': ['North Greenland', 'South Greenland', 'Arctic'],
            'South Pole': ['Antarctica'],
            'Rest': ['Africa', 'Asia', 'Europe', 'North America', 'South ZAmerica'],
            }
        old2new = t.invert_dict_list(new_regions)
        rdf = pd.DataFrame(columns=['label'], data=bar_labels)
        rdf.insert(len(rdf.columns), 'region', [s.split('-')[0] for s in bar_labels])
        [rdf.insert(len(rdf.columns), model_key, bar_means[model_key]) for model_key in sub.keys()]
        rdf['region'] = rdf['region'].apply(lambda x: old2new[x])
        #organize into final datastruct
        data = {key: [] for key in sub.keys()}
        for model in data.keys():
            for region in new_regions.keys():
                data[model].append(rdf[rdf['region'] == region][model])
        #plot
        fig, ax = plt.subplots(layout='constrained')
        x = np.arange(len(list(new_regions.keys())))
        multiplier = 0
        width = 0.2
        bar_labels = list(new_regions.keys())
        bar_colors = [patches[region2region[s]][-1] + '30' for s in bar_labels]
        #for i in range(3): + (1/3) * i
        bar_width = 1
        box_heights = []
        for model in data.keys():
            c = model_colors[model]
            offset = width * multiplier
            bplot = ax.boxplot(data[model], widths=0.2, positions=x+offset, patch_artist=True, boxprops=dict(facecolor=c, color=c), capprops=dict(color=c), medianprops=dict(color='black'), flierprops=dict(color=c, markerfacecolor=c, markeredgecolor=c, marker= '.'), whiskerprops=dict(color=c))
            box_heights += [item.get_ydata()[1] for item in bplot['whiskers']]
            multiplier += 1
        bars = ax.bar(x + width * 1.5, np.max(box_heights) + 0.1, bar_width, color=bar_colors)
        ax.set_yscale('log')
        ax.set_xticks(x + width * 1.5, bar_labels)
        ax.set_xlim([x[0] + width * 1.5 - bar_width / 2, x[-1] + width * 1.5 + bar_width / 2])
        ax.set_ylim([0, np.max(box_heights) + 0.1])
        ax.set_yticks([0.3, 0.5, 1, 2, 4])
        ax.set_ylabel("1980/1850 Ratio")
        ax.set_xlabel("Region")
        ax.get_yaxis().set_major_formatter(ScalarFormatter())
        #manually setup legend
        legend_handels = []
        leg_dict = {'Ice Core': 'Ice Core', 'loadbc': 'loadbc', 'CESM': 'drybc - wetbc', 'CESM-SOOTSN': 'sootsn'}
        for label in sub.keys():
            if '+' not in label:
                legend_handels.append(Patch(label=leg_dict[label]))
        ax.legend(handles=legend_handels)
        #manualy change legend colors
        leg = ax.get_legend()
        new_model_colors = {k: models_colors[k] for k in list(sub.keys())}
        for i in range(len(list(new_model_colors.items()))):
            key, color = list(new_model_colors.items())[i]
            leg.legend_handles[i].set_color(color)
        for i in range(len(bar_labels)):
            color = patches[region2region[bar_labels[i]]][-1]
            ax.get_xticklabels()[i].set_color(color)
        plt.savefig('figures/ice-cores/test-var.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
    #plot antartica supersets
    elif sys.argv[2] == 'ant':
        #setup east and west
        antarctic_names = set()
        for value, key in filename_region.items():
            if key == 'Antarctica':
                antarctic_names.add(value)
        def name_set2mean(sets):
            data = {}
            for key in bar_means.keys():# for model name
                east_temp = []
                east_temp_2 = []
                west_temp = []
                west_temp_2 = []
                for east in sets:# for set of eastern ice cores
                    west = antarctic_names.difference(east)
                    for i in range(len(filenames)):# for each ice core
                        filename = filenames[i]
                        if filename in east:
                            east_temp.append(bar_means[key][i])
                        elif filename in west:
                            west_temp.append(bar_means[key][i])
                    east_temp_2.append(np.mean(east_temp))
                    west_temp_2.append(np.mean(west_temp))
                data[key] = list(zip(east_temp_2, west_temp_2))
            return data
        ant_data = name_set2mean(list(powerset(antarctic_names)))
        #ant data structure:
        #ant_data = {keys: model_name,  values: list of (east mean, west mean)}
        for key, value in ant_data.items():
            #plot
            fig, ax = plt.subplots(layout='constrained')
            plt.title(key)
            width = 0.49
            multiplier = 0
            x = np.arange(len(list(ant_data.values())[0]))
            new_data = np.reshape(ant_data[key], len(ant_data[key])*2)
            for i in range(1):
                color = '#FF0000' if i == 0 else '#00FF00'
                offset = width * multiplier
                #ax.bar(x + offset, new_data[i::2], width, color=color)
                plt.hist(new_data[i::2])
                multiplier += 1
            plt.savefig('figures/ice-cores/test5' + key + '.png', bbox_inches='tight', pad_inches=0.0, dpi=200)
            plt.close()
    elif sys.argv[2] == 'scatter':
        fig, ax = plt.subplots(layout='constrained')
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(bar_means['loadbc'], bar_means['CESM-SOOTSN'])
        plt.scatter(bar_means['loadbc'], bar_means['CESM-SOOTSN'], color=model_colors['CMIP6'])
        perfect_x = [x * ax.get_xlim()[1] / 100 for x in range(100)]
        plt.plot(perfect_x, perfect_x, color='black')
        plt.xlabel('loadbc pd/pi')
        plt.ylabel('sootsn pd/pi')
        plt.title('R^2 = ' + str(r_value**2))
        plt.savefig('figures/ice-cores/sootsn-loadbc-scatter.png', bbox_inches='tight', pad_inches=0.0, dpi=200)
    #plot distribution bars by model
    '''
    #get lens distributions by model
        for index, row in lens_pi.iterrows():
            model_index = int(row['model number'].split('-')[0])
            ice_based_labels.append(model_index)
            ratio_dists = lens_avg.iloc[model_index - 19].iloc[1:len(lens_avg.iloc[model_index - 19])]#row.iloc[3:len(row)] / lens_pi.iloc[index].iloc[3:len(row)]
            distribution_map[model_index] = ratio_dists
        distribution_lables = row.iloc[3:len(row)].index
    x = np.arange(len(distribution_lables))
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
    '''plt.clf()
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
    bins = [1/np.power(10, i * -1) for i in t.get_ticks(-15, -9, 10)]
    #powers = (-16, -13.67, -12.67, -12.33, -11.33, -9)
    #powers =  t.get_ticks(-16, -9, 4)
    #bins = [1/np.power(10, i * -1) for i in powers]
    #distribution_labels = t.bins2labels(["E" + str(i) for i in powers])
    distribution_labels = t.bins2labels(bins)
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
    hist = np.histogram(all_lens_data, bins=bins, range=(min_bc, max_bc))
    ax.bar(x, hist[0], width, label=distribution_labels, color=color)
    ax.set_xticks(x, distribution_labels, fontsize=4)
    plt.savefig('figures/ice-cores/test1.png', dpi=150)
    #pdf
    plt.clf()
    stdev, location, mean = lognorm.fit(all_lens_data)#model_ratios#all_lens_data
    #stdev = np.std(model_ratios)
    #mean = np.mean(model_ratios)
    phi = (stdev ** 2 + mean ** 2) ** 0.5
    mu = np.log(mean ** 2 / phi)
    sigma = (np.log(phi ** 2 / mean ** 2)) ** 0.5
    data=np.random.lognormal(mu, sigma , 1000)
    a, b, _  = plt.hist(all_lens_data, bins=10, density=True, alpha=0.5, color='b')
    xmin, xmax = plt.xlim()#(lognorm.ppf(0.01, sigma), lognorm.ppf(0.99, sigma))#(-2.4730680657277782e-11, 5.19344481734187e-10)#plt.xlim()
    x = np.linspace(xmin, xmax, 1000)
    p = lognorm.pdf(x = x, scale = mean, s=sigma)
    #p = norm.cdf(x = x, scale = stdev, loc=mean)
    cdf_target = x[-10]
    #cdf_xmax = norm.cdf(x = cdf_target, scale = stdev, loc=mean)
    cdf_xmax = lognorm.cdf(x = cdf_target, scale = mean, s=sigma)
    plt.plot(x, p, 'green', linewidth=2)
    #plt.vlines(cdf_target, 0, cdf_xmax, color="red")
    #plt.hlines(cdf_xmax, xmin, cdf_target, color="red")
    plt.yscale('log')
    #plt.legend(['CDF', "", "CDF at " + str(round(cdf_target, 3)) + "=" + str(round(cdf_xmax, 4)), 'Histogram'])
    plt.legend(['PDF', 'Histogram'])
    plt.savefig('figures/ice-cores/test-pdf.png', dpi=150)
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
    plt.savefig('figures/ice-cores/test2.png', dpi=50)'''
elif (inp == 's'): #smoothing
    def latest_range(l, window):
        return l[len(l)-window:len(l)]
    for index, row in p.iterrows():
        for i in range(row['n_cores']):
            filename = row['First Author'].lower() + '-' + str(row['Year']) + '-' + str(i + 1) + '.csv'
            if filename == 'thompson-2002-1.csv':
                lat = row['N']
                lon = row['E']
                abbr = row['Abbreviation']
                d = pd.read_csv('data/standardized-ice-cores/' + filename)
                #must be flipped bec they are in decending order
                window = 70
                BC = np.flip(d['BC'].to_numpy())
                Yr = np.flip(d['Yr'].to_numpy())
                test = {}
                fracs = [0, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
                fig, ax = plt.subplots()
                ax.set_yscale('log')
                #plt.axhline(0, c='black', xmin=0, xmax=1, linewidth=1)
                for frac in fracs:
                    if frac != 0:
                        temp = np.flip(lowess(BC, Yr, frac=frac, is_sorted=True, return_sorted=False))
                    else:
                        temp = BC
                    y = latest_range(temp, window)
                    x = latest_range(Yr, window)
                    #plt.scatter(x, y)
                    #plt.ylim((-100, 500))
                    plt.plot(x, y)
                plt.legend(fracs)
                plt.show()
#timeseries
elif (inp == 't'):
    df_time = pd.read_csv(os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'binned-timeseries.csv'))
    big_arr = []
    csv_filenames = []
    dont_use = set(['thompson-2002-1.csv', 'liu-2021-1.csv', 'liu-2021-2.csv', 'liu-2021-3.csv', 'liu-2021-4.csv', 'liu-2021-5.csv', 'mcconnell-2022-1.csv'])
    #for hem in ['Northern Hemisphere', 'Southern Hemisphere']:
    x = [i + 0.5 for i in range(1850, 1981)]
    for name in name_bc.keys():
        if name in dont_use:
            continue
        #elif (main_dict[name]['lat'] > 0 and hem == 'Northern Hemisphere') or (main_dict[name]['lat'] < 0 and hem == 'Southern Hemisphere'):
        if main_dict[name]['lat'] <= -60:
            y = np.interp(x, name_yr[name], name_bc[name])
            big_arr.append(y)
            csv_filenames.append(name)
    path = 'data/model-ice-depo/timeseries-.csv'
    np.savetxt(path, np.asarray(big_arr).T, delimiter=",", header=','.join(csv_filenames), comments='')
    df_ice = pd.read_csv(path)
    df_time['Ice Core'] = df_ice.mean(axis=1)
    fig, ax = plt.subplots()
    ax.plot(x, [0 for i in x], c='grey') #line at y=0
    legend_elms = []
    for model in df_time.columns:
        if model in ['Unnamed: 0', 'LENS', 'CMIP6', 'CESM', 'Ice Core']:#['Unnamed: 0', 'LENS']:#
            continue
        color = model_colors[model]
        ax.plot(x, df_time[model], c=color) #avg line
        legend_elms.append(Patch(facecolor=color, label=model))
        if model == 'Ice Core':
            ax.plot(x, df_ice.min(axis=1), c='grey') #lower bound
            ax.plot(x, df_ice.max(axis=1), c='grey') #upper bound
    plt.yscale("log")
    plt.xlim([1850, 1980])
    #plt.ylim([0, 1])
    legend_elms.append(Patch(facecolor='grey', label='Ice Core Min/Max'))
    ax.legend(handles=legend_elms)
    plt.savefig('figures/ice-cores/test-timesries.png', bbox_inches='tight', pad_inches=0.0, dpi=300)
    plt.close()
elif (inp == 'z'):#testing
    pass

print("n=" + str(len(main_dict)))