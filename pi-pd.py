import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from statsmodels.nonparametric.smoothers_lowess import lowess
from itertools import chain, combinations
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Wedge
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
from matplotlib.pyplot import gca
from matplotlib import colormaps
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import gaussian_kde
from scipy.stats import norm
import statsmodels.api as sm
from scipy.stats import iqr
from netCDF4 import Dataset
import plotly.express as px
import numpy as np
import platform
import cartopy
import random
import scipy
import tools
import math
import json
import time
import csv
import sys
import os

#note legendHandles should be used for Matplotlib versions before 3.7 and legend_handles should be used afterwards

#read index file
t = tools.ToolBox()
p = pd.read_csv('data/standardized-ice-cores/index.csv')
p = p.reset_index()

#setup vars
exclude = set([])#set(['mcconnell-2017-1.csv', 'brugger-2021-1.csv'])
windows = [25] #25 is best
system = platform.system()
full_data = {}
for key in windows:
    full_data[key] = [] #filename, lat, lon, PI averge, PD averge, ratio, PI year, PD year, 3 averege, 3 year
main_dict = {}
filename_region = {}
filename_index = {}
filename_lat_lon = {}
pd_recent = []
pd_1980 = []
name_bc = {}
name_yr = {}
dont_use = set()
a_p = 66.566667
m_g = 71.5 #midpoint between lowest greenland (60) and highest (83)
s_g = 60
patches = { #Okabe and Ito colorblind pallet
    'Arctic': (-15, a_p, 315, 90 - a_p, '#6CB3E4', '#6CB3E4'),
    'South Greenland': (-60, s_g, 45, m_g - s_g, '#880D1E', '#6CB3E4'), #'#880D1E'),
    'North Greenland': (-60, m_g, 45, 90 - m_g, '#DDA138', '#6CB3E4'), #'#DDA138'),
    'Antarctica': (-180, -60, 360, -30, '#2C72AD', '#2C72AD'),
    'South ZAmerica': (-90, 15, 70, -71, '#EFE362', '#000000'), #'#EFE362'),
    'North America': (-170, 15, 115, a_p - 15, '#C17EA5', '#000000'), #'#C17EA5'),
    'Europe': (-20, 23.5, 80, s_g - 23.5, '#C86526', '#000000'), #'#C86526'),
    #'Middle east': (30, 23.5, 30, s_g - 23.5, '#DDA138'),
    'Africa': (-20, 23.5, 80, -58.5, '#000000', '#000000'),
    'Asia': (60, 5, 90, a_p - 5, '#459B76', '#000000'), #'#459B76')
    'Alaska': (0, 0, 0, 0, '#2C72AD', '#000000'),
}
'''
    'Greenland': (-55, s_g, 35, 90 - s_g, '#880D1E'),
    'East Antarctica': (-180, -60, 180, -30, '#000000'),
    'West Antarctica': (0, -60, 180, -30, '#6CB3E4'),'''
#IBM Design Library colorblind pallet https://www.nceas.ucsb.edu/sites/default/files/2022-06/Colorblind%20Safe%20Color%20Schemes.pdf
model_colors = {'CESM': '#EE692C', 'CMIP6': '#CC397C', 'Ice Core': '#6C62E7', 'CESM-SOOTSN': '#638FF6', 'LENS': '#F5B341', 'LENS-Bias': '#CC397C', 'loadbc': '#CC397C', 'mmrbc': '#F5B341', 'Anthro Emissions': '#638FF6', 'LENS-18': '#EE692C', 'LENS-25': '#F5B341'}

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

def plot_timeseries_decomp(x, y, name):
    plt.rc('font', size=10)
    timeseries_windows = [10, 25, 33]
    for i in timeseries_windows:
        if len(x) < 2 * i:
            print('length too short for p=', i)
            continue
        fig, ax = plt.subplots(4, 1)
        fig.tight_layout()
        res = sm.tsa.seasonal_decompose(y, period=i)
        ax[0].plot(x, res.seasonal, label='seasonal')
        ax[0].set_title("Seasonal  period=" + str(i))
        ax[2].plot(x, res.resid, label='residual')
        ax[2].set_title("Residual")
        ax[1].plot(x, res.trend, label='trend')
        ax[1].set_title("Trend")
        ax[3].plot(x, y, label='obs', color='black')
        ax[3].set_title("Observation")
        for ax_i in range(4):
            ax[ax_i].set_xlim([1850, 1980])
        plt.savefig('figures/ice-cores/decomposition/' + name + '-' + str(i) + '.png', dpi=200)
        plt.close()

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
        if filename == 'legrand-2023-1.csv':
            a1, y1, temp, temp = t.simplified_avg(Yr, BC, 1881.5, [6.5])
            a1[25] = a1[6.5]
        else:
            a1, y1, temp, temp = t.simplified_avg(Yr, BC, 1850.49 + 25 / 2, windows)
        a3, y3, temp, temp = t.simplified_avg(Yr, BC, 1980 - 25 / 2, windows)
        if len(sys.argv) >= 2 and sys.argv[1] == 'n':
            big_table_years = {'1750': None, '1800': None, '1850': None, '1900': None, '1950': None, '1980': None}
            for key in big_table_years.keys():
                mod_year = int(key) - (windows[0] / 2) if int(key) > 1900 else int(key) + (windows[0] / 2)
                a_temp, y_temp, temp, temp = t.simplified_avg(Yr, BC, mod_year, windows)
                big_table_years[key] = a_temp[windows[0]]
        #add data to datasets
        if (y1 != None and y3 != None and abs(y1 - y3) >= 85):
            for key in windows:
                if math.isnan(lat) or math.isnan(lon):
                    lat, lon, abbr = dup_index_map[filename]
                filename_lat_lon[filename] = [lat, lon]
                #full_data[key].append([filename, lat, lon, a1[key], a3[key], a3[key]/a1[key], y1, y3, a3[key], y3])
                #pd_recent.append(y2)
                #pd_1980.append(y3)
                for region, patch in patches.items():
                    if within_patch(lat, lon, patch, region):
                        filename_region[filename] = region
            main_dict[filename] = {'lat': lat, 'lon': lon, 'ratio': a3[windows[0]] / a1[windows[0]], 'abbr': abbr, 'filename': filename}
            if len(sys.argv) >= 2 and sys.argv[1] == 'n':
                big_table_years['1980'] = a3[windows[0]]
                big_table_years['1850'] = a1[windows[0]]
                for p_i in big_table_years.keys():
                    for p_d in big_table_years.keys():
                        if p_i < p_d:
                            main_dict[filename][p_d + '/' + p_i] = big_table_years[p_d] / big_table_years[p_i]
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
    bins = [0.5 * i for i in range(11)] + [99]
    def plot_pdf(data, plt, color):
        kde = gaussian_kde(data)
        hist = np.histogram(data, bins=bins)[0]
        plt.plot(t.bins2labels(bins), hist / np.sum(hist), color=color)
    diff_mode = False
    #setup data
    filenames = [x['filename'] for x in main_dict.values()]
    ratios = [x['ratio'] for x in main_dict.values()]
    index = [i + 1 for i in range(len(filenames))]
    df_n = pd.DataFrame({'core index': pd.Series(index, index=filenames), '1980/1850': pd.Series(ratios, index=filenames)}, index=filenames)
    for era in ['pi', 'pd']:
        sorted_keys = list(main_dict[filenames[0]].keys())
        sorted_keys.sort()
        '''if era == 'pd':
            sorted_keys = list(reversed(sorted_keys))'''
        cmap = colormaps['Greys']
        color_list = list(reversed([cmap(i) for i in range(cmap.N)][51::64]))
        fix, ax = plt.subplots(figsize=(4, 2), dpi=300)
        #plt.xticks(ticks=bins)
        if era == 'pi':
            i = 0
            plot_pdf(ratios, plt, color_list[2])
            legend_elements = pd.Series(data=[Patch(facecolor=color_list[2], label='1980/1850')], index=[1850 if era == 'pd' else 1980])
        else:
            i = 0
            plot_pdf(ratios, plt, color_list[2])
            legend_elements = pd.Series(data=[Patch(facecolor=color_list[2], label='1980/1850')], index=[1850 if era == 'pd' else 1980])
        for key in sorted_keys:
            if '/' in key and key != '1980/1850' and key != '1980/1950':
                key_pd, key_pi = key.split('/')
                if (era == 'pi' and key_pi == '1850') or (era == 'pd' and key_pd == '1980'):
                    ratio = pd.Series([x[key] for x in main_dict.values()], index=filenames)
                    if diff_mode:
                        diff = ratio.sub(df_n['1980/1850']).div(ratio)
                        df_n.insert(len(df_n.columns), key, diff)
                    else:
                        df_n.insert(len(df_n.columns), key, ratio)
                    plot_pdf(ratio, plt, color_list[i])
                    legend_elements.loc[int(key.split('/')[0 if era == 'pi' else 1])] = (Patch(facecolor=color_list[i], label=key))
                    i += 1
                    if era == 'pd' and i == 2:
                        i += 1
        legend_elements = legend_elements.sort_index()
        ax.legend(handles=list(legend_elements), loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xticks(rotation=-60)
        plt.xlabel('PD/PI')
        plt.ylabel("Probability")
        plt.savefig('figures/ice-cores/test-' + era + '-pdfs.png', bbox_inches='tight', pad_inches=0.0, dpi=300)
        plt.close()
    #setup color scale
    cmap = colormaps['BrBG_r']
    c_norm = Normalize(vmin=0, vmax=2) if not diff_mode else Normalize(vmin=-1, vmax=1)
    #plot with color
    vals = np.vectorize(lambda a : round(a, 2))(df_n.to_numpy())
    ax.axis('off')
    colors = cmap(c_norm(vals))
    colors[:,0,:] = [1, 1, 1, 1] #make first column white
    table = plt.table(cellText=vals, colLabels=df_n.columns, loc='center', cellColours=colors)
    table.auto_set_font_size(False)
    table.set_fontsize(3)
    #set column widths
    cell_dict = table.get_celld()
    for row in range(len(vals) + 1):
        for c in range(len(vals[0])):
            cell_dict[(row, c)].set_width(0.1)
    plt.savefig('figures/ice-cores/test-big-table-pd-comparison.png', bbox_inches='tight', pad_inches=0.0, dpi=300)
    #save table as csv
    df_n = df_n.drop(['1900/1850', '1950/1850', '1980/1750', '1980/1800', '1980/1900'], axis=1)
    #df_n['Site' 'Lat', 'Lon', 'Elevation (m above sea lvl)', 'Publication Abbreviation', 'Data Source'] = [None] * len(df_n.index)
    for index, row in p.iterrows():
        for i in range(row['n_cores']):
            filename = row['First Author'].lower() + '-' + str(row['Year']) + '-' + str(i + 1) + '.csv'
            if (filename in exclude):
                continue
            df_n.loc[filename, 'Publication Abbreviation'] = row['First Author'] + ' et al. (' + str(row['Year']) + ')'
            df_n.loc[filename, 'Data Source'] = row['Data Link']
            if math.isnan(row['N']) or math.isnan(row['E']):
                temp = p_dup.loc[p_dup['Filename'] == filename].squeeze(axis=0)
                df_n.loc[filename, 'lat'] = temp['Lat']
                df_n.loc[filename, 'lon'] = temp['Lon']
                df_n.loc[filename, 'Site/Abbreviation'] = temp['Abbreviation']
            else:
                df_n.loc[filename, 'lat'] = row['N']
                df_n.loc[filename, 'lon'] = row['E']
                df_n.loc[filename, 'Site/Abbreviation'] = row['Abbreviation']
            #df_n.loc[filename, 'Elevation (m above sea lvl)'] = row['Elevation (m above sea lvl)']
    df_n.to_csv('data/manuscript-ice-core-table.csv')
elif (inp == 'big-table'): #make table comparing individual models
    #setup cmip6 data
    filenames = [x['filename'] for x in main_dict.values()]
    ratios = [x['ratio'] for x in main_dict.values()]
    index = [i + 1 for i in range(len(filenames))]
    df = pd.DataFrame({'Index': pd.Series(index, index=filenames), 'filename': pd.Series(filenames, index=filenames), 'Ice Core': pd.Series(ratios, index=filenames)}, index=filenames)
    df = df.drop(['filename'], axis=1)
    cmip_binned = pd.read_csv('data/model-ice-depo/cmip6/drybc-25.csv').drop(['Unnamed: 0', 'model'], axis=1).T.mean(axis=1).rename('CMIP6')
    #cmip_binned.columns = cmip_binned.loc['model']
    #cmip_binned = cmip_binned.drop(['model'])
    #cmip_binned = cmip_binned.join(cmip_binned.mean(axis=1).rename('CMIP6'))
    cesm = pd.read_csv('data/model-ice-depo/cesm-wetdry/cesm.csv').drop(['Unnamed: 0', 'model'], axis=1).T.mean(axis=1).rename('CESM2')
    lens = pd.read_csv('data/model-ice-depo/lens/lens.csv')
    lens = lens.rename(columns={"Unnamed: 0": "Restart"}).T
    lens.columns = lens.loc['Restart']
    lens = lens.drop(['Restart'])
    lens = lens.drop(['pi'], axis=1)
    #sort column names
    lens = lens.reindex(sorted(lens.columns), axis=1)
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
    df = df.join(cesm, how='outer')
    df = df.join(lens.mean(axis=1).rename('LENS'), how='outer')
    df = df.join(lens, how='outer')
    df = df[df['Index'].notna()]
    df = df.sort_values('Index')
    #calcualte n within
    within_cesm = pd.Series()
    within_ice = pd.Series()
    for c_name in df.columns:
        column = df[c_name]
        if c_name == 'Index':
            within_cesm.at[c_name] = -1
            within_ice.at[c_name] = -1
        else:
            within_cesm.at[c_name] = (np.abs(column - df['CESM2']) < 0.25).sum()
            within_ice.at[c_name] = (np.abs(column - df['Ice Core']) < 0.25).sum()
    df.loc['n near ice core'] = within_ice
    df.loc['n near CESM2'] = within_cesm
    #setup color scale
    cmap = colormaps['BrBG_r']
    c_norm = Normalize(vmin=0, vmax=2)
    #calcuate red cells and setup colors
    vals = np.vectorize(lambda a : round(a, 3))(df.to_numpy())
    red_mask = np.zeros(np.shape(df))
    for column in [str(n + 18) for n in range(18)] + ['LENS', 'CESM2', 'CMIP6']:
        for i in range(len(df[column]) - 2):
            if np.abs(df['Ice Core'].iloc[i] - df[column].iloc[i]) < 0.25:
                red_mask[i][df.columns.get_loc(column)] = 1
    fix, ax = plt.subplots(figsize=(4, 2), dpi=300)
    ax.axis('off')
    colors = cmap(c_norm(vals))
    colors[:,0,:] = [1, 1, 1, 1] #make first column white
    colors[-2:,:,:] = [1, 1, 1, 1] #make last 2 rows white
    for il in range(len(colors)):
        for ic in range(len(colors[il])):
            if red_mask[il][ic]:
                colors[il][ic] = [0.8, 0.2235, 0.486, 1]
    vals = vals.tolist()
    vals[37][0] = 'n near Ice Core'
    vals[38][0] = 'n near CESM2'
    rename_cols = {str(i): 'LENS' +str(i) for i in range(18,36)}
    rename_cols['Index'] = 'Ice Core Index'
    table = plt.table(cellText=vals, colLabels=df.rename(columns=rename_cols).columns, loc='center', cellColours=colors, colWidths=[0.2] + [0.1] * (len(df.columns) - 1))
    table.auto_set_font_size(False)
    table.set_fontsize(5)
    df = df[df['Index'] < 0]
    df.reindex(sorted(df.columns), axis=1).to_csv('data/big-table-within.csv')
    plt.savefig('figures/ice-cores/test-big-table-cmip-models.png', bbox_inches='tight', pad_inches=0.0, dpi=300)
elif (inp == 'p'): #Plotly
    fig = px.scatter_geo(final_pd, lat='lat', lon='lon', hover_name='ratio', title='PD/PI Ratios')
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
        ax.add_feature(cartopy.feature.COASTLINE, edgecolor='grey', linewidth=0.5)

        #elevation
        elev = Dataset('data/elevation-land-only.nc') #from https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2/ETOPO2v2-2006/ETOPO2v2c/netCDF/
        elev_lon = elev['lon'][:]
        elev_lat = elev['lat'][:]
        elev_z = np.transpose(elev['land_elev'][:])
        mesh = plt.pcolormesh(elev_lon, elev_lat, elev_z, cmap=colormaps['Greys'], vmin=0, transform=cartopy.crs.PlateCarree())

        #setup color scale
        cmap = colormaps['BrBG']#inferno
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
            modification = ""
            if key == 'sigl-2018-1.csv':
                modification = ",36"
            if key not in ('eichler-2023-1.csv'):
                #split european cores in half
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
            pass
            #rcParams.update({'font.size': 10})
            plt.colorbar(mappable=sm, label="1980/1850 BC Concentration", orientation="horizontal", ax=ax)
            #rcParams.update({'font.size': 7})
            #plt.colorbar(mappable=mesh.colorbar, label="Elevation (m)", orientation="vertical")
        
        #remove border from plot
        ax.patch.set_visible(False)
        ax.axis('off')

        #patches
        '''for patch in patches.values():
            ax.add_patch(Rectangle(xy=[patch[0], patch[1]], width=patch[2], height=patch[3], facecolor=patch[4] + '50', edgecolor=patch[4],transform=cartopy.crs.PlateCarree()))'''

        plt.savefig('figures/ice-cores/testmap-' + projection + '.png', bbox_inches='tight', pad_inches=0.0)
        #plt.show()
    s = set()
    for pub in str(index_name_map).replace('}', '').replace('{', '').split(','):
        s.add(pub[t.find_nth(pub, " ", 2) + 2:t.find_nth(pub, "-", 2)])
    print(len(s), "unique ice core pubs")
elif (inp == 'c-anthro'):
    #setup
    dpi = 300
    fig, ax = plt.subplots(dpi=dpi, subplot_kw={'projection': cartopy.crs.Robinson()})
    ax.add_feature(cartopy.feature.COASTLINE, edgecolor='grey', linewidth=0.5)
    
    #elevation
    elev = Dataset('data/elevation-land-only.nc') #from https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2/ETOPO2v2-2006/ETOPO2v2c/netCDF/
    elev_lon = elev['lon'][:]
    elev_lat = elev['lat'][:]
    elev_z = np.transpose(elev['land_elev'][:])
    mesh = plt.pcolormesh(elev_lon, elev_lat, elev_z, cmap=colormaps['Greys'], vmin=0, transform=cartopy.crs.PlateCarree())

    #patches
    colors = {k: l[-2] for k, l in patches.items()}
    colors['South America'] = colors['South ZAmerica']
    colors['USA'] = colors['North America']
    colors['Alaska'] = colors['Arctic']
    colors['Greenland'] = colors['North Greenland']
    anthro_boxes = json.load(open('data/emission-boxes.json'))
    for region, boxes in anthro_boxes.items():
        for box in boxes:
            ax.add_patch(Rectangle(xy=[box[2], box[0]], width=np.abs(box[3]-box[2]), height=np.abs(box[1]-box[0]), edgecolor=colors[region], facecolor='#00000000', zorder=10, transform=cartopy.crs.PlateCarree()))
    plt.savefig('figures/ice-cores/test-anthro-map-.png', bbox_inches='tight', pad_inches=0.0)

elif (inp == 'l'):
    #setup data:
    models = {
        'LENS': {
            'dataset': pd.read_csv('data/model-ice-depo/lens/lens.csv').loc[pd.read_csv('data/model-ice-depo/lens/lens.csv')['Unnamed: 0'] != 'pi'],
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': model_colors['LENS'],#IBM Design library's colorblind pallete
            },
        'LENS-18': {
            'dataset': pd.read_csv('data/model-ice-depo/lens/lens.csv').loc[pd.read_csv('data/model-ice-depo/lens/lens.csv')['Unnamed: 0'] == '18'],
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': model_colors['LENS'],#IBM Design library's colorblind pallete
            },
        'LENS-25': {
            'dataset': pd.read_csv('data/model-ice-depo/lens/lens.csv').loc[pd.read_csv('data/model-ice-depo/lens/lens.csv')['Unnamed: 0'] == '25'],
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': model_colors['LENS'],#IBM Design library's colorblind pallete
            },
        'LENS-Bias': {
            'dataset': pd.read_csv('data/model-ice-depo/lens/lens-bias.csv'),
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': model_colors['CESM'],#IBM Design library's colorblind pallete
            },
        'CESM': {
            'dataset': pd.read_csv('data/model-ice-depo/cesm-wetdry/cesm.csv').drop(['model'], axis=1).mean(axis=0),
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': model_colors['CESM'],
            },
        'CMIP6': {
            #'dataset': pd.read_csv('data/model-ice-depo/cmip6/drybc.csv').loc[pd.read_csv('data/model-ice-depo/cmip6/drybc.csv')['model'] == 'CESM2'],
            'dataset': pd.read_csv('data/model-ice-depo/cmip6/drybc-25.csv').drop(['model'], axis=1).mean(axis=0), #pd.read_csv(os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'cmip6', 'alt-method.csv')),
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': model_colors['CMIP6'],
            },
        'Ice Core': {
            'dataset': main_dict,
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': model_colors['Ice Core'],
            },
        'CESM-SOOTSN': { #changed to intentionally wrong input file
            #'dataset': pd.read_csv('data/model-ice-depo/cesm-sootsn/sootsn.csv').loc[pd.read_csv('data/model-ice-depo/cesm-sootsn/sootsn.csv')['model'] == 'CESM2'],
            'dataset': pd.read_csv('data/model-ice-depo/cesm-sootsn/sootsn.csv').drop(['model'], axis=1).replace(to_replace='--', value=np.nan).drop([0,2], axis=0).mean(axis=0),
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': model_colors['CESM-SOOTSN'],
            },
        'loadbc': {
            'dataset': pd.read_csv('data/model-ice-depo/loadbc/loadbc.csv').drop(['model'], axis=1).mean(axis=0),#.loc[pd.read_csv('data/model-ice-depo/loadbc/loadbc.csv')['model'] == 'CESM2'],
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': model_colors['CMIP6'],
            },
        'mmrbc': {
            'dataset': pd.read_csv('data/model-ice-depo/mmrbc/mmrbc.csv').drop(['model'], axis=1).drop([1,2,3], axis=0).mean(axis=0),
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': model_colors['LENS'],
            }
    }
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
    order_of_columns = []
    #get model means
    counter = 0
    for col_name in lens_avg.columns:
        if col_name in dont_use:
            continue
        order_of_columns.append(col_name)
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
    print('ice core mean', np.mean(bar_means['CESM']))
    print('ice core mean', np.mean(bar_means['CMIP6']))
    print('ice core mean', np.mean(bar_means['LENS']))
    print('ice core mean', np.mean(bar_means['Ice Core']))
    print('ice core std', np.std(bar_means['Ice Core']))
    print('ice core min', np.min(bar_means['Ice Core']))
    print('ice core max', np.max(bar_means['Ice Core']))
    #print(np.mean(bar_means['CMIP6']), np.mean(bar_means['LENS']), np.mean(bar_means['CESM']))
    bar_labels, filenames, bar_means['LENS'], bar_means['Ice Core'], bar_means['CESM'], bar_means['CMIP6'], bar_stds['LENS'], bar_stds['Ice Core'], bar_stds['CESM'], bar_stds['CMIP6'], bar_means['CESM-SOOTSN'], bar_stds['CESM-SOOTSN'], background_colors = zip(*sorted(list(zip(bar_labels, filenames, bar_means['LENS'], bar_means['Ice Core'], bar_means['CESM'], bar_means['CMIP6'], bar_stds['LENS'], bar_stds['Ice Core'], bar_stds['CESM'], bar_stds['CMIP6'], bar_means['CESM-SOOTSN'], bar_stds['CESM-SOOTSN'], background_colors))))
    #bar_labels, bar_means['Ice Core'], bar_stds['Ice Core'], bar_means['LENS-S1'], bar_stds['LENS-S1'], bar_means['LENS-S4'], bar_stds['LENS-S4'], bar_means['LENS-S8'], bar_stds['LENS-S8'], background_colors = zip(*sorted(list(zip(bar_labels, bar_means['Ice Core'], bar_stds['Ice Core'], bar_means['LENS-S1'], bar_stds['LENS-S1'], bar_means['LENS-S4'], bar_stds['LENS-S4'], bar_means['LENS-S8'], bar_stds['LENS-S8'], background_colors))))
    #bar_labels, bar_means['Ice Core'], bar_stds['Ice Core'], bar_means['LENS-LV30'], bar_stds['LENS-LV30'], bar_means['LENS-LV29'], bar_stds['LENS-LV29'], bar_means['LENS-LV28'], bar_stds['LENS-LV28'], background_colors = zip(*sorted(list(zip(bar_labels, bar_means['Ice Core'], bar_stds['Ice Core'], bar_means['LENS-LV30'], bar_stds['LENS-LV30'], bar_means['LENS-LV29'], bar_stds['LENS-LV29'], bar_means['LENS-LV28'], bar_stds['LENS-LV28'], background_colors))))
    #bar_labels, bar_means['CMIP6 PD'], bar_stds['CMIP6 PD'], bar_means['CMIP6 PI'], bar_stds['CMIP6 PI'], background_colors = zip(*sorted(list(zip(bar_labels, bar_means['CMIP6 PD'], bar_stds['CMIP6 PD'], bar_means['CMIP6 PI'], bar_stds['CMIP6 PI'], background_colors))))
    #Remove Duplicate Region Labels
    region_lables = list(map(lambda x: x.split('-')[0], bar_labels))
    if len(sys.argv) == 2: #or sys.argv[2] == 'var':
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
                color = models_colors[model_key]
                #plot mean bars
                row = {'Model': model_key}
                for i in range(len(transition_indexes) - 1):
                    trans_i = transition_indexes[i]
                    next_i = transition_indexes[i + 1]
                    region = bar_labels[trans_i].split('-')[0]
                    y_pos = np.mean([x for x in bar_means[model_key][trans_i:next_i] if x != 0])
                    x_start = (plt.getp(bars[trans_i], 'x') - offset + width + 0.07 - (trans_i / len(bar_labels)) * (1 - width * n_models)) / len(bar_labels)
                    x_end   = x_start + ((next_i - trans_i) - width - 0.07) / len(bar_labels) #(x_start + width * 4 * (next_i - trans_i)) / 36
                    plt.axhline(y_pos, c=color, xmin=x_start, xmax=x_end, linewidth=0.75)
                    row[region] = str(round(y_pos - np.mean(bar_means['Ice Core'][trans_i:next_i]), 2))
                if model_key not in models_in_csv:
                    csv_dict.append(row)
                    models_in_csv.add(model_key)
                multiplier += 1
                good_error_indexes = []
                for i in range(len(bar_means[model_key])):
                    if bar_means[model_key][i] != 0:
                        good_error_indexes.append(i)
                error_bar_data = [1 if x == 0 else x for x in bar_means[model_key]]
                plt.errorbar(x + offset, error_bar_data, yerr=bar_stds[model_key], errorevery=good_error_indexes, fmt='.', color=color, elinewidth=0.5, capthick=0.5)
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
                leg.legendHandles[i].set_color(color)
            for a in plt.gcf().get_axes():
                for i in range(len(bar_labels)):
                    filename = bar_labels[i].split('-')[0]
                    color = patches[filename][-1]
                    a.get_xticklabels()[i].set_color(color)
            plt.savefig('figures/ice-cores/test4' + list(sub.keys())[1] + '.png', dpi=300)
        plt.close()
    if len(sys.argv) >= 3 and sys.argv[2] == 'test-table-color':
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
    #plot new main figure
    elif len(sys.argv) >= 3 and sys.argv[2] == 'ln':
        filenames = [x['filename'] for x in main_dict.values()]
        ratios = [x['ratio'] for x in main_dict.values()]
        index = [i + 1 for i in range(len(filenames))]
        df = pd.DataFrame({
            'core index': pd.Series(index, index=filenames),
            'region': pd.Series([filename_region[i] for i in filenames], index=filenames),
            'Ice Core': pd.Series(ratios, index=filenames),
            #'loadbc': pd.Series(bar_means['loadbc'], index=order_of_columns),
            #'mmrbc': pd.Series(bar_means['mmrbc'], index=order_of_columns),
            'CESM': pd.Series(bar_means['CESM'], index=order_of_columns),
            #'CESM-SOOTSN': pd.Series(bar_means['CESM-SOOTSN'], index=order_of_columns),
            'CMIP6': pd.Series(bar_means['CMIP6'], index=order_of_columns),
            'LENS': pd.Series(bar_means['LENS'], index=order_of_columns),
            #'LENS-18': pd.Series(bar_means['LENS-18'], index=order_of_columns),
            #'LENS-25': pd.Series(bar_means['LENS-25'], index=order_of_columns),
            #'LENS-Bias': pd.Series(bar_means['LENS-Bias'], index=order_of_columns)
            }, index=filenames)
        #reformat data
        region_filename = t.invert_dict_list(filename_region)
        #sorted_regions = list(region_filename.keys())
        #sorted_regions.sort()
        sorted_regions = ['Arctic', 'North Greenland', 'South Greenland', 'Antarctica', 'Africa', 'Asia', 'Europe', 'North America', 'South ZAmerica']
        region_filename = {i: region_filename[i] for i in sorted_regions}
        data = {model: [] for model in df.columns}
        del data['core index'], data['region']
        for model in data.keys():
            for region in region_filename.keys():
                data[model].append(df[df['region'] == region][model])
        #plot
        fig, ax = plt.subplots(layout='constrained')
        x = np.arange(len(region_filename.keys()))
        multiplier = 0
        width = 0.2
        #bar_labels = list(region_filename.keys())
        bar_labels = ['Arctic', 'North Greenland', 'South Greenland', 'Antarctica', 'Africa', 'Asia', 'Europe', 'North America', 'South ZAmerica']
        bar_colors = [patches[s][-1] + '30' for s in bar_labels]
        bar_width = 1
        max_box_height = df.drop('core index', axis=1).max(numeric_only=True).max()
        box_heights = []
        for model in df.keys():
            if model in ['core index', 'region']:
                continue
            c = model_colors[model]
            ca = c + '90'
            offset = (width) * multiplier
            for i in range(len(data[model])):
                pos = x[i] + offset
                if len(data[model][i]) != 2:
                    bplot = ax.boxplot(data[model][i], widths=width, positions=[pos], patch_artist=True, boxprops=dict(facecolor=ca, color=c, linewidth=0), capprops=dict(color=c), medianprops=dict(color='black', linewidth=0), flierprops=dict(color=c, markerfacecolor=c, markeredgecolor=c, marker= '.'), whiskerprops=dict(color=c), showfliers=False, showcaps=False, showmeans=False, showbox=True)
                    for median in bplot['medians']:
                        #median.set(color='k', linewidth=1.5,)
                        med_x, med_y = median.get_data()
                        xn = (med_x - (med_x.sum()/2.)) * 0.5 + (med_x.sum()/2.)
                        plt.plot(med_x, med_y, color="k", linewidth=1, solid_capstyle="butt", zorder=4)
                    #box_heights += [item.get_ydata()[1] for item in bplot['whiskers']]
                else:
                    plt.plot(2 * [pos], data[model][i], c=c, linewidth=1)
                    #box_heights += [data[model][i]]
            for i in range(len(data[model])):
                if len(data[model][i]) > 1:
                    plt.scatter(len(data[model][i]) * [x[i] + offset], data[model][i], c=c, s=8)
                    x_color = 'black'
                else:
                    x_color = c
                plt.scatter(x[i] + offset, np.mean(data[model][i]), c=x_color, s=30, marker='x', zorder=2.5)
            multiplier += 1
        #bars = ax.bar(x + width * 1.5, np.max(box_heights) + 0.1, bar_width, color=bar_colors, zorder=0)
        y_ticks = [0.3, 0.5, 1, 2, 4, 7, 10, 20]
        max_box_height = np.max([max_box_height + 0.1] + y_ticks)
        spacing = width * 1.5
        bars = ax.bar(x + spacing, max_box_height, bar_width - 0.03, color=bar_colors, zorder=0)
        bar_labels[bar_labels.index('South ZAmerica')] = 'South America'
        plt.xticks(rotation=90)
        ax.set_yscale('log')
        ax.set_xticks(x + spacing, bar_labels)
        ax.set_xlim([x[0] + spacing - bar_width / 2, x[-1] + spacing + bar_width / 2])
        ax.set_ylim([0.2, max_box_height])
        ax.set_yticks(y_ticks)
        ax.set_ylabel("1980/1850 BC Ratio")
        ax.set_xlabel("Region")
        ax.get_yaxis().set_major_formatter(ScalarFormatter())
        #manually setup legend
        legend_handels = []
        legend_names = {'CMIP6': 'CMIP6 (n=8)', 'CESM': 'CESM (n=1)', 'LENS': 'LENS (n=18)', 'LENS-Bias': 'LENS-Bias', 'Ice Core': 'Ice Core', 'mmrbc': 'mmrbc', 'LENS-18': 'LENS-18', 'LENS-25': 'LENS-25'}
        for model in data.keys():
            legend_handels.append(Patch(label=legend_names[model], facecolor=model_colors[model]))
        ax.legend(handles=legend_handels, loc=9, bbox_to_anchor=(-0.05, -0.15))
        #axis labels colors
        for a in plt.gcf().get_axes():
            for i in range(len(bar_labels)):
                label = bar_labels[i].replace('South America', 'South ZAmerica')
                color = patches[label][-1]
                a.get_xticklabels()[i].set_color(color)
        plt.savefig('figures/ice-cores/test-new-main.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
    #plot variable figure
    if len(sys.argv) >= 3 and sys.argv[2] == 'var':
        print('var comparison boxplot')
        region2region = {'North Pole': 'Arctic', 'South Pole': 'Antarctica', 'Alpine': 'Africa'}
        sub = dict(
            [['Ice Core', bar_means['Ice Core']],
                ['loadbc', bar_means['loadbc']],
                ['CESM', bar_means['CESM']],
                ['mmrbc', bar_means['mmrbc']],
                ['CESM-SOOTSN', bar_means['CESM-SOOTSN']]]
            )
        #get regional means
        new_regions = {
            'North Pole': ['North Greenland', 'South Greenland', 'Arctic'],
            'South Pole': ['Antarctica'],
            'Alpine': ['Africa', 'Asia', 'Europe', 'North America', 'South ZAmerica'],
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
                data[model].append(rdf[rdf['region'] == region][model].dropna())
        #plot
        fig, ax = plt.subplots(layout='constrained')
        x = np.arange(len(list(new_regions.keys())))
        multiplier = 0
        width = 0.2 * 4 / 5
        bar_labels = list(new_regions.keys())
        bar_colors = [patches[region2region[s]][-1] + '30' for s in bar_labels]
        bar_width = 1
        box_heights = []
        for_json = {}
        for model in data.keys():
            for_json[model] = []
            c = model_colors[model]
            ca = c + '90'
            offset = width * multiplier
            bplot = ax.boxplot(data[model], widths=width, positions=x + offset - width / 2, patch_artist=True, boxprops=dict(facecolor=ca, color=c, linewidth=0), capprops=dict(color=c), medianprops=dict(color='black'), flierprops=dict(color=c, markerfacecolor=c, markeredgecolor=c, marker= '.'), whiskerprops=dict(color=c), showfliers=False, showcaps=False, showmeans=False, showbox=True)
            for i in range(len(data[model])):
                for_json[model].append(np.median(data[model][i]))
                plt.scatter(len(data[model][i]) * [x[i] + offset - width / 2], data[model][i], c=c, s=8)
                x_color = 'black'
                plt.scatter(x[i] + offset - width / 2, np.mean(data[model][i]), c=x_color, s=30, marker='x', zorder=2.5)
            #bplot = ax.boxplot(data[model], widths=width, positions=x+offset, patch_artist=True, boxprops=dict(facecolor=c, color=c), capprops=dict(color=c), medianprops=dict(color='black'), flierprops=dict(color=c, markerfacecolor=c, markeredgecolor=c, marker= '.'), whiskerprops=dict(color=c))
            box_heights += [item.get_ydata()[1] for item in bplot['whiskers']]
            multiplier += 1
        bars = ax.bar(x + width * 1.5, rdf['Ice Core'].max() + 0.1, bar_width, color=bar_colors, zorder=0)
        ax.set_xticks(x + width * 1.5, bar_labels)
        ax.set_xlim([x[0] + width * 1.5 - bar_width / 2, x[-1] + width * 1.5 + bar_width / 2])
        ax.set_yticks([x for x in range(1, 11)])
        ax.set_ylabel("1980/1850 Ratio")
        ax.set_xlabel("Region")
        ax.get_yaxis().set_major_formatter(ScalarFormatter())
        ax.set_ylim([0, rdf['Ice Core'].max() + 0.1])
        #manually setup legend
        legend_handels = []
        leg_dict = {'Ice Core': 'Ice Core',
                    'loadbc': 'CESM2 BC in air column',
                    'CESM': 'CESM2 BC deposition to snow',
                    'CESM-SOOTSN': 'CESM2 BC in snow',
                    'mmrbc': 'CESM2 BC in surface air'}
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
        for_json['regions'] = bar_labels
        with open('data/var-medians.json', 'w') as f:
            json.dump(for_json, f)
        plt.savefig('figures/ice-cores/test-var.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
    #plot nh anthro emissions
    elif len(sys.argv) >= 3 and sys.argv[2] == 'anth':
        filenames = [x['filename'] for x in main_dict.values()]
        ratios = [x['ratio'] for x in main_dict.values()]
        index = [i + 1 for i in range(len(filenames))]
        new_region_map = filename_region
        new_region_map['zhang-2024-12.csv'] = 'Alaska'
        new_region_map['zhang-2024-11.csv'] = 'Alaska'
        df = pd.DataFrame({
            'core index': pd.Series(index, index=filenames),
            'region': pd.Series([new_region_map[i] for i in filenames], index=filenames),
            'Ice Core': pd.Series(ratios, index=filenames),
            }, index=filenames)
        anth_df = pd.read_csv(os.path.join('data', 'model-ice-depo', 'anthro-ratios-new.csv'))
        anth_df['Method'] = anth_df['Unnamed: 0'].apply(lambda s: s.split(':')[0])
        anth_df = anth_df.set_index('Unnamed: 0').rename(columns={'USA': 'North America'})
        #reformat data
        region_filename = t.invert_dict_list(filename_region)
        #sorted_regions = list(region_filename.keys())
        #sorted_regions.sort()
        sorted_regions = ['Alaska', 'Africa', 'Asia', 'Europe', 'North America', 'South ZAmerica']
        region_filename = {i: region_filename[i] for i in sorted_regions}
        data = {model: [] for model in df.columns}
        del data['core index'], data['region']
        for region in region_filename.keys():
            for model in data.keys():
                data[model].append(df[df['region'] == region][model])
        #add anth methods
        bad_methods = {'Hoesly+MarlePD', 'Hoesly+MarlePI'}
        for method in anth_df['Method']:
            if method not in bad_methods:
                data[method] = []
        used_methods = set()
        for method in anth_df['Method']:
            if method in bad_methods:
                continue
            if not method in used_methods:
                for region in region_filename.keys():
                    used_methods.add(method)
                    data[method].append(anth_df[anth_df['Method'] == method][region.replace('ZAmerica', 'America')])
        regionless_data = {k: np.mean([x for xs in v for x in xs]) for k, v in data.items()}
        del regionless_data['Ice Core']
        #plot simplified regionless
        red_ice_df = df.loc[df['core index'].isin({33, 36, 37, 31, 5, 29, 1, 35, 32, 15})]
        ice_core_min_max = (np.min(red_ice_df['Ice Core']), np.max(red_ice_df['Ice Core']), np.median(red_ice_df['Ice Core']))
        legend_names = {'Ice Core': 'Ice Core', 'Hoesly': 'Anthropogenic', 'Marle': 'Biomass Burning', 'Hoesly+MarlePI': 'Anth+$Bio_{PI}$', 'Hoesly+MarlePD/PI': 'Anth+$Bio_{PD/PI}$'}
        anth_model_map = {'Hoesly': 'CESM', 'Marle': 'CMIP6', 'Hoesly+MarlePI': 'LENS', 'Hoesly+MarlePD': 'mmrbc', 'Hoesly+MarlePD/PI': 'CESM-SOOTSN', 'Ice Core': 'Ice Core'}
        colors = [model_colors[anth_model_map[model]] for model in regionless_data.keys()]
        fig, ax = plt.subplots()#2, layout='constrained')
        ax.scatter([legend_names[k] for k in regionless_data.keys()], regionless_data.values(), c=colors, s=60, marker='x')
        #ax.add_patch(plt.Rectangle((-0.2, ice_core_min_max[0]), 3.4, ice_core_min_max[1], color='#1177bc90', linewidth=0, zorder=0))
        for index, row in red_ice_df.iterrows():
            ax.plot((-0.2, 3.4), (row['Ice Core'], row['Ice Core']), color='#1177bc90')
        ax.text(0, ice_core_min_max[0] * 1.1, 'Min Ice Core')
        ax.text(0, ice_core_min_max[1] * 1.1, 'Max Ice Core')
        #ax.text(0, ice_core_min_max[2] * 1.1, 'Median Ice Core')
        ax.set_xlim((-0.2, 2.2))
        ax.set_yscale('log')
        #plt.xticks(rotation=90)
        #ax[0].set_ylim((20, 50000))
        #ax[1].scatter(regionless_data.keys(), regionless_data.values())
        #ax[1].set_yscale('log')
        #ax[1].set_ylim([0.2, 20])
        print('saved as figures/ice-cores/test-anth-simple.png')
        plt.savefig('figures/ice-cores/test-anth-simple.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
        #plot top extremes
        x = np.arange(len(region_filename.keys()))
        multiplier = 0
        width = 0.2
        spacing = width * 2
        bar_width = 1
        bar_labels = list(region_filename.keys())
        bar_colors = [patches[s][-2] + '30' for s in bar_labels]
        fig, ax = plt.subplots(2, layout='constrained')
        extremes = json.load(open('data/extreme-anth-values.json'))
        for model in data.keys():
            offset = (width) * multiplier
            c = model_colors[anth_model_map[model]]
            for i in range(len(data[model])):
                if extremes[model][i] > 20:
                    ax[0].scatter(x[i] + offset + 0.1, extremes[model][i], c=c, s=30, marker='x', zorder=2.5)
            multiplier += 1
        ax[0].bar(x + spacing + 0.1, 50000, bar_width, color=bar_colors, zorder=0)
        ax[0].set_yscale('log')
        ax[0].set_xlim((0, 6))
        ax[0].set_ylim((20, 50000))
        ax[0].get_xaxis().set_visible(False)
        #plot main figure
        ax = ax[1]
        multiplier = 0
        max_box_height = 0
        for model in data.keys():
            for l in data[model]:
                if np.max(l) > max_box_height:
                    max_box_height = np.max(l)
        box_heights = []
        for_json = {}
        for model in data.keys():
            for_json[model] = []
            if model in ['core index', 'region']:
                continue
            c = model_colors[anth_model_map[model]]
            ca = c + '90'
            offset = (width) * multiplier
            for i in range(len(data[model])):
                pos = x[i] + offset
                for_json[model].append(np.mean(data[model][i]))
                if len(data[model][i]) != 2:
                    bplot = ax.boxplot(data[model][i], widths=width, positions=[pos], patch_artist=True, boxprops=dict(facecolor=ca, color=c, linewidth=0), capprops=dict(color=c), medianprops=dict(color='black', linewidth=0), flierprops=dict(color=c, markerfacecolor=c, markeredgecolor=c, marker= '.'), whiskerprops=dict(color=c), showfliers=False, showcaps=False, showmeans=False, showbox=True)
                    for median in bplot['medians']:
                        #median.set(color='k', linewidth=1.5,)
                        med_x, med_y = median.get_data()
                        xn = (med_x - (med_x.sum()/2.)) * 0.5 + (med_x.sum()/2.)
                        plt.plot(med_x, med_y, color="k", linewidth=1, solid_capstyle="butt", zorder=4)
                    #box_heights += [item.get_ydata()[1] for item in bplot['whiskers']]
                else:
                    plt.plot(2 * [pos], data[model][i], c=c, linewidth=1)
                    #box_heights += [data[model][i]]
            for i in range(len(data[model])):
                if len(data[model][i]) > 1:
                    plt.scatter(len(data[model][i]) * [x[i] + offset], data[model][i], c=c, s=8)
                    x_color = 'black'
                else:
                    x_color = c
                plt.scatter(x[i] + offset, np.mean(data[model][i]), c=x_color, s=30, marker='x', zorder=2.5)
            multiplier += 1
        #bars = ax.bar(x + width * 1.5, np.max(box_heights) + 0.1, bar_width, color=bar_colors, zorder=0)
        y_ticks = [0.3, 0.5, 1, 2, 4, 7, 10, 20]
        max_box_height = np.max([max_box_height + 0.1] + y_ticks)
        bars = ax.bar(x + spacing, max_box_height, bar_width, color=bar_colors, zorder=0)
        bar_labels[bar_labels.index('South ZAmerica')] = 'South America'
        for_json['regions'] = bar_labels
        plt.xticks(rotation=90)
        ax.set_yscale('log')
        ax.set_xticks(x + spacing, bar_labels)
        ax.set_xlim([x[0] + spacing - bar_width / 2, x[-1] + spacing + bar_width / 2])
        ax.set_ylim([0.2, 20])
        ax.set_yticks(y_ticks)
        ax.set_ylabel("1980/1850 BC Ratio")
        ax.set_xlabel("Region")
        ax.get_yaxis().set_major_formatter(ScalarFormatter())
        #manually setup legend
        legend_handels = []
        for model in data.keys():
            legend_handels.append(Patch(label=legend_names[model], facecolor=model_colors[anth_model_map[model]]))
        ax.legend(handles=legend_handels, loc=9, bbox_to_anchor=(-0.05, -0.15)).set_zorder(0)
        #axis labels colors
        for a in plt.gcf().get_axes():
            for i in range(len(bar_labels)):
                label = bar_labels[i].replace('South America', 'South ZAmerica')
                color = patches[label][-2]
                a.get_xticklabels()[i].set_color(color)
        plt.savefig('figures/ice-cores/test-new-main-anth.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
        plt.close()
        with open('data/extreme-anth-values.json', 'w') as f:
            json.dump(for_json, f)
        print('saved as ' + 'figures/ice-cores/test-new-main-anth.png')
    #plot antartica supersets
    elif len(sys.argv) >= 3 and sys.argv[2] == 'ant':
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
            plt.title('Distribution of Mean Ice Core PD/PI BC Ratio Across Different Antarctic Regional Boundaries')
            plt.xlabel('Mean Ice Core PD/PI BC Ratio')
            plt.ylabel('Number of Possible Antarctic Regions')
            plt.savefig('figures/ice-cores/test5' + key + '.png', bbox_inches='tight', pad_inches=0.0, dpi=300)
            plt.close()
    elif len(sys.argv) >= 3 and sys.argv[2] == 'scatter':
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
#new timeseries
elif (inp == 'nt'):
    '''if len(sys.argv) >= 3:
        sys.argv[2]
    else:
        #python3 pi-pd.py nt <path to sootsn file
        raise Exception('3 command line arguments required: python3 pi-pd.py nt <sootsn file name>')'''
    nco_model_colors = {'ice core': '#1177bc', 'CESM2': '#ed292c'} #, 'TaiESM1': '#CC397C'}#{'ice core': '#6C62E7', 'CESM2': '#EE692C'}#, 'TaiESM1': '#CC397C'}# 'sootsn': '#638FF6'}
    valid_keys_set = set(main_dict.keys())
    axis_ticks = [(i + 0.5) for i in range(1850, 1981)]
    figures = {'North Pole': [a_p, 90], 'South Pole': [-90, -60], 'Alpine': [-60, a_p]}
    model_data = {}
    for fig_name, min_max_lat in figures.items():
        plt.rc('font', size=16)
        mode = 'one-line'
        df = pd.DataFrame(index = axis_ticks)
        timeseries = []
        for csv_file in os.listdir(os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'timeseries')):
            if '.csv' == csv_file[len(csv_file) - 4: len(csv_file)] and csv_file not in ['sootsn.csv', 'TaiESM1.csv']:
                df_model = pd.read_csv(os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'timeseries', csv_file))
                df_model_adj = pd.DataFrame(index = df_model.index)
                #get data
                for filename, coords in t.get_ice_coords('data/standardized-ice-cores/index.csv', 'data/standardized-ice-cores/index-dup-cores.csv').items():
                    if filename in valid_keys_set and min_max_lat[0] <= coords[0] <= min_max_lat[1]:
                        temp_df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'standardized-ice-cores', filename)).sort_values(by=['Yr'])
                        if mode == 'all-lines':
                            timeseries.append({'x': temp_df['Yr'], 'y': temp_df['BC'], 'group': 'ice core'})
                        y = np.interp(axis_ticks, temp_df['Yr'], temp_df['BC'])
                        df[filename] = y #np.divide(y, np.max(y))
                        if not (df_model[filename].loc[0] == 0 or df_model[filename].loc[0] >= np.power(10, 20)):
                            df_model_adj[filename] = df_model[filename]
                timeseries.append({
                    'x': axis_ticks,
                    'y': df_model_adj.mean(axis=1), #np.divide(df_model_adj.mean(axis=1), np.max(df_model_adj.mean(axis=1))),
                    'group': csv_file[0: len(csv_file) - 4:]
                    })
        soot_ax_min_max = [df_model_adj.mean(axis=1).values.min(), df_model_adj.mean(axis=1).values.max()]
        ice_ax_min_max = [df.mean(axis=1).values.min(), df.mean(axis=1).values.max()]
        #plot
        fig, ax1 = plt.subplots(figsize=(8,5), layout='constrained')
        #ax1.set_zorder(ax2.get_zorder()+1)
        #ax1.patch.set_visible(False)
        ax1.plot(df.index, np.divide(df.mean(axis=1), np.max(df.mean(axis=1))), c=nco_model_colors['ice core'])
        #make hemisphere avg timeseries plot
        plot_timeseries_decomp(df.index, df.mean(axis=1), fig_name)
        for series in timeseries:
            if series['group'] not in nco_model_colors:
                nco_model_colors[series['group']] = random.choice(list(model_colors.keys()))
            alpha = 0.1 if series['group'] == 'ice core' else 1
            print(series['group'] == 'ice core')
            y = np.interp(axis_ticks, series['x'], series['y']) if series['group'] == 'ice core' else series['y']
            y = np.divide(y, np.max(y))
            color = nco_model_colors[series['group']]
            #ax = ax1 if series['group'] == 'ice core' else ax1.twinx()
            #ax1.plot(axis_ticks, y, c=color, alpha=alpha)
            #if series['group'] != 'ice core':
                #ax.spines['right'].set_position(('outward', 60 * (n_axis - 2)))
        ax1.set_ylim([0, np.max(y)])
        #ax1.set_ylabel('Normalized BC Concentration')
        #ax1.set_yticks(np.arange(0, 100.1, 100/3))
        ax1.set_ylabel('Relative Northern Hemisphere Wildfire')
        label_map = {'CESM2': 'Model', 'ice core': 'Observations'}
        ax1.legend([Line2D([0], [0], color=v, lw=1.5, label=k) for k, v in nco_model_colors.items()], [label_map[s] for s in list(nco_model_colors.keys())])#, prop={'size': 6})
        #ax1.tick_params(axis='y', labelcolor=nco_model_colors['ice core'])
        plt.xlabel("Year (CE)")
        #plt.title(fig_name + ' Ice Core vs Modeled BC Comparison')
        save_path = 'figures/ice-cores/timeseries-' + fig_name + '.png'
        plt.savefig(save_path, dpi=300) #bbox_inches='tight', pad_inches=0.0, dpi=300)
        print('saved to ' + save_path)
        plt.close()
        print(fig_name + ' n=' + str(len(df.columns)))
elif (inp == 't'): #timeseries
    df_time = pd.read_csv(os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'timeseries', 'binned-timeseries.csv'))
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
    path = os.path.join(os.getcwd(), 'data', 'model-ice-depo', 'timeseries', 'timeseries-.csv')
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
elif (inp == 'mmrbc'):
    import matplotlib.pyplot as plt

    l = [0.39889964, 0.54118013, 0.63362288, 0.41201419, 0.37622252, 0.50968623, 0.76643807, 0.92786038, 0.66107136, 0.51291454, 0.48298901, 0.52448446, 0.614811, 0.72133279, 0.8062796, 0.86228544, 0.91575783, 0.98148191, 1.0303092, 1.01354635, 0.84013784, 0.54736644, 0.31034935, 0.16389731, 0.12988052, 0.13001627]
    x = [i for i in range(len(l))]

    plt.plot(x, l)
    plt.xlabel('mmrbc level (Pa)')
    plt.ylabel('pd/pd mmrbc ratio')
    plt.title('pd/pi mmrbc at lat,lon=0,0')
    plt.show()
    plt.close()
elif (inp == 'ets'): #breakdown of how timeseries become pd/pi
    def plot_from_avg(avg_data, color, label):
        avg, center, lower, upper = avg_data
        plt.plot([lower, upper], [avg[windows[0]], avg[windows[0]]], c=color, label=label)
        return (avg[windows[0]], np.mean((lower, upper)))
    for file in ['zdanowicz-2018-1.csv']: #['chellman-2017-1.csv', 'liu-2021-3.csv', 'liu-2021-2.csv', 'matthew-2016-1.csv']: #['zhang-2024-9.csv', 'liu-2021-5.csv', 'mcconnell-2021-5.csv', 'zdanowicz-2018-1.csv']:
        df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'standardized-ice-cores', file))
        plt.plot(df['Yr'], df['BC'], c='grey', label='Raw Data')
        BC = np.flip(df['BC'].to_numpy())
        Yr = np.flip(df['Yr'].to_numpy())
        t_series_max = np.max(BC[t.nearest_search(Yr, 1850.49):t.nearest_search(Yr, 1980)]) * 1.1
        t_series_max += 0.3 if file == 'mcconnell-2021-5.csv' else 0
        #plot
        p_i, pi_center = plot_from_avg(t.simplified_avg(Yr, BC, 1850.49 + 25/2, windows), 'black', 'PI')
        p_d, pd_center = plot_from_avg(t.simplified_avg(Yr, BC, 1967.5, windows), 'black', 'PD')
        print(p_i, p_d)
        plt.bar(pi_center, t_series_max, width=windows[0], color='#CC397C50')
        plt.bar(pd_center, t_series_max, width=windows[0], color='#6C62E750')
        plt.plot([1850.49, 1850.49], [0, t_series_max], linewidth=3, color='#CC397C')
        plt.plot([1980, 1980], [0, t_series_max], linewidth=3, color='#6C62E7')
        #label
        plt.xlabel('Year (CE)')
        plt.ylabel('BC Concentration (ng/g)')
        plt.xlim([1830, 2000])
        plt.ylim([0, t_series_max])
        plt.title(file)
        legend_handels = [
            Patch(label='Raw Data', facecolor='grey'),
            Patch(label='PI', facecolor='#CC397C'),
            Patch(label='PD', facecolor='#6C62E7')
        ]
        #plt.legend(handles=legend_handels, loc='upper center', bbox_to_anchor=(0.42, 1))
        #plt.show()
        plt.savefig('figures/ice-cores/test-explain-tseries.png', dpi=300)
        plt.close()
elif (inp == 'cmpwin'): #compare which averaging window is closest to bugged method
    cmpwin_df = pd.DataFrame(columns=list(main_dict.keys()))
    c_windows = ['bugged', 'fix', 25, 11]
    for window in c_windows:
        for filename in cmpwin_df.columns:
            df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'standardized-ice-cores', filename))
            BC = np.flip(df['BC'].to_numpy())
            Yr = np.flip(df['Yr'].to_numpy())
            if window == 'bugged':
                pi_window = pd_window = 11
                pi_avg, center, lower, upper = t.get_avgs(Yr, BC, 1850.49, [pi_window])
                pd_avg, pd_center, pd_lower, pd_upper = t.get_avgs(Yr, BC, 1980, [pi_window])
            elif window == 'fix':
                pi_window = 136
                pd_window = 170#32
                pi_avg, center, lower, upper = t.simplified_avg(Yr, BC, 1948, [pi_window])
                pd_avg, pd_center, pd_lower, pd_upper = t.simplified_avg(Yr, BC, 1974, [pd_window])
            else:
                pi_window = pd_window = window
                pi_avg, center, lower, upper = t.simplified_avg(Yr, BC, 1850.49, [window])
                pd_avg, pd_center, pd_lower, pd_upper = t.simplified_avg(Yr, BC, 1980, [window])
            cmpwin_df.loc[str(window) + '-ratio', filename] = pd_avg[pd_window] / pi_avg[pi_window]
            cmpwin_df.loc[str(window) + '-pd', filename] = pd_avg[pd_window]
            cmpwin_df.loc[str(window) + '-upper', filename] = upper
            cmpwin_df.loc[str(window) + '-lower', filename] = lower
            cmpwin_df.loc[str(window) + '-center', filename] = center
            cmpwin_df.loc[str(window) + '-pd_center', filename] = pd_center
            cmpwin_df.loc[str(window) + '-dist', filename] = upper - lower
            cmpwin_df.loc[str(window) + '-pd_upper', filename] = pd_upper
            cmpwin_df.loc[str(window) + '-pd_lower', filename] = pd_lower
            cmpwin_df.loc[str(window) + '-pd_dist', filename] = pd_upper - pd_lower
    #plot correlations
    bars = []
    labels = []
    x = cmpwin_df.loc['bugged-ratio']
    x_hashed = list(x)
    '''#minimizer function
    bc_yr_map = {}
    def rsq(p_i, p_d, pi_w, pd_w):
        def get_ratio(filename):
            if filename in bc_yr_map.keys():
                BC, Yr = bc_yr_map[filename]
            else:
                temp = pd.read_csv(os.path.join(os.getcwd(), 'data', 'standardized-ice-cores', filename))
                BC = np.flip(temp['BC'].to_numpy())
                Yr = np.flip(temp['Yr'].to_numpy())
                bc_yr_map[filename] = (BC, Yr)
            return t.simplified_avg(Yr, BC, p_d, [pd_w])[0][pd_w] / t.simplified_avg(Yr, BC, p_i, [pi_w])[0][pi_w]
        return np.power(scipy.stats.linregress(x_hashed, [get_ratio(i) for i in main_dict.keys()])[2], 2)
    start = time.time()
    print(rsq(1948, 1974, 140, 172))
    print(time.time() - start)
    #exit()
    #optimize pi, pd
    max = 0
    max_params = ()
    seed = (1949, 1966, 136, 170)
    for p_i in [seed[0] + i - 3 for i in range(7)]:
        print(str(p_i + 1) + '/40, max=' + str(max), max_params)
        for p_d in [seed[1] + i - 3 for i in range(7)]:
            for pi_w in [seed[2] + i - 3 for i in range(7)]:
                for pd_w in [seed[3] + i - 3 for i in range(7)]:
                    temp = rsq(p_i, p_d, pi_w, pd_w)
                    if temp > max:
                        max = temp
                        max_params = (p_i, p_d, pi_w, pd_w)
    print('final max=' + str(max), max_params)'''
    for key in ['25-ratio', 'fix-ratio', '11-pd']:
        if key == 'fix-ratio':
            bars.append(0.0041)
            labels.append('136-pi 170-pd')
        else:
            y = cmpwin_df.loc[key]
            slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(list(cmpwin_df.loc['bugged-ratio']), list(y))
            bars.append(np.power(r_value, 2))
            labels.append(key)
    plt.bar([i for i in range(len(bars))], bars, tick_label=labels)
    plt.ylabel('R^2 correlation with bugged method')
    plt.xlabel('Averaging Method')
    plt.show()
    plt.close()
    #plot scatter
    b = list(x.drop(['thompson-2002-1.csv']))
    t = list(y.drop(['thompson-2002-1.csv']))
    plt.scatter(b, t)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(b, t)
    x = np.array([0, np.max(b)])
    y = x * slope + intercept
    plt.plot(x, y)
    plt.xlabel('bugged ratio, r^2=' + str(np.power(r_value, 2)))
    plt.ylabel('fix ratio')
    plt.show()
    plt.close()
    #plot some histograms
    for method in ['bugged', '25']:
        for prefix in ['pd', 'pi']:
            u_pre = prefix + '_' if prefix == 'pd' else ''
            print(prefix + ':')
            print('mean, median ' + method + ' dist=' + str(np.mean(cmpwin_df.loc[method + '-' + u_pre + 'dist'])) + ',' + str(np.median(cmpwin_df.loc[method + '-' + u_pre + 'dist'])))
            print('mean, median ' + method + ' center=' + str(np.mean(cmpwin_df.loc[method + '-' + u_pre + 'center'])) + ',' + str(np.median(cmpwin_df.loc[method + '-' + u_pre + 'center'])))
            print('mean, median ' + method + ' lower=' + str(np.mean(cmpwin_df.loc[method + '-' + u_pre + 'lower'])) + ',' + str(np.median(cmpwin_df.loc[method + '-' + u_pre + 'lower'])))
            print('mean, median ' + method + ' upper=' + str(np.mean(cmpwin_df.loc[method + '-' + u_pre + 'upper'])) + ',' + str(np.median(cmpwin_df.loc[method + '-' + u_pre + 'upper'])))
            plt.hist(cmpwin_df.loc[method + '-' + u_pre + 'dist'])
            plt.title(method + ' ' + prefix + ' ranges')
            plt.savefig('figures/ice-cores/test-cmpwin-' + method + '-dist' + prefix + '.png', dpi=200)
            plt.close()
            plt.hist(cmpwin_df.loc[method + '-' + u_pre + 'lower'])
            plt.title(method + ' ' + prefix + ' lower range')
            plt.savefig('figures/ice-cores/test-cmpwin-' + method + '-lower' + prefix + '.png', dpi=200)
            plt.close()
            plt.hist(cmpwin_df.loc[method + '-' + u_pre + 'upper'])
            plt.title(method + ' ' + prefix + ' upper range')
            plt.savefig('figures/ice-cores/test-cmpwin-' + method + '-upper' + prefix + '.png', dpi=200)
            plt.close()
            plt.hist(cmpwin_df.loc[method + '-' + u_pre + 'center'])
            plt.title(method + ' ' + prefix + ' center')
            plt.savefig('figures/ice-cores/test-cmpwin-' + method + '-center' + prefix + '.png', dpi=200)
            plt.close()
elif (inp == 'yawc'): #year avergeing window comparison
    def bxp_data(label, median, std):
        std *= 0.5
        item = {}
        item["label"] = label
        item["med"] = median
        item["q1"] = item["med"]
        item["q3"] = item["med"]
        item["whislo"] = median + std
        item["whishi"] = np.max([median - std, 0])
        item["fliers"] = []
        return item
    stats = [
        bxp_data('5 Year',  2.30, 2.25),
        bxp_data('10 Year', 2.07, 2.38),
        bxp_data('15 Year', 1.88, 1.90),
        bxp_data('20 Year', 1.76, 1.56),
        bxp_data('25 Year', 1.69, 1.42),
        bxp_data('30 Year', 1.75, 1.90)
    ]
    fig, ax = plt.subplots(1, 1)
    ax.bxp(stats, medianprops=dict(color='black'))
    plt.xlabel('Averaging Technique')
    plt.ylabel('Mean Ice Core PD/PI BC Ratio (wiskers span std/2)')
    plt.ylim([0, 4])
    plt.title('Comparison of Averaging Techniques for PD and PI')
    plt.savefig('figures/ice-cores/test-yawc.png', dpi=300)
elif (inp == 'tdc'):#timeseries decomposition
    region2region = {'North Pole': 'Arctic', 'South Pole': 'Antarctica', 'Alpine': 'Africa'}
    three_regions = {
            'North Pole': ['North Greenland', 'South Greenland', 'Arctic'],
            'South Pole': ['Antarctica'],
            'Alpine': ['Africa', 'Asia', 'Europe', 'North America', 'South ZAmerica'],
            }
    linewidth = 0.5
    periods = [10, 33]#[10, 25, 33]
    fig, ax = plt.subplots(1 + 2 * len(periods), 3, sharex=True)
    fig.tight_layout(rect=(0.05, 0.05, 0.95, 0.95))
    #fig.set_figheight(10)
    #fig.set_figwidth(5)
    plt.rc('font', size=5)
    reg_i = 0
    for region, subs in three_regions.items():
        color = patches[region2region[region]][-1] + '30'
        index_ex = pd.read_csv(os.path.join(os.getcwd(), 'data', 'standardized-ice-cores', 'zhang-2024-12.csv')).interpolate()['Yr']
        df_means = pd.DataFrame(index=[i for i in range(len(index_ex))], columns=['Yr'], data=index_ex)
        for i in periods:
            df_count = 0
            for ax_i in range(2 * len(periods)):
                if ax_i % 2 == 0:
                    label_s = "Seasonal, period=" + str(periods[ax_i // 2])
                    if ax_i == 0:
                        label_s = region + '\n' + label_s
                else:
                    label_s = "Trend, period=" + str(periods[ax_i // 2])
                ax[ax_i, reg_i].set_title(label_s)
            ax[-1, reg_i].set_title("Observation")
            for ax_i in range(1 + 2 * len(periods)):
                    ax[ax_i, reg_i].set_xlim([1850, 1980])
                    ax[ax_i, reg_i].tick_params(axis='both', labelsize=5)
            for file in main_dict.keys():
                if (not filename_region[file] in subs) or (file in ('ruppel-2014-1.csv', 'thompson-2002-1.csv', 'zhang-2024-12.csv', 'eichler-2023-1.csv', 'legrand-2023-1.csv')):
                    continue
                df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'standardized-ice-cores', file))
                df.interpolate(inplace=True)
                x = df['Yr']
                y = df['BC']
                if filename_region[file] in three_regions['Alpine']:
                    #print(x[len(x)- 1], file)
                    pass
                if len(x) < 2 * i:
                    continue
                res = sm.tsa.seasonal_decompose(y, period=i)
                for key, l in {'Seasonal-' + str(i): res.seasonal, 'Trend-' + str(i): res.trend, 'Observation': y}.items():
                    if df_count == 0:
                        df_means[key] = l
                    else:
                        df_means[key] = np.average([df_means[key], pd.Series(index=df_means.index, data=l)], axis=0, weights=[df_count, 1])
                df_count += 1
                for ax_i in range(2 * len(periods)):
                    if ax_i % 2 == 0:
                        ax[ax_i, reg_i].plot(x, res.seasonal, label='seasonal', color=color, linewidth=linewidth)
                    else:
                        ax[ax_i, reg_i].plot(x, res.trend, label='trend', color=color, linewidth=linewidth)
                ax[-1, reg_i].plot(x, y, label='obs', color=color, linewidth=linewidth)
        for ax_i in range(2 * len(periods)):
            observation = 'Seasonal' if ax_i % 2 == 0 else 'Trend'
            period = str(periods[ax_i // 2])
            key = observation + '-' + period
            ax[ax_i, reg_i].plot(df_means['Yr'], df_means[key], color='black', linewidth=linewidth)#color[0:-2])
            if observation == 'Seasonal':
                ax[ax_i, reg_i].set_ylim((-1.5 * np.max(df_means[key]), 1.5 * np.max(df_means[key])))
            else:
                ax[ax_i, reg_i].set_ylim((0, 1.5 * np.max(df_means[key])))
        ax[-1, reg_i].plot(df_means['Yr'], df_means['Observation'], color='black', linewidth=linewidth)
        ax[-1, reg_i].set_ylim((0, 1.5 * np.max(df_means['Observation'])))
        reg_i += 1
    path = 'figures/ice-cores/decomposition/Combined-decomp.png'
    print('saved to ' + path)
    plt.savefig(path, dpi=200)
    plt.close()
elif (inp == 'lat-plt'):#lat vs ratio greenland plot
    lats = []
    ratios = []
    for filename, region in filename_region.items():
        if 'Greenland' in region:
            data = main_dict[filename]
            lats.append(data['lat'])
            ratios.append(data['ratio'])
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(lats, ratios)
    x = [np.min(lats) + i * (np.max(lats) - np.min(lats)) / 99 for i in range(100)]
    y = np.array(x) * slope + intercept
    plt.plot(x, y)
    plt.scatter(lats, ratios)
    plt.xlabel('Lattitude (Degrees)')
    plt.ylabel('BC 1980/1850 Ratio')
    plt.title('Greenland Ratio vs Lattitude, r^2=' + str(round(np.power(r_value, 2), 3)) + ', m=' + str(round(slope, 3)))
    plt.savefig('figures/ice-cores/test-greenland-lat-ratio', dpi=300)
elif (inp == 'methods'):#bar chart for ice core analysis methods
    df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'methods-fig-data.csv'))
    #annual layer counting
    alc_df = df.pop('Annual Layer Counting Abbrv').str.get_dummies(',')
    r = [np.random.randint(-2, 2) for i in range(len(alc_df.columns))]
    x = alc_df.sum(axis=0).to_numpy() * 4 + r
    plt.bar(alc_df.columns, [2, 2, 8, 1])
    plt.savefig('figures/ice-cores/test-methods-alc.png', dpi=200)
    plt.close()
    #BC methods
    bc_df = df.pop('Methodology Abbreviation').str.get_dummies(',')
    r = [np.random.randint(-2, 2) for i in range(len(bc_df.columns))]
    x = bc_df.sum(axis=0).to_numpy() * 3 + r
    plt.bar(bc_df.columns, [7, 1 ])
    plt.savefig('figures/ice-cores/test-methods-bc.png', dpi=200)
    plt.close()
elif (inp == 'quick-lens-bias-boxplot'):
    df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'lens-bias.csv')).set_index('Unnamed: 0').replace(to_replace='--', value=np.nan)
    data = [[x['ratio'] for x in main_dict.values()]]
    labels = ['ice core']
    colors = ['#6C62E7']
    for index in df.index:
        d = [float(x) for x in df.loc[index]]
        print(np.count_nonzero(~np.isnan(d)))
        data.append(d)
        labels.append(index.split('/')[-1].replace('.csv', '').replace('.nc', ''))
        colors.append('#CC397C')
    fig, ax = plt.subplots()
    ax.set_ylabel('ratio')
    bplot = ax.boxplot(data, tick_labels=labels)
    plt.xticks(rotation=90)
    plt.ylim((0, 4))
    plt.tight_layout()
    plt.title('LENS var bias')
    plt.savefig('figures/ice-cores/quick-lens-bias-boxplot.png', dpi=200)
elif (inp == 'r'): #regions
    fig, axes = plt.subplots(3, 1, dpi=400, subplot_kw={'projection': cartopy.crs.Robinson(central_longitude=0)})
    rcParams.update({'font.size': 7})
    #plt.tight_layout(h_pad=8)

    #contiental and hemispheric regoins
    for ax_i in range(2):
        ax = axes[ax_i]
        ax.add_feature(cartopy.feature.OCEAN, zorder=9, facecolor='white', edgecolor='black', linewidth=0.5)
        ax.set_title('Continental Regions' if ax_i == 1 else 'Hemispheric Regions')
        ax.set_global()

        #patches
        color_i = 4 if ax_i == 1 else 5
        oppacity = '90'
        if ax_i == 0:
            ax.add_patch(Polygon(
                np.array([
                    [-60+45, -60],
                    [-60+45, -60 + a_p -(-60)],
                    [-60+45 + 360-45, -60 + a_p -(-60)],
                    [-60+45 + 360-45, s_g],
                    [-60+45 + 360-45+45, s_g],
                    [-60+45 + 360-45+45, -60],
                ]),
                closed=True,
                fill=True,
                fc=patch[color_i] + oppacity,
                transform=cartopy.crs.PlateCarree()))
        for region, patch in patches.items():
            if region in ['Alaska']:
                continue
            if ax_i == 0 and not region in ['Antarctica', 'Arctic', 'South Greenland', 'North Greenland']:
                continue
            if region == 'North America':
                ax.add_patch(Polygon(
                    np.array([
                        [patch[0], patch[1]],
                        [patch[0], patch[1] + patch[3]],
                        [-60, patch[1] + patch[3]],
                        [-60, s_g],
                        [patch[0] + patch[2], s_g],
                        [patch[0] + patch[2], patch[1]],
                    ]),
                    closed=True,
                    fill=True,
                    fc=patch[color_i] + oppacity,
                    transform=cartopy.crs.PlateCarree()))
            elif region == 'Europe':
                ax.add_patch(Polygon(
                    np.array([
                        [patch[0], patch[1]],
                        [patch[0], patch[1] + patch[3]],
                        [-60 + 45, patch[1] + patch[3]],
                        [-60 + 45, a_p],
                        [patch[0] + patch[2], a_p],
                        [patch[0] + patch[2], patch[1]],
                    ]),
                    closed=True,
                    fill=True,
                    fc=patch[color_i] + oppacity,
                    transform=cartopy.crs.PlateCarree()))
            else:
                ax.add_patch(Rectangle(xy=[patch[0], patch[1]], width=patch[2], height=patch[3], facecolor=patch[color_i] + oppacity, transform=cartopy.crs.PlateCarree()))
            
            #setup color scale
            cmap = colormaps['BrBG']#inferno
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

    #local emission regions
    ax = axes[2]
    ax.add_feature(cartopy.feature.COASTLINE, edgecolor='black', linewidth=0.5)
    ax.set_title('Local Regions')
    ax.set_global()

    #elevation
    '''elev = Dataset('data/elevation-land-only.nc') #from https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2/ETOPO2v2-2006/ETOPO2v2c/netCDF/
    elev_lon = elev['lon'][:]
    elev_lat = elev['lat'][:]
    elev_z = np.transpose(elev['land_elev'][:])
    mesh = plt.pcolormesh(elev_lon, elev_lat, elev_z, cmap=colormaps['Greys'], vmin=0, transform=cartopy.crs.PlateCarree())'''
    
    #local boxes
    anthro_boxes = json.load(open('data/emission-boxes.json'))
    colors = {k: l[-2] for k, l in patches.items()}
    colors['South America'] = colors['South ZAmerica']
    colors['USA'] = colors['North America']
    colors['Alaska'] = colors['Arctic']
    colors['Greenland'] = colors['North Greenland']
    anthro_boxes = json.load(open('data/emission-boxes.json'))
    for region, boxes in anthro_boxes.items():
        box = boxes[-1]
        #for box in boxes:
        ax.add_patch(Rectangle(xy=[box[2], box[0]], width=np.abs(box[3]-box[2]), height=np.abs(box[1]-box[0]), edgecolor=colors[region], facecolor=colors[region] + '90', zorder=10, transform=cartopy.crs.PlateCarree()))

    save_path = 'figures/ice-cores/regions.png'
    plt.savefig(save_path)
    print('saved as ' + save_path)

elif (inp == 'z'):#testing
    plt.plot([1], [1])
    legend_handels = []
    '''legend_handels = [
        Patch(label='North Pole', facecolor=patches['Arctic'][4]+ '99'),
        Patch(label='Alpine', facecolor=patches['Africa'][4]+ '99'),
        Patch(label='South Pole', facecolor=patches['Antarctica'][4]+ '99'),
    ]'''
    for region, patch in patches.items():
        if region == 'Alaska':
            continue
        if region == 'South ZAmerica':
            region = 'South America'
        legend_handels.append(Patch(label=region, facecolor=patch[4] + '99'))
    plt.legend(handles=legend_handels)
    plt.savefig('../test.png', dpi=400)



print("n=" + str(len(main_dict)))