import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Wedge
import matplotlib.patheffects as pe
from matplotlib import colormaps
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib import rcParams
#from scipy.integrate import quad
#from scipy.stats import lognorm
from scipy.stats import norm
#import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
import plotly.express as px
from netCDF4 import Dataset
import numpy as np
import cartopy
import scipy
import tools
import math
import json
import sys

#read index file
t = tools.ToolBox()
p = pd.read_csv('data/standardized-ice-cores/index.csv')
p = p.reset_index()

#setup vars
exclude = set([])#set(['mcconnell-2017-1.csv', 'brugger-2021-1.csv'])
windows = [11]#[1, 3, 5, 11]
full_data = {}
for key in windows:
    full_data[key] = [] #filename, lat, lon, PI averge, PD averge, ratio, PI year, PD year, 3 averege, 3 year
for_cartopy = {}
filename_region = {}
filename_index = {}
dont_use = set()
a_p = 66.566667
m_g = 71 #midpoint between lowest greenland (59) and highest (83)
s_g = 59
patches = { #Okabe and Ito colorblind pallet
    'Arctic': (-15, a_p, 315, 90 - a_p, '#6CB3E4'),
    'Greenland': (-55, s_g, 35, 90 - s_g, '#880D1E'),
    'Antarctica': (-180, -60, 360, -30, '#2C72AD'),
    'South America': (-90, 15, 70, -71, '#EFE362'),
    'North America': (-170, 15, 115, a_p - 15, '#C17EA5'),
    'Europe': (-20, 23.5, 80, s_g - 23.5, '#C86526'),
    #'Middle east': (30, 23.5, 30, s_g - 23.5, '#DDA138'),
    'Africa': (-20, 23.5, 80, -58.5, '#000000'),
    'Asia': (60, 5, 90, a_p - 5, '#459B76')
}
''''South Greenland': (-55, s_g, 35, m_g - s_g, '#880D1E'),
    'North Greenland': (-60, m_g, 45, 90 - m_g, '#DDA138'),
    'East Antarctica': (-180, -60, 180, -30, '#000000'),
    'West Antarctica': (0, -60, 180, -30, '#6CB3E4'),'''

def patch_min_max(patch):
    lon_bounds = (patch[0], patch[0] + patch[2])
    lat_bounds = (patch[1], patch[1] + patch[3])
    lon_min = np.min(lon_bounds)
    lon_max = np.max(lon_bounds)
    lat_min = np.min(lat_bounds)
    lat_max = np.max(lat_bounds)
    return lat_min, lat_max, lon_min, lon_max

def divide(p_d, p_i):
    n = p_d.copy()
    for column in p_d:
        if column not in ('model', 'n ensemble members', 'window'):
            for i in range(len(p_d[column])):
                n[column].iloc[i] = np.abs(p_d[column].iloc[i] / p_i[column].iloc[i])
    return n

def within_patch(lat, lon, patch, name):
    lat_min, lat_max, lon_min, lon_max = patch_min_max(patch)
    if name == 'Arctic':
        return lat_min <= lat <= lat_max and not within_patch(lat, lon, patches['Greenland'], 'Greenland')
        #return lat_min <= lat <= lat_max and not within_patch(lat, lon, patches['South Greenland'], 'South Greenland') and not within_patch(lat, lon, patches['North Greenland'], 'North Greenland')
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max

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
        #add data to datasets
        if (y1 != None and y3 != None and abs(y1 - y3) >= 100):
            for key in windows:
                if math.isnan(lat) or math.isnan(lon):
                    lat, lon, abbr = dup_index_map[filename]
                #full_data[key].append([filename, lat, lon, a1[key], a2[key], a3[key]/a1[key], y1, y2, a3[key], y3])
                for region, patch in patches.items():
                    if within_patch(lat, lon, patch, region):
                        filename_region[filename] = region
            for_cartopy[filename] = {'lat': lat, 'lon': lon, 'ratio': a3[11] / a1[11], 'abbr': abbr, 'filename': filename}
        else:
            dont_use.add(filename)

for_cartopy = {k: v for k, v in sorted(for_cartopy.items(), key=lambda item: item[1]['ratio'])} #sort by ratio
for i in range(len(for_cartopy.keys())):
    filename_index[list(for_cartopy.keys())[i]] = i + 1
final_pd = pd.DataFrame.from_records(for_cartopy).T

#plot
inp = 't' if len(sys.argv) < 2 else sys.argv[1]
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
        ax.add_feature(cartopy.feature.COASTLINE, edgecolor='grey')

        #elevation
        elev = Dataset('data/elevation-land-only.nc') #from https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2/ETOPO2v2-2006/ETOPO2v2c/netCDF/
        elev_lon = elev['lon'][:]
        elev_lat = elev['lat'][:]
        elev_z = np.transpose(elev['land_elev'][:])
        mesh = plt.pcolormesh(elev_lon, elev_lat, elev_z, cmap=colormaps['Greys'], vmin=0, transform=cartopy.crs.PlateCarree())

        #setup color scale
        max_ratio = 0
        for key in for_cartopy.keys():
            #r = math.log(for_cartopy[key]['ratio'], 10)
            r = for_cartopy[key]['ratio']
            max_ratio = r if r > max_ratio else max_ratio
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
        norm = BoundaryNorm(bounds, cmap.N)
        sm = ScalarMappable(cmap=cmap, norm=norm)

        #plot
        i = 1
        for key in for_cartopy.keys():
            obj = for_cartopy[key]
            [lat, lon] = [obj['lat'], obj['lon']]
            scale = 0.7468 if projection == 'rotated-pole' else 1
            color = cmap(norm(obj['ratio']))
            stroke_color = "black" if color == (0.32941176470588235, 0.18823529411764706, 0.0196078431372549, 1.0) else "black"
            #temp = key.split('-')
            #print(temp[0].capitalize() + " et al. " + temp[1], lat, lon, round(obj['ratio'], 3), temp[2])
            modification = ""
            if key == 'sigl-2018-1.csv':
                modification = ",35"
            if key not in ('eichler-2023-1.csv'):
                if key != 'sigl-2018-1.csv':
                    plt.plot(lon, lat, c=color, markeredgecolor='black', marker='.', markersize=16*scale, transform=cartopy.crs.PlateCarree())
                else:
                    ax.add_patch(Wedge((-34, -8), 4 * scale, -90, 90, fc=cmap(norm(for_cartopy['eichler-2023-1.csv']['ratio'])), ec='black', zorder=999999))
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
        for patch in patches.values():
            ax.add_patch(Rectangle(xy=[patch[0], patch[1]], width=patch[2], height=patch[3], facecolor=patch[4] + '50', edgecolor=patch[4],transform=cartopy.crs.PlateCarree()))

        plt.savefig('figures/ice-cores/testmap-' + projection + '.png', bbox_inches='tight', pad_inches=0.0)
        #plt.show()
    s = set()
    for pub in str(index_name_map).replace('}', '').replace('{', '').split(','):
        s.add(pub[t.find_nth(pub, " ", 2) + 2:t.find_nth(pub, "-", 2)])
    print(len(s), "unique ice core pubs")
    #print(index_name_map)
elif (inp == 'l'): #Lens data
    #setup data:
    lens_pi = pd.read_csv('data/model-ice-depo/lens/pi.csv')
    lens_pd = pd.read_csv('data/model-ice-depo/lens/pd.csv')
    lens_avg = pd.read_csv('data/model-ice-depo/lens/a10lv30.csv')
    models = {
        'LENS': {
            'dataset': lens_avg,
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': '#F5B341',#IBM Design library's colorblind pallete
            },
        'CESM': {
            'dataset': divide(pd.read_csv('data/model-ice-depo/cesm-wetdry/pd.csv'), pd.read_csv('data/model-ice-depo/cesm-wetdry/pi.csv')),
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': '#EE692C',
            },
        'CMIP6': {
            'dataset': divide(pd.read_csv('data/model-ice-depo/cmip6/pd.csv'), pd.read_csv('data/model-ice-depo/cmip6/pi.csv')),
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': '#CC397C',
            },
        'Ice Core': {
            'dataset': for_cartopy,
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': '#6C62E7',
            },
        'CESM-SOOTSN': {
            'dataset': divide(pd.read_csv('data/model-ice-depo/cesm-sootsn/pd.csv'), pd.read_csv('data/model-ice-depo/cesm-sootsn/pi.csv')),
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': '#638FF6',
            }
    }
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
            'dataset': for_cartopy,
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
            'dataset': for_cartopy,
            'data': {'ratios': None, 'means': None, 'stds': None},
            'color': '#6C62E7',
            }
    }'''
    models_datasets = {key: value['dataset'] for key, value in models.items()}
    models_colors = {key: value['color'] for key, value in models.items()}
    n_models = len(list(models))
    dont_use = dont_use.union({'model number', 'Unnamed: 0', 'BC_vars', 'year'})
    bar_lables = []
    bar_means = {key: [] for key in models.keys()}
    bar_stds = {key: [] for key in models.keys()}
    distribution_map = {}
    ice_based_dists = {}
    ice_based_labels = []
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
                ice_mean = for_cartopy[col_name]['ratio']
                bar_means[model_key].append(ice_mean)
                #ice_based_dists[col_name] = {'PD': lens_pd[col_name], 'PI': lens_pi[col_name]}
                bar_lables.append(filename_region[col_name] + '-' + str(filename_index[col_name]).zfill(2))
                bar_stds[model_key].append(0)
            else:
                model_ratios = ds[col_name]#lens_pd[col_name] / lens_pi[col_name]
                model_mean = np.mean(model_ratios)
                model_std = np.std(model_ratios.dropna())#scipy.stats.gstd(model_ratios.dropna())
                bar_stds[model_key].append(0)#(model_std)
                bar_means[model_key].append(model_mean)
    #resort everything by region
    bar_lables, bar_means['LENS'], bar_means['Ice Core'], bar_means['CESM'], bar_means['CMIP6'], bar_stds['LENS'], bar_stds['Ice Core'], bar_stds['CESM'], bar_stds['CMIP6'], bar_means['CESM-SOOTSN'], bar_stds['CESM-SOOTSN'], background_colors = zip(*sorted(list(zip(bar_lables, bar_means['LENS'], bar_means['Ice Core'], bar_means['CESM'], bar_means['CMIP6'], bar_stds['LENS'], bar_stds['Ice Core'], bar_stds['CESM'], bar_stds['CMIP6'], bar_means['CESM-SOOTSN'], bar_stds['CESM-SOOTSN'], background_colors))))
    #bar_lables, bar_means['Ice Core'], bar_stds['Ice Core'], bar_means['LENS-LV30'], bar_stds['LENS-LV30'], bar_means['LENS-LV29'], bar_stds['LENS-LV29'], bar_means['LENS-LV28'], bar_stds['LENS-LV28'], background_colors = zip(*sorted(list(zip(bar_lables, bar_means['Ice Core'], bar_stds['Ice Core'], bar_means['LENS-LV30'], bar_stds['LENS-LV30'], bar_means['LENS-LV29'], bar_stds['LENS-LV29'], bar_means['LENS-LV28'], bar_stds['LENS-LV28'], background_colors))))
    #bar_lables, bar_means['CMIP6 PD'], bar_stds['CMIP6 PD'], bar_means['CMIP6 PI'], bar_stds['CMIP6 PI'], background_colors = zip(*sorted(list(zip(bar_lables, bar_means['CMIP6 PD'], bar_stds['CMIP6 PD'], bar_means['CMIP6 PI'], bar_stds['CMIP6 PI'], background_colors))))
    #Remove Duplicate Region Labels
    region_lables = list(map(lambda x: x.split('-')[0], bar_lables))
    transition_indexes = []
    old = ''
    for i in range(len(region_lables)):
        region = region_lables[i]
        if region == old:
            region_lables[i] = ''
        else:
            old = region
            transition_indexes.append(i)
    transition_indexes.append(transition_indexes[-1] + 1)
    #get lens distributions by model
    for index, row in lens_pd.iterrows():
        model_index = int(row['model number'].split('-')[0])
        ice_based_labels.append(model_index)
        ratio_dists = lens_avg.iloc[model_index - 19].iloc[1:len(lens_avg.iloc[model_index - 19])]#row.iloc[3:len(row)] / lens_pi.iloc[index].iloc[3:len(row)]
        distribution_map[model_index] = ratio_dists
    distribution_lables = row.iloc[3:len(row)].index
    #plot mean bars
    all_h = []
    for key, value in bar_means.items():
        if key != 'LENS':
            all_h.append(value)
    max_lens_h = np.max(all_h)
    min_h = np.min(all_h)
    #max_lens_h = bar_stds["LENS"][bar_means["LENS"].index(max_lens_bar)] + max_lens_bar
    x = np.arange(len(bar_lables))  # the label locations
    width = 0.18 # the width of the bars
    multiplier = 0
    fig, ax = plt.subplots(layout='constrained')
    true_start = true_end = None
    #plot line at y=1
    plt.axhline(1, c='black', xmin=0, xmax=1, linewidth=0.75)
    #plot rest of data
    for model_key, measurement in bar_means.items():
        offset = width * multiplier
        color = models_colors[model_key]
        #rects = ax.bar(x + offset, measurement, width, label=model_key, color=color)
        bars = ax.bar(x + offset, max_lens_h, width, label=model_key, color=background_colors)
        if multiplier == 0:
            true_start = plt.getp(bars[0], 'x')
        elif model_key in list(bar_means)[-1]:
            true_end = plt.getp(bars[-1], 'x') + plt.getp(bars[-1], 'width')
        #plot mean bars
        for i in range(len(transition_indexes) - 1):
            trans_i = transition_indexes[i]
            next_i = transition_indexes[i + 1]
            region = bar_lables[trans_i].split('-')[0]
            x_start = (plt.getp(bars[trans_i], 'x') - offset + width + 0.07 - (trans_i / len(bar_lables)) * (1 - width * n_models)) / len(bar_lables)
            x_end   = x_start + ((next_i - trans_i) - width - 0.07) / len(bar_lables) #(x_start + width * 4 * (next_i - trans_i)) / 36
            plt.axhline(np.mean(bar_means[model_key][trans_i:next_i]), c=color, xmin=x_start, xmax=x_end, linewidth=0.75)
        multiplier += 1
        #plt.axhline(np.mean(bar_means[model_key]), c=color, xmin=0, xmax=0.5)
        plt.errorbar(x + offset, bar_means[model_key], yerr=bar_stds[model_key], fmt=".", color=color, elinewidth=0.5, capthick=0.5)
    ax.set_ylabel('1980/1850 BC Ratio')
    ax.set_xlabel('Region')
    ax.set_title('Modeled and Observed BC Deposition Change')
    ax.set_xticks(x + width, region_lables)
    plt.xticks(rotation=90)
    plt.xlim([-width, x[-1] + width * n_models])
    plt.ylim([min_h - 0.01, max_lens_h])
    ax.get_xaxis().get_major_formatter().labelOnlyBase = False
    ax2 = ax.twiny()
    plt.xlim([-width, x[-1] + width * n_models])
    plt.ylim([min_h - 0.01, max_lens_h])
    ax2.set_xticks(x + width, [int(label[len(label)-2:len(label)]) for label in bar_lables], fontsize=10)
    plt.xticks(rotation=90)
    ax2.set_xlabel('Ice Core Number')
    rcParams.update({'font.size': 9})
    plt.yscale('log')
    #ax.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
    #manualy change legend colors
    #leg = ax.get_legend()
    for i in range(len(list(models_colors.items()))):
        key, color = list(models_colors.items())[i]
        #leg.legend_handles[i].set_color(color)
    for a in plt.gcf().get_axes():
        for i in range(len(bar_lables)):
            filename = bar_lables[i].split('-')[0]
            color = patches[filename][-1]
            a.get_xticklabels()[i].set_color(color)
    plt.savefig('figures/ice-cores/test4.png', dpi=300)
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
elif (inp == 't'): #smoothing
    pass

print("n=" + str(len(for_cartopy)))