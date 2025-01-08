import math
import pandas as pd
import numpy as np
from sys import platform
import os
if platform != "darwin":
    from netCDF4 import Dataset

class ToolBox:
    def __init__(self):
        pass

    def remove_list_indexs(self, l, to_remove):
        for i in sorted(to_remove, reverse=True):
            del l[i]
        return l
    
    def nearest_search(self, arr, x):
        low = 0
        high = len(arr) - 1
        mid = 0
        while low <= high:
            mid = (high + low) // 2
            #if there is approximate solution...
            if (high - low <= 2):
                #check nearest 7 indexes for closest match
                poss = [mid + x - 4 for x in range(7)]
                #remove out of bounds indexes
                out_of_bounds = []
                for i in range(len(poss)):
                    if poss[i] < 0 or poss[i] >= len(arr):
                        out_of_bounds.append(i)
                poss = self.remove_list_indexs(poss, out_of_bounds)
                #find closest
                m = mid
                for i in poss:
                    if (abs(arr[i] - x) < abs(arr[m] - x)):
                        m = i
                return m
            # If x is greater, ignore left half
            if arr[mid] < x:
                low = mid + 1
            # If x is smaller, ignore right half
            elif arr[mid] > x:
                high = mid - 1
            # means x is present at mid
            else:
                return mid
        # If we reach here, then the element was not present
        return -1

    def datestr2int(self, s):
        l = s.split("/")
        return (int(l[0]) - 1) * 12 + int(l[1])
    
    def matplot_tooltips(self, ax, fig, sc, annotations):
        annot = ax.annotate("", xy=(0,0), xytext=(5,5),textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def update_annot(ind):
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            text = "{}".format(" ".join([annotations[n] for n in ind["ind"]]))
            annot.set_text(text)
            annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect("motion_notify_event", hover)
    
    def in_bounds(self, index, arr):
        return 0 <= index < len(arr)
    
    #list of time, list of x, year float, list of windows
    def get_avgs(self, t_l, x_l, year, windows):
        output = {}
        focus_index = self.nearest_search(t_l, year)
        volacno_threshold = 99999999
        if (focus_index == -1):
            return [output, None]
        focus_year = t_l[focus_index]
        def above_y_min(i, max_yr, w_size):
            if max_yr != -99999999:
                return t_l[i] >= max_yr - w_size
            else:
                return focus_year - w_size // 2 <= t_l[i] <= focus_year + w_size // 2
        for w_size in windows:
            n = max_yr = 1
            my_sum = float(x_l[focus_index])
            def search_bounds(i, t_l, x_l, w_size, max_yr, n, my_sum, volacno_threshold, direc):
                max_yr = -99999999
                while self.in_bounds(i, t_l) and above_y_min(i, max_yr, w_size):
                    n += 1
                    x = float(x_l[i])
                    if math.isnan(x):
                        x = 0
                        n -= 1
                    if x > volacno_threshold:
                        break
                    my_sum += x
                    max_yr = t_l[i] if t_l[i] > max_yr else max_yr
                    i += direc
                return [n, my_sum, max_yr]
            [n, my_sum, max_yr] = search_bounds(focus_index + 1, t_l, x_l, w_size, max_yr, n, my_sum, volacno_threshold, 1)
            [n, my_sum, max_yr] = search_bounds(focus_index - 1, t_l, x_l, w_size, max_yr, n, my_sum, volacno_threshold, -1)
            output[w_size] = my_sum / n
        return [output, t_l[focus_index]]
    
    #returns mean, median, std, first quartile, last quartile, max, min
    def ncdf_avg(file_path, var_key):
        f = Dataset(file_path, "r")
        x = f[var_key][:]
        f.close()
        return (np.mean(x), np.median(x), np.std(x), np.quantile(x, 0.25), np.quantile(x, 0.75), np.max(x), np.min(x))
    
    #histogram bins to labels "1 to 2"
    def bins2labels(self, bins):
        labels = []
        for i in range(len(bins) - 1):
            if isinstance(bins[i], (int, float, complex)):
                labels.append(str("%.3g" % bins[i]) + ' to ' + str("%.3g" % bins[i + 1]))
            elif isinstance(bins[i], (str)):
                labels.append(bins[i] + ' to ' + bins[i + 1])
            else:
                raise Exception("in tools.py, bins2labels, unrecognised bin element type")
        return labels
    
    #get even spacing between numbers
    def get_ticks(self, low_bound, high_bound, n):
        diff = high_bound - low_bound
        step = diff / (n - 1)
        return [x * step + low_bound for x in range(n)]
    
    def get_ice_coords(self, index_path, dupe_path):
        ice_coords = {}
        dup_index_map = {}
        p_dup = pd.read_csv(dupe_path)
        for index, r in p_dup.iterrows():
            dup_index_map[r['Filename']] = [r['Lat'], r['Lon'], r['Abbreviation']]
        p = pd.read_csv(index_path)
        p = p.reset_index()
        for index, row in p.iterrows():
            for i in range(row['n_cores']):
                filename = row['First Author'].lower() + '-' + str(row['Year']) + '-' + str(i + 1) + '.csv'
                lat = row['N']
                lon = row['E']
                if math.isnan(lat) or math.isnan(lon):
                    lat, lon, abbr = dup_index_map[filename]
                ice_coords[filename] = (lat, lon)
        return ice_coords
    
    def find_nth(self, haystack: str, needle: str, n: int) -> int:
        start = haystack.find(needle)
        while start >= 0 and n > 1:
            start = haystack.find(needle, start+len(needle))
            n -= 1
        return start
    
    def within(self, a, b, range):
        return np.abs(a - b) <= range
    
    #lat:float (-90, 90), lon:float (-180, 180), res: width in degrees of gridbox
    #returns area of gridbox in km^2 or units of AVG_EARTH_RADIUS_KM
    def coords2area(self, lat, lon, res):
        AVG_EARTH_RADIUS_KM = 6371.0088
        west = math.radians(lon - res / 2)
        east = math.radians(lon + res / 2)
        south = math.radians(lat - res / 2)
        north = math.radians(lat + res / 2)
        area = (east - west) * (math.sin(north) - math.sin(south)) * (AVG_EARTH_RADIUS_KM**2)
        return area
    
    def patch_min_max(self, patch):
        lon_bounds = (patch[0], patch[0] + patch[2])
        lat_bounds = (patch[1], patch[1] + patch[3])
        lon_min = np.min(lon_bounds)
        lon_max = np.max(lon_bounds)
        lat_min = np.min(lat_bounds)
        lat_max = np.max(lat_bounds)
        return lat_min, lat_max, lon_min, lon_max
    
    def within_patch(self, lat, lon, patch, name):
        lat_min, lat_max, lon_min, lon_max = self.patch_min_max(patch)
        return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
    
    def custom_cmap(self, seq):
        """Return a LinearSegmentedColormap
        seq: a sequence of floats and RGB-tuples. The floats should be increasing
        and in the interval (0,1).
        from https://stackoverflow.com/questions/16834861/create-own-colormap-using-matplotlib-and-plot-color-scale
        ex. red, white, blue [(0, 0, 1), (1, 1, 1), 0.5, (1, 1, 1), (1, 0, 0)]
        """
        try:
            mcolors
        except NameError:
            import matplotlib.colors as mcolors
        seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
        cdict = {'red': [], 'green': [], 'blue': []}
        for i, item in enumerate(seq):
            if isinstance(item, float):
                r1, g1, b1 = seq[i - 1]
                r2, g2, b2 = seq[i + 1]
                cdict['red'].append([item, r1, r2])
                cdict['green'].append([item, g1, g2])
                cdict['blue'].append([item, b1, b2])
        return mcolors.LinearSegmentedColormap('CustomMap', cdict)
    
    #from https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
    def moving_average(self, a, n):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n:] / n
    
    #for dicts with structure {'k': ['a', 'b', 'c']}
    def invert_dict_list(self, input_dict):
        output = {}
        #if many values for each key
        if type(list(input_dict.values())[0]) in [type([]), type(pd.Series([0]))]:
            [output.update(d) for d in [{value: key for value in l} for key, l in input_dict.items()]]
        #if one value for each key
        else:
            for key, value in input_dict.items():
                if value in output:
                    output[value].append(key)
                else:
                    output[value] = [key]
        return output

    
    def smallest_grid(self, dir, qual_f=lambda s, p: '.nc' in s, f_param=None):
        smallest_name = ''
        small_lat = 10000000
        small_lon = 10000000
        for file in os.listdir(dir):
            if qual_f(file, f_param):
                try:
                    f = Dataset(os.path.join(dir, file))
                except OSError:
                    continue
                has_lat_lon = 'lat' in f.variables and 'lat' in f.variables
                if not has_lat_lon:
                    continue
                this_lat = f.variables['lat'].shape[0]
                this_lon = f.variables['lon'].shape[0]
                if this_lat < small_lat or this_lon < small_lon:
                    smallest_name = file
                    small_lat = this_lat
                    small_lon = this_lon
        return smallest_name
    
    def any_substrings_in_string(self, substrings, s):
        lowercase = map(lambda sub: sub.lower(), substrings)
        for sub in lowercase:
            if sub in s.lower():
                return True
        return False
    
    def adjust_lat_lon_format(self, lats, lons, lat_min_max=(90, -90), lon_min_max=(180, -180)):
        coord_min_maxes = lat_min_max + lon_min_max
        changes = [0, 0]
        coords = (lats, lons)
        coord_to_index = {'lat': 0, 'lon': 1}
        for i in coord_to_index.values():
            max_diff = coord_min_maxes[0 + i * 2] - np.max(coords[i])
            min_diff = coord_min_maxes[1 + i * 2] - np.min(coords[i])
            if np.abs(max_diff) > 5 or np.abs(min_diff) > 5:
                changes[i] = np.mean((max_diff, min_diff))
        return (lats + changes[0], lons + changes[1])