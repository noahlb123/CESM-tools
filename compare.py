import csv
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import scipy
import copy

def datestr2int(s):
    l = s.split("/")
    return (int(l[0]) - 1) * 12 + int(l[1])

def approx_search(arr, x):
    low = 0
    high = len(arr) - 1
    mid = 0
    while low <= high:
        mid = (high + low) // 2
        if (high - low <= 2):
            poss = [mid + x - 4 for x in range(7)]
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

#process csv
epa = {}
mine = {}
names = {'"Death Valley NP - Park Village"', '"Joshua Tree NP - Black Rock"', '"Redwood NP"', '"Los Angeles-North Main Street"'}
all_names = set()
namecoords = {} #lat ("32.6312420008573" "41.726892"), lon ("-115.48307" "-124.17949")
#get all names
with open('data/epa-califonia-pm2.5.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        name = row[7]
        if (name != '"Site Name"'):
            all_names.add(name)

#format dictionary object "epa"
for name in all_names:
    epa[name] = {"pm": [], "date": []}
    mine[name] = []

#extract vals from csv
with open('data/epa-califonia-pm2.5.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        name = row[7]
        if (name != '"Site Name"'):
            epa[name]["pm"].append(float(row[4].replace('"', "")))
            epa[name]["date"].append(datestr2int(row[0].replace('"', "")))
            if (not name in namecoords):
                namecoords[name] = {
                    "lat": float(row[-2].replace('"', "")),
                    "lon": float(row[-1].replace('"', ""))
                }

#process netcdf (dims 365, 192, 288)
f = Dataset("data/pm25-only-2010.nc", "r")
pm = f["PM25_SRF"][:]
lats = f["lat"][:]
lons = f["lon"][:] #(360,0)
for name in namecoords:
    dict = namecoords[name]
    dict["aprx_lat"] = approx_search(lats, dict["lat"])
    dict["aprx_lon"] = approx_search(lons, dict["lon"] + 180) #180 accounts for diff in coord systems
#extract pm vals
for name in all_names:
    coords = namecoords[name]
    dates = epa[name]["date"]
    for day in dates:
        mine[name].append(1000000000 * pm[day][coords["aprx_lat"]][coords["aprx_lon"]])

for name in all_names:
    x = epa[name]["pm"]
    y = mine[name]
    coef = np.polyfit(x,y,1)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    poly1d_fn = np.poly1d(coef)
    plt.scatter(x, y, s=5, c="#4d4564ff")
    plt.plot(x, poly1d_fn(x), '-', c="#da6032", label='R^2='+str(round(r_value, 4)))
    x2 = np.linspace(min(x), max(x), 100)
    plt.plot(x2, x2, '-', c="#000000", label='x=y')
    plt.title(name.replace('"', "") + " PM2.5 Comparison")
    plt.xlabel("EPA PM2.5 (μg/m3)")
    plt.ylabel("Modeled PM2.5 (μg/m3)")
    plt.legend()
    plt.savefig("figures/all-comparisons/" + name.replace('"', "").replace('.', "").replace('/', ""))
    plt.close()
    #plt.show()
print("done.")