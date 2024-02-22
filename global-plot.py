import cartopy
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

class IdenticalManager:
    def __init__(self):
        self.d = {}
    
    def get_key(self, vector):
        return str(vector[0]) + "," + str(vector[1])

    def add(self, root, new):
        key = self.get_key(root)
        if key in self.d:
            self.d[key].append(new)
        else:
            self.d[key] = [root, new]
    
    def print(self):
        for value in self.d.values():
            print(value)

class IdentityChecker:
    def __init__(self, threshold, ingnores=set()):
        self.t = threshold
        self.l = []
        self.identicals = IdenticalManager()
        self.n_unique = 0
        self.ignores = ingnores
    
    def is_unique_single(self, latlon):
        latlon = latlon[1:3] if len(latlon) == 4 else latlon
        for v in self.l:
            if (v[0] in self.ignores and latlon[0] in self.ignores) or (v[1] in self.ignores and latlon[1] in self.ignores):
                continue
            if abs(v[0] - latlon[0]) < self.t or abs(v[1] - latlon[1]) < self.t:
                self.identicals.add(v, latlon)
                return False
        self.l.append(latlon)
        self.n_unique += 1
        return True
    
    def is_unique(self, latlon):
        if type(latlon[0]) == type(pd.Series([])):
            #if list of latlons
            output = latlon[0].copy()
            for i in latlon[0].index:
                output[i] = (self.is_unique_single([latlon[0][i], latlon[1][i]]))
            return pd.Series(output)
        elif type(latlon[0]) == type(1):
            #if sinlg latlon
            return self.is_unique_single(vector)
        else:
            raise Exception(type(latlon[0]), "doesn't match either ", type(1), " or ", type(pd.Series([])))
    
    def results(self):
        self.identicals.print()

checker = IdentityChecker(0.01, set([]))
p = pd.read_csv('data/ice-core-data.csv').dropna(subset=["S"])
#remove duplicate ice core locations
print(len(p), "total ice core datasets")
p = p[checker.is_unique([p["N"], p["E"]])]
coords = p.loc[:, ["N", "S", "E", "W"]].to_numpy()
print(len(p), "unique ice core locations")

inp = input("Plotly or Cartopy? (p/c): ")
if (inp == "c"):
    #Matplot
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

    for i in range(len(coords)):
        vector = coords[i]
        plt.plot(vector[2], vector[1], color='black', marker='.', markersize=3, transform=cartopy.crs.PlateCarree())

    plt.savefig('figures/ice-core-locations.png', dpi=300)
    plt.show()
elif (inp == "p"):
    #Plotly
    fig = px.scatter_geo(p, lat='N', lon='E', hover_name='First Author', title='Ice Core Locations')
    #color='Earliest Year (CE)'
    fig.show()

print("duplicate locations:")
checker.results()