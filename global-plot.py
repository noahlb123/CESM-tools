import cartopy
import matplotlib.pyplot as plt
import pandas as pd

class IdentityChecker:
    def __init__(self, threshold):
        self.t = threshold
        self.l = []
        self.identicals = []
    
    def is_unique(self, latlon, notes):
        latlon = latlon[1:3] if len(latlon) == 4 else latlon
        for v in self.l:
            if abs(v[0] - latlon[0]) < self.t or abs(v[1] - latlon[1]) < self.t:
                self.identicals.append([latlon, v, notes])
                return False
        self.l.append(latlon)
        return True
    
    def results(self):
        return self.identicals

checker = IdentityChecker(0.00001)
ax = plt.axes(projection=cartopy.crs.PlateCarree(0))
ax.set_extent([-180, 180, -90, 90], crs=cartopy.crs.PlateCarree())

#get this from https://www.naturalearthdata.com/features/
glaciers = cartopy.feature.NaturalEarthFeature(
    category='physical',
    name='glaciated_areas',
    scale='10m',
    facecolor='#ceeefb')
ax.add_feature(cartopy.feature.COASTLINE, edgecolor='grey')
ax.add_feature(glaciers)


p = pd.read_csv('data/ice-core-data.csv').dropna(subset=["S"])
coords = p.loc[:, ["N", "S", "E", "W"]].to_numpy()
latlons = [[-75, 43], [77.23, 28.61]]

counter = 0
for i in range(len(coords)):
    vector = coords[i]
    if checker.is_unique(vector, ""):
        counter += 1
        plt.plot(vector[2], vector[1], color='black', marker='o')

# Save the plot by calling plt.savefig() BEFORE plt.show()
#plt.savefig('coastlines.png')
plt.show()
print(counter, len(checker.results()), len(coords))
