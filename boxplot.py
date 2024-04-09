import matplotlib.pyplot as plt
import numpy as np
from tools import ToolBox

mean, median, std, q1, q3, max, min = ToolBox.ncdf_avg("/Users/noahliguori-bills/Downloads/CESM-tools/data/douglas-course-emissions.nc", "BC")
print(mean, median, std, q1, q3, max, min)

#plt.errorbar(1, mean, yerr=std)
#plt.show()

item = {}
item["label"] = 'box' # not required
item["mean"] = mean # not required
item["med"] = median
item["q1"] = q1
item["q3"] = q3
item["whislo"] = max # required
item["whishi"] = min # required 
item["fliers"] = []

fig, axes = plt.subplots(1, 1)
axes.bxp([item])
axes.set_title('Default')
y_values = [0] + [1/np.power(10,x) for x in reversed(range(13, 15))]
y_axis = [0, 1, 2]
plt.yticks(y_axis, y_values)
#plt.show()

'''# rectangular box plot
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
plt.show()'''