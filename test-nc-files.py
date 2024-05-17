from netCDF4 import Dataset
import os
import platform

system = platform.system() #differentiate local and derecho env by sys platform
data_path = os.path.join(os.getcwd(), 'data', 'cmip6') if system == "Darwin" else os.getcwd()

#get data from cmip6 files
for filename in os.listdir(data_path):
    if filename not in ('tools.py', 'index-dup-cores.csv', '__pycache__', 'cmip.py', 'index.csv', 'ip.py', 'pi.csv', 'pd.csv', ".DS_Store") and 'wget' not in filename:
        file_path = os.path.join(data_path, filename)
        try:
            f = Dataset(file_path)
            f.close()
        except:
            print(file_path, end=' ')
print('done.')