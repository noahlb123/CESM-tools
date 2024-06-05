from netCDF4 import Dataset
import os
import platform

system = platform.system() #differentiate local and derecho env by sys platform
data_path = os.path.join(os.getcwd(), 'data', 'cmip6', 'wetdrybc') if system == "Darwin" else os.getcwd()
target_v = 'wetbc' #sootsn

def filename2modelname(filename):
    prefix = 'LImon_' if target_v == 'sootsn' else 'AERmon_'
    model_name = filename[filename.index(prefix) + len(prefix): filename.index('_historical')]
    start_year = filename[filename.rfind("_") + 1:filename.rfind("-") - 2]
    end_year = filename[filename.rfind("-") + 1:filename.rfind(".") - 2]#sootsn_LImon_TaiESM1_historical_r1i1p1f1_gn_185001-201412.nc
    return model_name, int(start_year), int(end_year)

#get data from cmip6 files
for filename in os.listdir(data_path):
    if '.nc' in filename:
        file_path = os.path.join(data_path, filename)
        f = Dataset(file_path)
        print(file_path)
        units = f['time'].units
        unit_year = int(units[11:15])
        start_year = filename2modelname(filename)[1]
        print(units)
        print(unit_year)
        print(start_year)
        assert unit_year == start_year or unit_year == 1
        f.close()
print('done.')