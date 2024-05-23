import pyperclip
import numpy as np

'''def filename2modelname(filename):
    model_name = filename.split('_historical')[0].replace('sootsn_LImon_', '')
    start_year = filename[filename.rfind("_") + 1:filename.rfind("-") - 2]
    end_year = filename[filename.rfind("-") + 1:filename.rfind(".") - 2]
    return model_name, int(start_year), int(end_year)

def valid_range(s_year, e_year):
    def within_one(a, b):
        return np.abs(a - b) <= 1
    target_years = (1850, 1980)
    for year in target_years:
        if within_one(s_year, year) or within_one(e_year, year):
            return True
    return False

print("rm ")
m = {}
for name in pyperclip.paste().split('\n'):
    model, s_year, e_year = filename2modelname(name)
    if not valid_range(s_year, e_year):
        print(name, end=" ")'''

print(pyperclip.paste().replace(' ', ","))
#bash wget-20240521121843.sh -H; bash wget-20240521192012.sh -H; bash wget-20240521211909.sh -H; bash wget-20240521211944.sh -H; bash wget-20240521211955.sh -H