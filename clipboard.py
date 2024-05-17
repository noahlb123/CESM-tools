import pyperclip

def filename2modelname(filename):
    model_name = filename.split('_historical')[0].replace('sootsn_LImon_', '')
    start_year = filename[filename.rfind("_") + 1:filename.rfind("-") - 2]
    return model_name, int(start_year)

print("rm ")
m = {}
for name in pyperclip.paste().split('\n'):
    model, year = filename2modelname(name)
    if year not in (1850, 1950, 1940):
        print(name, end=' ')
    else:
        if model in m:
            m[model].append(year)
        else:
            m[model] = [year]