import os

#print("scp 'nlbills@derecho.hpc.ucar.edu:/glade/campaign/cesm/collections/cesmLE/restarts/{", end='')
#print("cp /glade/campaign/cesm/collections/cesmLE/restarts/{", end='')

def get_dir(i, year):
   root = "b.e11.B20TRC5CNBDRD.f09_g16.0" + "{:02d}".format(i)
   tar_file = root + ".rest." + str(year) + "-01-01-00000.tar"
   return root #os.path.join(root, tar_file)

def rename(s):
    #b.e11.B20TRC5CNBDRD.f09_g16.001.rest.1990-01-01-00000.tar
    rest_i = s.find(".rest.")
    year = s[rest_i + 6:len(s) - 4]
    n = s[rest_i - 3: rest_i]
    return year + "." + n

for i in range(1, 36):
    #years = [1850 + 5 * (x - 2) for x in range(5)] if i != 34 else (1977, 1982, 1987, 1972, 1970)
    path = os.path.join("/glade/campaign/cesm/collections/cesmLE/restarts/", get_dir(i, 0))
    sorted = list(map(rename, os.listdir(path)))
    sorted.sort()
    print(sorted[0])
    '''for year in years:
        if not os.path.exists(os.path.join(os.getcwd(), get_dir(i, year))):
            print(get_dir(i, year))'''