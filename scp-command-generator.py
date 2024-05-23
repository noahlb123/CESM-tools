import os

#print("scp 'nlbills@derecho.hpc.ucar.edu:/glade/campaign/cesm/collections/cesmLE/restarts/{", end='')
#print("cp /glade/campaign/cesm/collections/cesmLE/restarts/{", end='')

def get_dir(i, year):
   #return "b.e11.B20TRC5CNBDRD.f09_g16.0" + "{:02d}".format(i) + "/b.e11.B20TRC5CNBDRD.f09_g16.0" + "{:02d}".format(i) + ".rest." + str(year) + "-01-01-00000.tar"
   return "b.e11.B20TRC5CNBDRD.f09_g16.0" + "{:02d}".format(i) + ".rest." + str(year) + "-01-01-00000.tar"

for i in range(1, 36):
    years = [1980 + 5 * (x - 2) for x in range(5)] if i != 34 else (1977, 1982, 1987, 1972, 1970)
    for year in years:
        if not os.path.exists(os.path.join(os.getcwd(), get_dir(i, year))):
            print(get_dir(i, year))