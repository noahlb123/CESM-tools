#print("scp 'nlbills@derecho.hpc.ucar.edu:/glade/campaign/cesm/collections/cesmLE/restarts/{", end='')
print("cp '/glade/campaign/cesm/collections/cesmLE/restarts/{", end='')

def print_dir(i):
    print("b.e11.B20TRC5CNBDRD.f09_g16.0" + "{:02d}".format(i) + "/b.e11.B20TRC5CNBDRD.f09_g16.0" + "{:02d}".format(i) + ".rest.1980-01-01-00000.tar", end='')
    if (i != 35):
        print(',', end='')

def print_dir(i):
    print('"b.e11.B20TRC5CNBDRD.f09_g16.0' + "{:02d}".format(i) + '"', end='')
    if (i != 35):
        print(',', end='')

for i in range(1, 36):
#for i in [1]:
    print_dir(i)


print("}' .")