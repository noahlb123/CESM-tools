print("scp 'nlbills@derecho.hpc.ucar.edu:/glade/campaign/cesm/collections/cesmLE/restarts/{", end='')
for i in range(8, 36):
    print("b.e11.B20TRC5CNBDRD.f09_g16.0" + "{:02d}".format(i) + "/b.e11.B20TRC5CNBDRD.f09_g16.0" + "{:02d}".format(i) + ".rest.1980-01-01-00000.tar", end='')

    if (i != 35):
        print(',', end='')

print("}' .")