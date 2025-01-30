#wget scripts im using
#/glade/derecho/scratch/nlbills/all-ice-core-data/sootsn/wget_script_2024-10-29_13-49-17.sh
#/glade/derecho/scratch/nlbills/all-ice-core-data/loadbc/wget_script_2024-10-29_13-55-31.sh
#/glade/derecho/scratch/nlbills/all-ice-core-data/mmrbc/wget_script_2024-10-29_14-4-3.sh
#/glade/derecho/scratch/nlbills/all-ice-core-data/wet-dry/wget_script_2024-10-29_14-12-6.sh
#/glade/derecho/scratch/nlbills/all-ice-core-data/wet-dry/wget_script_2024-10-29_14-19-49.sh

#download data and remove all unneeded data
for wget_file in /glade/derecho/scratch/nlbills/all-ice-core-data/sootsn/wget_script_2024-10-29_13-49-17.sh /glade/derecho/scratch/nlbills/all-ice-core-data/loadbc/wget_script_2024-10-29_13-55-31.sh /glade/derecho/scratch/nlbills/all-ice-core-data/mmrbc/wget_script_2024-10-29_14-4-3.sh /glade/derecho/scratch/nlbills/all-ice-core-data/wet-dry/wget_script_2024-10-29_14-12-6.sh /glade/derecho/scratch/nlbills/all-ice-core-data/wet-dry/wget_script_2024-10-29_14-19-49.sh;
do
  dir=${wget_file%/*}/ && chmod -x wget_file && bash wget_file -s && python3 remove-bad-dates.py 1850 1980 dir
done

#main data analysis
python3 nco-pi-pd.py drybc /glade/derecho/scratch/nlbills/cmip6-snow-dep/all x 10,10 &&
python3 nco-pi-pd.py drybc /glade/derecho/scratch/nlbills/cmip6-snow-dep/all CESM 10,10 &&
python3 nco-pi-pd.py sootsn /glade/derecho/scratch/nlbills/all-ice-core-data/sootsn x 10,10 &&
python3 nco-pi-pd.py loadbc /glade/derecho/scratch/nlbills/cmip-atmos x 10,10 &&
python3 nco-pi-pd.py mmrbc /glade/derecho/scratch/nlbills/all-ice-core-data/mmrbc x 10,10 &&

#timeseries
python3 nco-timeseries.py drybc /glade/derecho/scratch/nlbills/cmip6-snow-dep/all &&
python3 nco-timeseries.py drybc /glade/derecho/scratch/nlbills/cmip6-snow-dep/all CESM &&
python3 nco-timeseries.py sootsn /glade/derecho/scratch/nlbills/cmip6-snow-dep &&
python3 nco-timeseries.py loadbc /glade/derecho/scratch/nlbills/cmip-atmos