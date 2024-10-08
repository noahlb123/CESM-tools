#main data analysis
python3 nco-pi-pd.py drybc /glade/derecho/scratch/nlbills/cmip6-snow-dep/all &&
python3 nco-pi-pd.py drybc /glade/derecho/scratch/nlbills/cmip6-snow-dep/all CESM &&
python3 nco-pi-pd.py sootsn /glade/derecho/scratch/nlbills/cmip6-snow-dep &&
python3 nco-pi-pd.py loadbc /glade/derecho/scratch/nlbills/cmip-atmos &&

#timeseries
python3 nco-timeseries.py drybc /glade/derecho/scratch/nlbills/cmip6-snow-dep/all &&
python3 nco-timeseries.py drybc /glade/derecho/scratch/nlbills/cmip6-snow-dep/all CESM &&
python3 nco-timeseries.py sootsn /glade/derecho/scratch/nlbills/cmip6-snow-dep &&
python3 nco-timeseries.py loadbc /glade/derecho/scratch/nlbills/cmip-atmos