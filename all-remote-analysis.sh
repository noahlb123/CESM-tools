python3 nco-pi-pd.py drybc /glade/derecho/scratch/nlbills/cmip6-snow-dep/all &&
python3 cmip.py CMIP6 wetbc &&
python3 cmip.py CESM loadbc &&
python3 cmip.py CESM sootsn &&
python3 cmip.py CESM wetbc &&
python3 lens.py &&
python3 cmip-bin.py lens cmip6 loadbc &&
python3 lens-avg.py &&
python3 model-timeseries.py &&
python3 bin-timeseries.py