case_path="/glade/work/nlbills/cases/TEST"

CTSM/cime/scripts/create_newcase --case $case_path --res f45_g37 --compset I2000Clm60Fates --run-unsupported --project UNCS0051

cd $case_path

./case.setup

./xmlchange STOP_OPTION=nmonths
./xmlchange STOP_N=1
./xmlchange RESUBMIT=0