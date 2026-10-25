#!/bin/bash
D=/Users/hiroshi/projects/tensor4all/threadteam-rs-team/docs/design/thresholds-data
S=/private/tmp/claude-501/-Users-hiroshi-projects-tensor4all/1e14fd5e-6cd9-434f-ab57-23b41e514391/scratchpad
H="python3 $D/hostrun.py $D/raw/processes.log"
for i in 1 2 3; do
  $H dispatch-p$i $D/raw/dispatch-p$i.txt -- $S/bin/dispatch_cost $D/raw/dispatch-p$i.csv
done
for i in 1 2; do
  $H lusweep-p$i $D/raw/lusweep-p$i.csv -- $S/bin/thr-faer lusweep 2
done
echo ALLDONE
