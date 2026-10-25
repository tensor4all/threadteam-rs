#!/bin/bash
D=/Users/hiroshi/projects/tensor4all/threadteam-rs-team/docs/design/thresholds-data
S=/private/tmp/claude-501/-Users-hiroshi-projects-tensor4all/1e14fd5e-6cd9-434f-ab57-23b41e514391/scratchpad
H="python3 $D/hostrun.py $D/raw/processes.log"
for i in 1 2 3; do
  $H dispatch-p$i $D/raw/dispatch-p$i.txt -- $S/bin/dispatch_cost $D/raw/dispatch-p$i.csv
  $H faer-off-p$i $D/raw/faer-off-p$i.csv -- $S/bin/thr-faer off 2
  $H stack-p$i $D/raw/stack-p$i.csv -- $S/bin/thr-stack 2
done
$H faer-default-p1 $D/raw/faer-default-p1.csv -- $S/bin/thr-faer default 2
echo ALLDONE
