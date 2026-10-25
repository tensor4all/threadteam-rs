#!/bin/bash
# Three processes of the confirmation sweep under the host protocol (hostrun.py).
D=/Users/hiroshi/projects/tensor4all/threadteam-rs-team/docs/design/thresholds-data
S=/private/tmp/claude-501/-Users-hiroshi-projects-tensor4all/1e14fd5e-6cd9-434f-ab57-23b41e514391/scratchpad
H="python3 $D/hostrun.py $D/raw/processes.log"
for i in 1 2 3; do
  $H confirm-p$i $D/raw/confirm-p$i.csv -- $S/bin/thr-confirm 2
done
echo ALLDONE
