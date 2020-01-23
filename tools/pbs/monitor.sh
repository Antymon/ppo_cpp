#!/bin/bash

# checks if PBS qstat yields all the job ids associated with files under hardcoded location
# adds missing jobs

while :; do
  a=(jobs/ppo*)

  count=$(./qstat_count.sh)

  qstat | tail -n $count >monitor.txt

  grep -Poh "_\K[0-9]{2} " monitor.txt >tmp

  for i in {0..49}; do

    flag=0

    while IFS= read -r line; do
      if [ $i == $((10#$line)) ]; then
        flag=1
      fi
    done <tmp

    if [ $flag == 0 ]; then
      missing=${a[$i]}
      qsub $missing
      echo missing $i
    fi

  done

  sleep 30
done
