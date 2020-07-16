# dump of one-liners to process checkpoints

# match files and trim last 5 characters corresponding to the ".json" extension
for i in */checkpoints/*/*.60*.json; do echo ${i:0:$((${#i}-5))}; done

# same as before but with length separated out for some clarity
for i in exp/ppo_cpp/resources/batch_checkpoints/*.json; do length=${#i}; prefix=${i:0:$((length-5))}; echo $prefix; done

# additionally calling evaluation binary
for i in exp/ppo_cpp/resources/batch_checkpoints/*.json; do length=${#i}; prefix=${i:0:$((length-5))}; ./build/exp/ppo_cpp/ppo_cpp -p $prefix --cl --rn 0 --du 2; done

# each folder list numerically all files and pick penultimate one which corresponds to best json descriptor (checkpoints are triples of files)
for i in */checkpoints/*; do echo $i; ls $i -1v | tail -n 2 | head -n 1; done

# after aggregating all best checkpoints iterate over jsons, trim extensions and use it as a checkpoint prefix
for i in exp/ppo_cpp/cpkt_533M/*.json; do prefix=${i:0:$((${#i}-5))}; echo "###"; echo $prefix; ./build/exp/ppo_cpp/ppo_cpp --cl -p ${prefix} --fps 1000; done &> exp/ppo_cpp/533M.txt