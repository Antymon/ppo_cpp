for i in */checkpoints/*/*.60*.json; do echo ${i:0:$((${#i}-5))}; done
for i in exp/ppo_cpp/resources/batch_checkpoints/*.json; do length=${#i}; prefix=${i:0:$((length-5))}; echo $prefix; done
for i in exp/ppo_cpp/resources/batch_checkpoints/*.json; do length=${#i}; prefix=${i:0:$((length-5))}; ./build/exp/ppo_cpp/ppo_cpp -p $prefix --cl --rn 0 --du 2; done
