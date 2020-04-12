#!/usr/bin/env bash

#little util to help getting data from large number of experiment results
#a step us meant to execute after open_tensor_board.sh step (when data is ready to query)
#same idea: point at dir, 5 subdirs will be queried

echo input path $1
cd $1

dirs=(*)
length=${#dirs[@]}
counter=1
for (( i=0; i<$length; i++ ))
do
	echo
	port=$((6010+$i))
	name=${dirs[$i]}
	dir=$name/tensorboard

	if [ -d "$dir" ]; then
		echo $name
		echo $port
		
		cd $dir
		for j in *; do xdg-open "http://localhost:"${port}"/data/plugin/scalars/scalars?tag=episode_reward&run="${j}"&experiment=&format=csv"; done
		cd ..
		mkdir -p csv
		sleep 3s
		mv ~/Downloads/scalars* ./csv
		cd ..
		
		if [ $(($counter % 5)) -eq 0 ]; then
			echo "when ready to continue hit [ENTER]:"
			read
		fi
		
		counter=$(($counter+1))
	fi
done


