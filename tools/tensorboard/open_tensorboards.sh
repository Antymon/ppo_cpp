#!/usr/bin/env bash

#little util to help browsing through large number of experiment results
#point at directory with experiment results
#open all tensorboards in default browser on consecutive ports in batches of 5
#can kill tensorboards as you go

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
		tensorboard --logdir $dir --port $port &> tensorboard_dummy.log &
		sleep 3s
		xdg-open http://szymon-tws:$port/ &> tensorboard_dummy.log &
		
		if [ $(($counter % 5)) -eq 0 ]; then
			echo "when ready to continue hit [ENTER]:"
			read
		fi
		
		counter=$(($counter+1))
	fi
done


