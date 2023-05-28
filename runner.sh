for seed in {12340..12345}
do
	echo $seed
	python3 MyrtleGDNoise_SpeedLimit.py --train_seed $seed --n_train 500 & # This takes 4 GB GPU RAM 
	python3 MyrtleGDNoise_SpeedLimit.py --train_seed $seed --n_train 1250 & # This takes 8.4GB GPU RAM 
	python3 MyrtleGDNoise_SpeedLimit.py --train_seed $seed --n_train 2500 # This takes 10.5GM GPU RAM
done


for seed in {12340..12345}
do
	python3 MyrtleGDNoise_SpeedLimit.py --train_seed $seed --n_train 5000 # This takes 12GB GPU RAM
done


for seed in {12340..12345}
do
	python3 MyrtleGDNoise_SpeedLimit.py --train_seed $seed --n_train 10000 # This *probably* takes around 18GB GPU RAM
done


for seed in {12340..12345}
do
	python3 MyrtleGDNoise_SpeedLimit.py --train_seed $seed --n_train 500 --lr 1e-7 &
	python3 MyrtleGDNoise_SpeedLimit.py --train_seed $seed --n_train 1250 --lr 1e-7 &
	python3 MyrtleGDNoise_SpeedLimit.py --train_seed $seed --n_train 2500 --lr 1e-7 
done


for seed in {12340..12345}
do
	python3 MyrtleGDNoise_SpeedLimit.py --train_seed $seed --n_train 5000 --lr 1e-7
done


for seed in {12340..12345}
do
	python3 MyrtleGDNoise_SpeedLimit.py --train_seed $seed --n_train 10000 --lr 1e-7
done
