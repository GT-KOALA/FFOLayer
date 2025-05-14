EPS=0.01
LR=0.001

for SEED in 1 # {1..10}
do
	for YDIM in  10 # 20 50 100 200
	do
		for METHOD in ffoqp #  cvxpylayer qpth
		do
			sbatch --export=METHOD=$METHOD,SEED=$SEED,YDIM=$YDIM,LR=$LR,EPS=$EPS scripts/ffoqp.sbatch
		done
	done
done
