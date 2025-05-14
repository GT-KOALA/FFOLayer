EPS=0.1
LR=0.001

for YDIM in  10 20 50 100 200 500
do
	for METHOD in ffoqp # cvxpylayer qpth
	do
        for SEED in {1..10}
        do
			sbatch --export=METHOD=$METHOD,SEED=$SEED,YDIM=$YDIM,LR=$LR,EPS=$EPS scripts/ffoqp.sbatch
        done
	done
done
