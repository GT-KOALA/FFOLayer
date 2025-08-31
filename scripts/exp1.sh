EPS=0.1
LR=0.00001
BATCH_SIZE=32

for YDIM in 200 # 10 20 50 100 200 500
do
	for METHOD in qpth ffoqp ffoqp_eq_cst # cvxpylayer qpth ffoqp ffoqp_eq_cst ffoqp_eq_cst_pdipm ffoqp_eq_cst_parallelize
	do
        for SEED in {1..1}
        do
			sbatch --export=METHOD=$METHOD,SEED=$SEED,YDIM=$YDIM,LR=$LR,EPS=$EPS,BATCH_SIZE=$BATCH_SIZE ffoqp_zihao.sbatch
        done
	done
done
