EPS=0.1
LR=0.00001

for YDIM in 10 20 50 100 200 # 500
do
	for METHOD in ffoqp_eq_cst ffoqp_eq_cst_pdipm ffoqp_eq_cst_parallelize
	do
        for SEED in {1..3}
        do
			sbatch --export=METHOD=$METHOD,SEED=$SEED,YDIM=$YDIM,LR=$LR,EPS=$EPS scripts/ffoqp.sbatch
        done
	done
done
