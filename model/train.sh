
debug=$1
iters=$2

nohup python -u main.py -debug ${debug} -iters ${iters}  > ../trian_log/train.log 2>&1 &
