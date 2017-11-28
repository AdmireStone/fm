iter_num=$1
total_iters=$2
flod_name="u1"
eval_num=500
trained_models=$3
weightpath=/home/zju/dgl/dataset/recommend/ml-100k/${trained_models}/
result_folder=../result/${flod_name}/${trained_models}/complete
eval_programm="../source/eval_100k.py"

if [ ! -d ${result_folder} ]; then
  mkdir -p ${result_folder}
fi

for((iter_num=${iter_num};iter_num<${total_iters};iter_num++));
do 
nohup python -u ${eval_programm}  -flod_name ${flod_name}  -iter_num ${iter_num} -eval_num ${eval_num}  -weightpath ${weightpath}   > ${result_folder}"/"${iter_num}".log"  2>&1 &
done
