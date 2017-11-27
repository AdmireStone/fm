total_iters=$1
iter_num=$2
flod_name="u1"
eval_num=500
weightpath="/home/zju/dgl/dataset/recommend/ml-100k/models_11_23_20/"

result_folder="../result/"${flod_name}"/complete"
cd mkdir ${result_folder}

for((iter_num=${iter_num};iter_num<${total_iters};iter_num++));
do 
nohup python -u ../source/eval_100k.py  -flod_name ${flod_name}  -iter_num ${iter_num} -eval_num ${eval_num}  -weightpath ${weightpath}   > ${result_folder}"/"${iter_num}".log"  2>&1 &
done
