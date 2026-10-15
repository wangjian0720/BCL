# nvidia-smi

# - - - - - - - - - - - - - - - - - - - - - - - - #
file_name=TrainPROMISE   # run file name, e.g., mean_teacher_promise.py

# - - - - - - - - - - - - - - - - - - - - - - - - #

log_dir=./logs/${file_name}
mkdir -p ${log_dir}
version="s1"
numlb=7     # TBD: 3,7,14
gpuid=1


echo "Running script: ${file_name}.py"
echo "gpu_id=${gpuid}"

echo "exp ${file_name}/${version}"
echo "labeled_num ${numlb}"
echo "${log_dir}/${version}_${numlb}.log"

nohup python3 ${file_name}.py \
    --gpu_id=${gpuid} \
    --labeled_num ${numlb}\
    --exp ${file_name}/${version} \
    >${log_dir}/log_${version}_${numlb}.log 2>&1 &

