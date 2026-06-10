# nvidia-smi


# - - - - - - - -     Testing      - - - - - - - # 
expname="TrainACDC"
version="s0"
numlb=7    # TBD 3, 7, 14
gpuid=1


python3 ./performance_test_acdc.py \
    --root_path /data/wj/ACDC \
    --res_path ./results \
    --gpu_id=${gpuid} \
    --exp ${expname}/${version} \
    --flag_check_with_best_stu=False\
    --labeled_num ${numlb}\
