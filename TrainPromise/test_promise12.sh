# nvidia-smi



# - - - - - - - -     Testing      - - - - - - - # 

expname=TrainPROMISE   # run file name
version="s1"
numlb=7   #
gpuid=1


python3 ./performance_test_promise.py \
    --root_path /data/wj/promise/npy_image/ \
    --res_path ./results \
    --gpu_id=${gpuid} \
    --exp ${expname}/${version} \
    --flag_check_with_best_stu=True\
    --labeled_num ${numlb}\



# python3 ./performance_test_promise_unet_2d.py \
#     --root_path /data/wj/promise/npy_image/ \
#     --res_path ./results \
#     --gpu_id=${gpuid} \
#     --exp ${expname}/${version} \
#     --flag_check_with_best_stu=True\
#     --labeled_num ${numlb}\

# True使用学生, False使用教师

