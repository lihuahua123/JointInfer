# scp -r ./experimental/my_cost_model.py root@192.168.249.123:/home/server/DistributedOffload/experimental/
w_gpu=20
w_cpu=80
c_gpu=20
c_cpu=80
a_gpu=100
a_cpu=0
pp=1
num_inner_iterations=2 #$pp
gen_len=128
prompt_len=512
num_prompts=8
overlap=True
comm_device=gpu
model=/home/server/models/OPT-30B
PYTHON_EXEC=/usr/bin/python3.8
compress="--compress-weight"
tp_nums=(2)
tp_num=2
$PYTHON_EXEC ./joint_infer.py  $compress --profile --head-ip $1 --port 7733 --world-size 2 --rank 0 --local-rank 2 --model $model --local --percent $w_gpu $w_cpu $c_gpu $c_cpu $a_gpu $a_cpu --comm-device $comm_device --gen-len  $gen_len --prompt-len $prompt_len --num-prompts $num_prompts --pp $pp --tp $tp_num --overlap $overlap --num-inner-iterations $num_inner_iterations & 
$PYTHON_EXEC ./joint_infer.py  $compress --profile --head-ip $1 --port 7733 --world-size 2 --rank 1 --local-rank 3 --model $model --local --percent $w_gpu $w_cpu $c_gpu $c_cpu $a_gpu $a_cpu --comm-device $comm_device --gen-len  $gen_len --prompt-len $prompt_len --num-prompts $num_prompts --pp $pp --tp $tp_num --overlap $overlap --num-inner-iterations $num_inner_iterations & 
# for tp_num in ${tp_nums[*]}
# do 
#     # python3 ./flexgen/flex_opt.py --model /home/server/models/OPT-30B --local --percent 24 76 55 44 100 0 --gen-len $gen_len --prompt-len $prompt_len --overlap $overlap --num-hidden-layers $num >> test_layer.log 
#     i=0  
#     while [ $i -lt $tp_num ] 
#     do
#     $PYTHON_EXEC ./joint_infer.py  $compress --profile --head-ip $1 --port 7777 --world-size $tp_num --rank $i --local-rank $i --model $model --local --percent $w_gpu $w_cpu $c_gpu $c_cpu $a_gpu $a_cpu --comm-device $comm_device --gen-len  $gen_len --prompt-len $prompt_len --num-prompts $num_prompts --pp $pp --tp $tp_num --overlap $overlap --num-inner-iterations $num_inner_iterations & 
#     let i++ 
#     done
#     wait
#     rm ~/flexgen_offload_dir/*
# done