w_gpu=67
w_cpu=33
c_gpu=0 # GPU80 disk 20 居然不行
c_cpu=100
a_gpu=100
a_cpu=0
pp=$1
tp=$2
word_size=`expr $pp \* $tp`
num_inner_iterations=2 # num_inner_iterations = 1 只有这样，张量并行才比流水线并行强
script=./joint_infer.py #./dist_flex_opt.py
overlap=True
async_comm=True
model=/home/server/models/OPT-30B
master_host=$3
PYTHON_EXEC=/usr/bin/python3.8
tensor_ranks="[[0,1,2,3],[4,5]]"
pipeline_ranks="[[0,4],[1,4],[2,5],[3,5]]"
tensor_parallelism="--tensor-ranks $tensor_ranks --pipeline-ranks $pipeline_ranks"
tensor_parallelism=""
num_prompts=8
prompt_len=32
gen_len=128
compress="--compress-weight --compress-cache"
memory_fraction=0.5
#compress=""
i=0
by_layer=False
log_file="flexgen_$pp-$tp-$w_gpu-$w_cpu-$c_gpu-$c_cpu-$4"
NCCL_NET=Socket $PYTHON_EXEC $script --log-file $log_file --memory-fraction $memory_fraction $tensor_parallelism $compress --by-layer $by_layer --head-ip $master_host --port 7779 --world-size $word_size --rank 0 --local-rank 2 --model $model --local --percent $w_gpu $w_cpu $c_gpu $c_cpu $a_gpu $a_cpu --comm-device gpu --gen-len  $gen_len --prompt-len $prompt_len --num-prompts $num_prompts --pp $pp --tp $tp --overlap $overlap --async-comm --num-inner-iterations $num_inner_iterations &

CMD="cd /home/server/DistributedOffload ; bash &&  "
CMD+="NCCL_NET=Socket $PYTHON_EXEC $script --log-file $log_file --memory-fraction $memory_fraction $tensor_parallelism $compress --by-layer $by_layer --head-ip $master_host --port 7779 --world-size $word_size --rank 1 --local-rank 1 --model $model --local --percent $w_gpu $w_cpu $c_gpu $c_cpu $a_gpu $a_cpu --comm-device gpu --gen-len  $gen_len --prompt-len $prompt_len --num-prompts $num_prompts --pp $pp --tp $tp --overlap $overlap --async-comm --num-inner-iterations $num_inner_iterations & "
echo $CMD
ssh -p 28 root@192.168.249.125 $CMD
wait