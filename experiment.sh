# bash experiment.sh 2 1 192.168.249.124
# PP=4 
# w_gpu=85 # 这些百分比随便填
# w_cpu=15 #15
# c_gpu=100 # GPU80 disk 20 居然不行
# c_cpu=0
# a_gpu=100
# a_cpu=0
# PP=2
# w_gpu=50
# w_cpu=50 
# c_gpu=100 # GPU80 disk 20 居然不行
# c_cpu=0
# a_gpu=100
# a_cpu=0
# PP=1
# w_gpu=37 # 这些百分比随便填
# w_cpu=63 #15
# c_gpu=100 # GPU80 disk 20 居然不行
# c_cpu=0
# a_gpu=100
# a_cpu=0
w_gpu=50 #50 # 这些百分比随便填
w_cpu=50 #50
c_gpu=100 # GPU80 disk 20 居然不行
c_cpu=0
a_gpu=100
a_cpu=0
pp=$1
tp=$2
word_size=`expr $pp \* $tp`
num_inner_iterations=2 # num_inner_iterations = 1 只有这样，张量并行才比流水线并行强
script=./joint_infer.py
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
#tensor_parallelism=""
compress="--compress-weight --compress-cache"
memory_fraction=0.5
#compress=""
i=0
by_layer=False
log_file="one_node_$pp-$tp-$w_gpu-$w_cpu-$c_gpu-$c_cpu_$4"
if [ $word_size -lt 5 ]; 
    then
    while [ $i -lt $word_size ] 
    do
    $PYTHON_EXEC $script $tensor_parallelism --log-file $log_file $compress --memory-fraction $memory_fraction --by-layer $by_layer --head-ip $master_host --port 7777 --world-size $word_size --rank $i --local-rank $i --model $model --local --percent $w_gpu $w_cpu $c_gpu $c_cpu $a_gpu $a_cpu --comm-device gpu --gen-len  $gen_len --prompt-len $prompt_len --num-prompts $num_prompts --pp $pp --tp $tp --overlap $overlap --async-comm --num-inner-iterations $num_inner_iterations &
    let i++ 
    done
else
    while [ $i -lt 4 ] 
        do
        NCCL_NET=Socket $PYTHON_EXEC $script --log-file $log_file --memory-fraction $memory_fraction $tensor_parallelism $compress --by-layer $by_layer --head-ip $master_host --port 7777 --world-size $word_size --rank $i --local-rank $i --model $model --local --percent $w_gpu $w_cpu $c_gpu $c_cpu $a_gpu $a_cpu --comm-device gpu --gen-len  $gen_len --prompt-len $prompt_len --num-prompts $num_prompts --pp $pp --tp $tp --overlap $overlap --async-comm --num-inner-iterations $num_inner_iterations &
        let i++ 
        done
    CMD="cd /home/server/DistributedOffload ; bash &&  "
    local_rank=`expr $word_size - 4`
    j=0
    i=4
    while [ $j -lt $local_rank ] 
        do
        CMD+="nohup $PYTHON_EXEC $script --log-file $log_file --memory-fraction $memory_fraction $tensor_parallelism $compress --by-layer $by_layer --head-ip $master_host --port 7777 --world-size $word_size --rank $i --local-rank $j --model $model --local --percent $w_gpu $w_cpu $c_gpu $c_cpu $a_gpu $a_cpu --comm-device gpu --gen-len  $gen_len --prompt-len $prompt_len --num-prompts $num_prompts --pp $pp --tp $tp --overlap $overlap --async-comm --num-inner-iterations $num_inner_iterations & "
        let j++ 
        let i++
        done
    echo $CMD
    ssh -p 28 root@192.168.249.125 $CMD
fi
wait
