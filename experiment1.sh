# bash experiment1.sh 1 2 experiment_remote2pipe.sh 192.168.249.123
# bash experiment1.sh 1 4 experiment_remote4pipe.sh 192.168.249.124
# bash experiment1.sh 1 2 experiment_remote2pipeflexgen.sh 192.168.249.124
# bash experiment1.sh 1 2 experiment.sh 192.168.249.124
# bash experiment1.sh 1 2 experiment_compress.sh 192.168.249.123
tp=$1
pp=$2
script=$3
host=$4
# ssh $host sudo tc qdisc delete dev eno1 root
# ssh $host tc qdisc add dev eno1 root netem delay 50ms rate 0.1Gbit limit 2250000
# # bash $script $pp $tp $host "different_output_one_GPU_24batch"
# bash $script $pp $tp $host

ssh $host sudo tc qdisc delete dev eno1 root
i=0
while [ $i -lt 1 ]
do
bash $script $pp $tp $host "32_512_0_$i"
let i++ 
done
ssh $host sudo tc qdisc delete dev eno1 root

networks=(50 100 150 200)
for network in ${networks[*]}
do
ssh $host tc qdisc add dev eno1 root netem delay ${network}ms rate 0.1Gbit limit 2250000
echo "tc qdisc add dev eno1 root netem delay ${network}ms rate 0.1Gbit limit 2250000"
i=0
    while [ $i -lt 1 ]
    do
    bash $script $pp $tp $host "32_512_${network}_${i}"
    let i++ 
    done
ssh $host sudo tc qdisc delete dev eno1 root
wait
done



