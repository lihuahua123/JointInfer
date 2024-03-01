host=$1
export INITIAL_PEERS=/ip4/192.168.249.122/tcp/41072/p2p/12D3KooWHScG77FBgkMbfwFR2gdms9asoXXytcZwbrunnvt7TAYG
ssh $host sudo tc qdisc delete dev eno1 root
ssh $host tc qdisc add dev eno1 root netem delay 50ms rate 0.1Gbit limit 2250000
python3.8 run_opt_requests.py --initial_peers $INITIAL_PEERS --prefix /home/server/models/OPT-30B -b 4 --num-micro-batches 2 --num-processes 1 --output petal_32_128_0 &
wait
# networks=(50 100 150 200)
# for network in ${networks[*]}
# do
# ssh $host tc qdisc add dev eno1 root netem delay ${network}ms rate 0.1Gbit limit 2250000
# echo "tc qdisc add dev eno1 root netem delay ${network}ms rate 0.1Gbit limit 2250000"
# python3.8 run_opt_requests.py --initial_peers $INITIAL_PEERS --prefix /home/server/models/OPT-30B -b 4 --num-micro-batches 2 --num-processes 1 --output petal_32_128_${network} &
# ssh $host sudo tc qdisc delete dev eno1 root
# wait
# done