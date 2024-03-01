export INITIAL_PEERS=/ip4/192.168.249.122/tcp/41072/p2p/12D3KooWHScG77FBgkMbfwFR2gdms9asoXXytcZwbrunnvt7TAYG
nohup python3.8 -m petals.cli.run_server --converted_model_name_or_path /home/server/models/OPT-30B --initial_peers $INITIAL_PEERS --device cuda:0 --block_indices 0:12 &
nohup python3.8 -m petals.cli.run_server  --converted_model_name_or_path /home/server/models/OPT-30B --initial_peers $INITIAL_PEERS --device cuda:1 --block_indices 12:24 &
nohup python3.8 -m petals.cli.run_server  --converted_model_name_or_path /home/server/models/OPT-30B --initial_peers $INITIAL_PEERS --device cuda:2 --block_indices 24:36 &
nohup python3.8 -m petals.cli.run_server  --converted_model_name_or_path /home/server/models/OPT-30B --initial_peers $INITIAL_PEERS --device cuda:3 --block_indices 36:48 &

#python3.8 run_opt_requests.py --initial_peers $INITIAL_PEERS --prefix /home/server/models/OPT-30B -b 4 --num-micro-batches 2 --num-processes 1 --output out_30b.tsv