ps -ef|grep dist_flex_opt|grep -v grep|cut -c 9-15|xargs kill -9
CMD="ps -ef|grep dist_flex_opt|grep -v grep|cut -c 9-15|xargs kill -9"
ssh -p 28 root@192.168.249.125 $CMD
