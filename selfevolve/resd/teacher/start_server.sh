export PROXY_FRONTEND_PORT=15555
export PROXY_BACKEND_PORT=15556

BACKEND=vllm
CKPT_PATH=$1 # path to the teacher checkpoint

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    TP_SIZE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')
else
    TP_SIZE=1
fi

wait_server_ready() {
    server=$1
    ip=$2
    port=$3
    while true; do
        echo "wait $server server ready at $ip:$port..."
        result=`echo -e "\n" | telnet $ip $port 2> /dev/null | grep Connected | wc -l`
        if [ $result -eq 1 ]; then
            break
        else
            sleep 1
        fi
    done
}

ps -ef | grep "python proxy.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9
ps -ef | grep "python worker.py" | grep -v grep | awk -F ' ' '{print $2}' | xargs -r kill -9

nohup python proxy.py &> proxy.log &

wait_server_ready proxy localhost $PROXY_BACKEND_PORT

echo "teacher proxy is ready"

nohup python worker.py --backend $BACKEND --tp-size $TP_SIZE --n-logprobs 256 --ckpt-path $CKPT_PATH &> worker.log &
echo "start teacher worker"

echo "teacher server is ready"