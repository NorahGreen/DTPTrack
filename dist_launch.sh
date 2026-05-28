#!/bin/bash

ssh_port=22

cleanup_command=""
if [[ "$1" == "--cleanup" ]]; then
    cleanup_command="tmux send-keys -t sess q; tmux send-keys -t sess C-c; tmux send-keys -t sess C-c; sleep 2; pkill python; sleep 10; pkill -9 python; sleep 2;"
    shift
fi

n=$1
shift

fqdns=("${@:1:$n}")
commands=("${@:$((n+1))}")
NUM_NODES=${#fqdns[@]}
DATE_WITH_TIME=`date "+%Y.%m.%d-%H.%M.%S-%6N"`
for index in "${!fqdns[@]}"; do
    command="$cleanup_command tmux send-keys -t sess \"NODE_RANK=$index NUM_NODES=$NUM_NODES MASTER_ADDRESS=${fqdns[0]} DATE_WITH_TIME=$DATE_WITH_TIME ${commands[*]}\" Enter"
    echo "[${fqdns[$index]}:$ssh_port]: run \"$command\""
    ssh -o StrictHostKeyChecking=no "root@${fqdns[$index]}" -p $ssh_port "$command" &
done
wait
