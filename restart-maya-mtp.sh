#!/bin/bash
# boot Maya's server + engine on the MTP config (kept as a FILE: a heredoc copy of this text on a
# shell's command line matches the pgrep patterns below and kills that shell)
cd /home/gopi/project-maya || exit 1
LOG=maya-maya-l.log
for p in $(pgrep -f 'serve/server[.]py'); do kill -TERM "$p" 2>/dev/null; done
sleep 3
for p in $(pgrep -f 'serve/server[.]py'); do kill -9 "$p" 2>/dev/null; done
for p in $(pgrep -f '^/home/gopi/project-maya/build/strata'); do kill -TERM "$p" 2>/dev/null; done
sleep 5
for p in $(pgrep -f '^/home/gopi/project-maya/build/strata'); do kill -9 "$p" 2>/dev/null; done
sleep 3
# wait for the cards to actually drain before anything measures free VRAM: the 10:22 boot on 2026-10-16 read
# CUDA2/3 while they still held the dead engine's 7559 MiB and sized 768-token chunks against memory that came
# back seconds later (the same config chose its usual 8192 on sixteen earlier boots)
for i in $(seq 1 60); do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  [ "$m" -lt 1200 ] && break
  sleep 2
done
echo "left: $(pgrep -f '^/home/gopi/project-maya/build/strata' | wc -l) engine, $(pgrep -f 'serve/serve[r].py' | wc -l) server"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
n0=$(grep -c "serve (M3.1)" "$LOG")
echo "MARK=$n0" > /tmp/maya-restart-mark
setsid nohup .venv/bin/python serve/server.py --engine strata --config /home/gopi/project-maya/maya-maya-l.json --port 8080 >> maya-maya-l.server.log 2>&1 < /dev/null &
echo "launched server pid $!"
for i in $(seq 1 200); do
  n=$(grep -c "serve (M3.1)" "$LOG")
  if [ "$n" -gt "$n0" ]; then echo "READY $(date +%H:%M:%S)"; exit 0; fi
  sleep 5
done
echo "TIMEOUT waiting for READY"
