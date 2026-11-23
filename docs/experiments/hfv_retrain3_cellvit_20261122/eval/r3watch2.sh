#!/bin/bash
# usage: r3watch2.sh "TILLICUM_JOBS" "HYAK_JOBS" "COLAB_LOGS"
# Exits with one line when: a Tillicum job leaves PENDING/RUNNING; a Hyak job changes state from its first-seen state
# (or leaves the queue); a Colab S3 log shows a DONE / FAIL / STOP / Traceback / rc != 0 line.
TJ=$1; HJ=$2; CL=$3
S3=s3://bagherilab-working/pohao/hfv_retrain3_cellvit_20261122/outputs
first=()
while true; do
  if [ -n "$TJ" ]; then
    t=$(ssh -o BatchMode=yes -o ConnectTimeout=15 tillicum "sacct -j $TJ -X -n -o JobID,State" 2>/dev/null | awk '{print $1":"$2}')
    bad=$(echo "$t" | grep -vE ':(RUNNING|PENDING)$' | grep -v '^$')
    [ -n "$bad" ] && { echo "$(date +%H:%M) tillicum: $(echo $t)"; exit 0; }
  fi
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=15 hyak "for j in ${HJ//,/ }; do echo J:\$j:\$(sacct -j \$j -X -n -o State | head -1 | tr -d ' '); done; module load coenv/aws/2.17.32 >/dev/null 2>&1; for l in $CL; do aws s3 cp --only-show-errors $S3/\$l - 2>/dev/null | grep -E '### .*(DONE|done|FAIL|STOP|rc [1-9])|Traceback|STOP' | tail -2 | sed \"s/^/C:\$l: /\"; done" 2>/dev/null)
  [ -z "$out" ] && { sleep 180; continue; }
  for line in $(echo "$out" | grep '^J:'); do
    j=$(echo $line | cut -d: -f2); s=$(echo $line | cut -d: -f3)
    [ -z "${first[$j]}" ] && first[$j]=$s
    [ "$s" != "${first[$j]}" ] && { echo "$(date +%H:%M) hyak $j: ${first[$j]} -> $s"; exit 0; }
  done
  c=$(echo "$out" | grep '^C:')
  [ -n "$c" ] && { echo "$(date +%H:%M) colab: $c"; exit 0; }
  sleep 180
done
