# hfv_retrain_20261118 PQ robustness readouts: source from a job script. Unpacks R/env2/sdenv.tar (eval/robust_env.slurm)
# to /tmp/rob_sdenv once per node (flock: several array tasks can share a node), then sets PY.
R=/gscratch/cheme/chiu/hfv_robust_20261120
(
  flock -w 900 9 || { echo "STOP flock"; exit 1; }
  if [ ! -f /tmp/rob_sdenv/.unpacked ]; then
    rm -rf /tmp/rob_sdenv && tar -xf $R/env2/sdenv.tar -C /tmp && touch /tmp/rob_sdenv/.unpacked || { echo "STOP unpack"; exit 1; }
  fi
) 9>/tmp/rob_sdenv.lock || exit 1
PY=/tmp/rob_sdenv/bin/python
export KERAS_HOME=$R/keras_home TF_CPP_MIN_LOG_LEVEL=2 MPLBACKEND=Agg
