#!/bin/bash
# Launch the full FLIP pipeline for each noise cGAN, one machine per noise type, in tmux session "flip_<variant>".
#   cher (local) : cgan_resnet32      vosges : cgan_resnet18
#   suede        : cgan_efficientnet  mouette: cgan_nosurrogate
# Protocol: expert + user = ResNet-32, 5 experts x 20 epochs, 5 user runs x 100 epochs,
# budgets 300 500 1000 1500, truck (9) -> deer (4), |delta| <= 6/255 (noise_generator/cgan, unconditional D).
# Remote machines run from an isolated copy /Data/nadine.hc/FLIP_cgan (code + cGAN checkpoints,
# data symlinked), so their existing FLIP folders are left untouched. Everything is resumable:
# re-running the launcher skips finished stages.
#   ./launch_cgan_flip.sh            # launch all
#   ./launch_cgan_flip.sh vosges     # launch one host
#   ./launch_cgan_flip.sh --pull     # copy remote results back into experiments/ here
set -e
SRC=/Data/nadine.hc/FLIP
RDIR=/Data/nadine.hc/FLIP_cgan
declare -A VARIANT=([cher]=cgan_resnet32 [vosges]=cgan_resnet18 [suede]=cgan_efficientnet [mouette]=cgan_nosurrogate)
ARGS="--n-experts 5 --n-user-runs 5 --expert-epochs 20 --user-epochs 100 --model r32p --budgets 300 500 1000 1500 --source 9 --target 4 --user-seed 100000"

if [ "$1" = --pull ]; then
  for host in vosges suede mouette; do
    v=${VARIANT[$host]}
    rsync -a $host:$RDIR/experiments/flip_$v $SRC/experiments/ && echo "pulled experiments/flip_$v from $host"
    mkdir -p $SRC/out/checkpoints/flip_$v
    rsync -a --include '*/' --include 'model_20_200.pth' --exclude '*' \
      $host:$RDIR/out/checkpoints/flip_$v/ $SRC/out/checkpoints/flip_$v/ 2>/dev/null || true
  done
  exit 0
fi

for host in ${@:-cher vosges suede mouette}; do
  v=${VARIANT[$host]}
  if [ "$host" = cher ]; then
    cmd="cd $SRC && $SRC/.venv/bin/python run_full_flip.py --name flip_$v --poisoner $v $ARGS 2>&1 | tee -a logs/flip_$v.log"
    tmux new -d -s flip_$v "$cmd"
  else
    ssh -o BatchMode=yes $host "mkdir -p $RDIR/noise_generator/cgan $RDIR/logs $RDIR/experiments && ln -sfn /Data/nadine.hc/FLIP/data $RDIR/data"
    rsync -a --exclude __pycache__ $SRC/modules $SRC/schemas $SRC/run_full_flip.py $SRC/run_experiment.py $host:$RDIR/
    rsync -a $SRC/noise_generator/cgan/$v.pth $SRC/noise_generator/cgan/${v}_cache.npz $host:$RDIR/noise_generator/cgan/
    cmd="cd $RDIR && /Data/nadine.hc/.venv/bin/python run_full_flip.py --name flip_$v --poisoner $v $ARGS 2>&1 | tee -a logs/flip_$v.log"
    ssh -o BatchMode=yes $host "tmux new -d -s flip_$v $(printf '%q' "$cmd")"
  fi
  echo "$host: started tmux session flip_$v"
done
