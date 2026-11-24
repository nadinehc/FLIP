#!/usr/bin/env bash
# Learned-trigger study (CIFAR-10, truck 9 -> deer 4, ResNet-32 expert and
# victim, L_inf <= 6/255).
#
# Modes:
#   bash sweep/seed0_pipeline.sh smoke                  # ~5 min sanity check
#   bash sweep/seed0_pipeline.sh finetune               # 4 trigger configs, budget 1500
#   bash sweep/seed0_pipeline.sh pick                   # print the winning letter
#   bash sweep/seed0_pipeline.sh final <A|B|C|D> [seed] # full protocol, one seed
#   bash sweep/seed0_pipeline.sh overnight              # smoke -> finetune -> pick -> final seeds 0..4
#
# Environment:
#   PYTHON=/path/to/python   interpreter with torch + CUDA (default: python)
#   DRY=1                    print the commands without running them
#
# Every step resumes from its outputs, so re-running a mode after an
# interruption skips work that already finished.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
DRY=${DRY:-0}
FINAL_SEEDS=${FINAL_SEEDS:-"0 1 2 3 4"}

# Fine-tuning grid: one set of extra trigger flags per letter.
# lambda_div pushes different images toward different perturbations;
# adaptive_floor scales the bound by a texture map; generator-lr is g_lr.
config_flags() {
  case "$1" in
    A) echo "--lambda-div 0.0 --generator-lr 1e-3" ;;
    B) echo "--lambda-div 1.0 --div-tau 0.3 --generator-lr 1e-3" ;;
    C) echo "--lambda-div 1.0 --div-tau 0.3 --adaptive-floor 0.25 --generator-lr 1e-3" ;;
    D) echo "--lambda-div 1.0 --div-tau 0.3 --adaptive-floor 0.25 --generator-lr 2e-4" ;;
    *) return 1 ;;
  esac
}

run() {
  echo "+ $*"
  if [ "$DRY" = "1" ]; then return 0; fi
  "$@"
}

# Triggers are trained for each seed with the same settings.
trigger_args() {
  echo "--epsilons 6 --norm linf --arch vit --trigger-epochs 20 --seed $1"
}

# Picks the fine-tuning winner at budget 1500. Rule: among configs whose GAN
# trigger keeps CTA within 2 points of that config's own sinusoidal baseline,
# take the highest PTA (ties go to higher CTA). If none qualifies, take the
# highest PTA overall and say so on stderr.
pick_config() {
  "$PYTHON" - <<'EOF'
import csv, sys
from pathlib import Path

best, fallback = None, None
for letter in "ABCD":
    path = Path(f"out/sweep/ft_s0_{letter}/results.csv")
    if not path.exists():
        print(f"missing {path}", file=sys.stderr)
        continue
    rows = [r for r in csv.DictReader(open(path))
            if r["m"] == "1500"]
    base = [r for r in rows if r["trigger_kind"] == "hand-coded"]
    gan = [r for r in rows if r["trigger_kind"] != "hand-coded"]
    if not base or not gan:
        print(f"incomplete results for {letter}", file=sys.stderr)
        continue
    base_cta = float(base[0]["cta"])
    cta, pta = float(gan[0]["cta"]), float(gan[0]["pta"])
    key = (pta, cta)
    if fallback is None or key > fallback[1]:
        fallback = (letter, key)
    if cta >= base_cta - 0.02 and (best is None or key > best[1]):
        best = (letter, key)

if best is not None:
    print(best[0])
elif fallback is not None:
    print("no config keeps CTA within 2 points; using best PTA", file=sys.stderr)
    print(fallback[0])
else:
    sys.exit("no fine-tuning results found")
EOF
}

final_run() {
  local letter=$1 seed=$2
  local flags
  flags=$(config_flags "$letter")
  echo "=== final run, config $letter, seed $seed ($flags) ==="
  # shellcheck disable=SC2086
  run "$PYTHON" sweep/run_trigger_sweep.py --tag "final_s${seed}_$letter" \
      $(trigger_args "$seed") $flags \
      --budgets 300 500 1000 1500 --experts 5 --expert-epochs 20 \
      --user-epochs 100
}

finetune_run() {
  for letter in A B C D; do
    local flags
    flags=$(config_flags "$letter")
    echo "=== fine-tune config $letter ($flags) ==="
    # shellcheck disable=SC2086
    run "$PYTHON" sweep/run_trigger_sweep.py --tag "ft_s0_$letter" \
        $(trigger_args 0) $flags \
        --budgets 1500 --experts 2 --expert-epochs 20 --user-epochs 40 \
        --attack-iterations 15
  done
}

case "${1:-}" in
  smoke)
    run "$PYTHON" sweep/run_trigger_sweep.py --tag smoke_s0 --smoke --seed 0
    ;;

  finetune)
    finetune_run
    echo "Compare: out/sweep/ft_s0_{A,B,C,D}/results.md (budget 1500)."
    ;;

  pick)
    pick_config
    ;;

  final)
    letter=${2:-}
    seed=${3:-0}
    if ! config_flags "$letter" >/dev/null; then
      echo "usage: $0 final <A|B|C|D> [seed]" >&2
      exit 1
    fi
    final_run "$letter" "$seed"
    echo "Results: out/sweep/final_s${seed}_$letter/results.md"
    ;;

  overnight)
    # Smoke test first: a failure here should stop everything.
    run "$PYTHON" sweep/run_trigger_sweep.py --tag smoke_s0 --smoke --seed 0
    finetune_run
    letter=$(pick_config)
    echo "=== fine-tuning winner: $letter ==="
    # Keep going if one seed fails, so the other seeds still run.
    for seed in $FINAL_SEEDS; do
      if final_run "$letter" "$seed"; then
        echo "=== seed $seed finished ==="
      else
        echo "=== seed $seed FAILED; continuing ===" >&2
      fi
    done
    echo "All seeds attempted. Results: out/sweep/final_s<seed>_$letter/results.md"
    ;;

  *)
    echo "usage: $0 {smoke|finetune|pick|final <A|B|C|D> [seed]|overnight}" >&2
    exit 1
    ;;
esac
