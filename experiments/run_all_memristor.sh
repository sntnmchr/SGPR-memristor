#!/bin/bash
# Re-measurement of the memristor experiments on one machine.
#
#   bash experiments/run_all_memristor.sh            # all the steps
#   bash experiments/run_all_memristor.sh 1 2        # only the given steps
#
# Steps
#   1  SGPR (RBF), Z = 50, 200, 600, settings of sgpr_train_*.py
#   2  GPR (all the training points)
#   3  Matern 3/2, Matern 5/2, and RQ kernels, Z = 50, 200, 600
#   4  SGPR (RBF) without optimization
#   5  archive of the results
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
MR_ROOT=${MR_ROOT:-$(cd "$HERE/.." && pwd)}
THREADS=${THREADS:-10}; REPEAT=${REPEAT:-3}; HSPICE=${HSPICE:-hspice}; PY=${PY:-python3}
OUT=${OUT:-$MR_ROOT/results/rerun}; mkdir -p "$OUT"
DRY=${DRY:-0}; D1=""; [ "$DRY" = 1 ] && D1="--dry-run"          # DRY=1: no HSPICE (check of the scripts)
STEPS=${*:-1 2 3 4 5}
SW="$PY $HERE/run_memristor_sweep.py --root $MR_ROOT --threads $THREADS --repeat $REPEAT --hspice $HSPICE --out $OUT/sweep.csv $D1"
step() { echo; echo "######## step $1 : $2  ($(date '+%F %T'))"; }

{ echo "date: $(date)"; echo "host: $(hostname)"; lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket"; free -g | sed -n 1,2p
  $PY -c "import sys,torch,pyro,numpy,pandas;print('python',sys.version.split()[0],'torch',torch.__version__,'pyro',pyro.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__)"
  echo "threads: $THREADS  hspice repeat: $REPEAT"; } > "$OUT/env.txt" 2>&1

for s in $STEPS; do case $s in
1) step 1 "SGPR (RBF)";            $SW --Z 50 200 600 || exit 1 ;;
2) step 2 "GPR";                   $SW --model gpr || exit 1 ;;
3) step 3 "other kernels";         $SW --Z 50 200 600 --kernels Matern32 Matern52 RQ || exit 1 ;;
4) step 4 "without optimization";  $SW --Z 50 200 600 --stop none || exit 1 ;;
5) step 5 "archive"
   mkdir -p $OUT/data
   for r in RBF_Z50_neg RBF_Z200_neg RBF_Z600_neg GPR_RBF_neg; do for d in HSPICE_train HSPICE_test; do
     cp $MR_ROOT/runs/$r/$d/output/sim_data.csv $OUT/data/${r}_${d}.csv 2>/dev/null; done; done
   tar czf $OUT/../memristor_results.tar.gz -C $OUT . && echo "archive: $(cd $OUT/.. && pwd)/memristor_results.tar.gz" ;;
*) echo "unknown step $s"; exit 1 ;;
esac; done
echo; echo "finished ($(date '+%F %T'))"
