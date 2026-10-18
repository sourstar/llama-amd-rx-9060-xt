#!/bin/sh
# Throughput on the RX 9060 XT with the B70's configuration, one row per
# client concurrency into baselines/history.csv, then the correctness gate
# (capture twice in shuffled order, bit-exact compare). Runs from the repo
# root in Git Bash; Python runs in louise-rag's image (no host Python).
#
#   sh scripts/run_bench.sh ["note"]
#
# ONE JOB ON THE CARD: nothing else may be loaded or running while this runs.
set -eu
cd "$(dirname "$0")/.."
. ./config.env
NOTE=${1:-"B70 flags: -c 8192 -np 1 -ub 8192 fa on, ROCBLAS_DEFAULT_ATOMICS_MODE=0"}
export MSYS_NO_PATHCONV=1
HERE=$(pwd -W 2>/dev/null || pwd)
PYIMG=louise-rag:local
py() {
  docker run --rm -v "$HERE:/w" -w /w -e HEALTH_PATH="$HEALTH_PATH" -e EMBED_MODEL="$EMBED_MODEL" \
    -e IMAGE="$IMAGE" "$PYIMG" python "$@"
}
URL_C=$(echo "$URL" | sed 's#127.0.0.1#host.docker.internal#; s#localhost#host.docker.internal#')

cp localai/$EMBED_MODEL.yaml "$MODELS_DIR/$EMBED_MODEL.yaml"
trap 'rm -f "$MODELS_DIR/$EMBED_MODEL.yaml"; docker restart "$LOCALAI_CONTAINER" >/dev/null' EXIT
docker restart "$LOCALAI_CONTAINER" >/dev/null
until curl -sf "$URL$HEALTH_PATH" >/dev/null; do sleep 2; done

for c in $CONCURRENCY; do
  py scripts/bench.py --url "$URL_C" --fixtures "$BENCH_FIXTURES" --concurrency "$c" \
     --passes "$PASSES" --note "$NOTE, client concurrency $c" --image "$IMAGE"
done
py scripts/capture.py --url "$URL_C" --fixtures "$FIXTURES" --out baselines/gate-last.json \
   --repeat 2 --pattern shuffle --seed 7 --quiet --label "$NOTE"
py scripts/compare.py --determinism baselines/gate-last.json | tail -3
