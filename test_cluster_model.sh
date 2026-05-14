#!/bin/bash
# Test suite for cluster_model.sh
# Uses --print-config and --print-raw to exercise all logic locally without SSH/network.

set -uo pipefail

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cluster_model.sh"

# --------------------------------------------------------------------------
# Setup: temp dirs and command stubs
# --------------------------------------------------------------------------
WORK_DIR="$(mktemp -d)"
FAKE_MODEL="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR" "$FAKE_MODEL"' EXIT

STUB_BIN="${WORK_DIR}/bin"
mkdir -p "$STUB_BIN"

for cmd in ssh scp; do
    printf '#!/bin/bash\nexit 0\n' > "${STUB_BIN}/${cmd}"
    chmod +x "${STUB_BIN}/${cmd}"
done

# curl stub returns minimal valid JSON for health check / status
cat > "${STUB_BIN}/curl" <<'STUB'
#!/bin/bash
echo '{"object":"list","data":[{"id":"test-model"}]}'
STUB
chmod +x "${STUB_BIN}/curl"

export PATH="${STUB_BIN}:${PATH}"

PASS=0; FAIL=0; declare -a ERRORS=()

# --------------------------------------------------------------------------
# Test helpers
# --------------------------------------------------------------------------
_pass() { printf 'PASS  %s\n' "$1"; PASS=$((PASS + 1)); }
_fail() { printf 'FAIL  %s\n' "$1"; FAIL=$((FAIL + 1)); ERRORS+=("$1"); }

# Run script in WORK_DIR so vllm.service.tmp lands there, not in the repo.
run_() { (cd "$WORK_DIR" && bash "$SCRIPT" "$@"); }

ok() {
    local desc="$1"; shift
    if run_ "$@" >/dev/null 2>&1; then _pass "$desc"; else _fail "$desc"; fi
}

fails() {
    local desc="$1"; shift
    if ! run_ "$@" >/dev/null 2>&1; then _pass "$desc"; else _fail "$desc"; fi
}

has() {
    local desc="$1" pattern="$2"; shift 2
    local out; out="$(run_ "$@" 2>&1)" || true
    if echo "$out" | grep -qF -- "$pattern"; then
        _pass "$desc"
    else
        _fail "$desc  [wanted: '$pattern']"
    fi
}

no() {
    local desc="$1" pattern="$2"; shift 2
    local out; out="$(run_ "$@" 2>&1)" || true
    if ! echo "$out" | grep -qF -- "$pattern"; then
        _pass "$desc"
    else
        _fail "$desc  [must not contain: '$pattern']"
    fi
}

M="${FAKE_MODEL}"   # short alias for model path

# --------------------------------------------------------------------------
# Group 1: Shell syntax
# --------------------------------------------------------------------------
printf '\n=== Group 1: Shell syntax ===\n'

if bash -n "$SCRIPT" 2>/dev/null; then
    _pass "bash -n: no syntax errors"
else
    _fail "bash -n: syntax errors found"
fi

# --------------------------------------------------------------------------
# Group 2: Required arguments
# --------------------------------------------------------------------------
printf '\n=== Group 2: Required arguments ===\n'

fails "start without --model exits non-zero"                         start
fails "start with nonexistent model path exits non-zero"             start --model /no/such/path/model
ok    "start with real model path and --print-config exits 0"        start --model "$M" --print-config
fails "no subcommand exits non-zero"
fails "unknown subcommand exits non-zero"                            badcmd

# --------------------------------------------------------------------------
# Group 3: Numeric parameter validation
# --------------------------------------------------------------------------
printf '\n=== Group 3: Numeric parameter validation ===\n'

fails "--tp non-integer rejected"              start --model "$M" --tp abc     --print-config
fails "--pp non-integer rejected"              start --model "$M" --pp abc     --print-config
fails "--tp 0 rejected (must be >= 1)"         start --model "$M" --tp 0       --print-config
fails "--pp 0 rejected (must be >= 1)"         start --model "$M" --pp 0       --print-config
fails "--max-model-len 0 rejected"             start --model "$M" --max-model-len 0          --print-config
fails "--max-num-seqs 0 rejected"              start --model "$M" --max-num-seqs 0           --print-config
fails "--block-size 0 rejected"                start --model "$M" --block-size 0             --print-config
fails "--max-num-batched-tokens 0 rejected"    start --model "$M" --max-num-batched-tokens 0 --print-config
fails "--kv-cache-gb 0 rejected"               start --model "$M" --kv-cache-gb 0            --print-config
fails "--data-parallel-size 0 rejected"        start --model "$M" --data-parallel-size 0     --print-config
ok    "--tp 2 accepted"                         start --model "$M" --tp 2       --print-config
ok    "--pp 2 accepted"                         start --model "$M" --pp 2       --print-config
ok    "--data-parallel-size 4 accepted"         start --model "$M" --data-parallel-size 4 --print-config

# --------------------------------------------------------------------------
# Group 4: Enum parameter validation
# --------------------------------------------------------------------------
printf '\n=== Group 4: Enum parameter validation ===\n'

fails "invalid --vllm-logging-level rejected"   start --model "$M" --vllm-logging-level badlevel    --print-config
fails "invalid --performance-mode rejected"      start --model "$M" --performance-mode   badmode     --print-config
ok    "--vllm-logging-level DEBUG (uppercase) accepted"      start --model "$M" --vllm-logging-level DEBUG   --print-config
ok    "--vllm-logging-level warning accepted"                start --model "$M" --vllm-logging-level warning --print-config
ok    "--performance-mode THROUGHPUT (uppercase) accepted"   start --model "$M" --performance-mode   THROUGHPUT --print-config
ok    "--performance-mode interactivity accepted"            start --model "$M" --performance-mode   interactivity --print-config

# --------------------------------------------------------------------------
# Group 5: PD / roles validation
# --------------------------------------------------------------------------
printf '\n=== Group 5: PD / roles validation ===\n'

fails "--roles with wrong count (2 instead of 4)" \
    start --model "$M" --roles prefill,decode --print-config
fails "--roles with invalid role value" \
    start --model "$M" --roles prefill,prefill,decode,bad --print-config
fails "--num-prefill + --num-decode != 4" \
    start --model "$M" --num-prefill 3 --num-decode 2 --print-config
fails "--nnodes without PD rejected" \
    start --model "$M" --nnodes 4 --print-config
fails "--nnodes 3 rejected (must be 1 or cluster size 4)" \
    start --model "$M" --num-prefill 2 --num-decode 2 --nnodes 3 --print-config
fails "unknown kv-connector name rejected" \
    start --model "$M" --num-prefill 2 --num-decode 2 --kv-connector badconnector --print-config
ok    "valid --roles accepted" \
    start --model "$M" --roles prefill,prefill,decode,decode --print-config
ok    "--num-prefill 2 --num-decode 2 accepted" \
    start --model "$M" --num-prefill 2 --num-decode 2 --print-config
ok    "--nnodes 1 with PD accepted" \
    start --model "$M" --num-prefill 2 --num-decode 2 --nnodes 1 --print-config
ok    "case-insensitive kv-connector accepted" \
    start --model "$M" --num-prefill 2 --num-decode 2 --kv-connector nixlconnector --print-config

# --------------------------------------------------------------------------
# Group 6: --no-pd overrides PD role flags
# --------------------------------------------------------------------------
printf '\n=== Group 6: --no-pd override ===\n'

ok    "--no-pd with --roles still exits 0" \
    start --model "$M" --roles prefill,prefill,decode,decode --no-pd --print-config
has   "--no-pd output shows PD disabled" "PD enabled:               false" \
    start --model "$M" --roles prefill,prefill,decode,decode --no-pd --print-config
has   "--no-pd with --num-prefill/decode shows PD disabled" "PD enabled:               false" \
    start --model "$M" --num-prefill 2 --num-decode 2 --no-pd --print-config

# --------------------------------------------------------------------------
# Group 7: Configuration output (--print-config)
# --------------------------------------------------------------------------
printf '\n=== Group 7: Configuration output (--print-config) ===\n'

has "model path appears in config"              "$M" \
    start --model "$M" --print-config
has "alias defaults to basename of model"       "$(basename "$M")" \
    start --model "$M" --print-config
has "--alias overrides default alias"           "mymodel" \
    start --model "$M" --alias mymodel --print-config
has "non-PD shows PD disabled"                  "PD enabled:               false" \
    start --model "$M" --print-config
has "--num-prefill/decode enables PD"           "PD enabled:               true" \
    start --model "$M" --num-prefill 2 --num-decode 2 --print-config
has "--roles enables PD"                        "PD enabled:               true" \
    start --model "$M" --roles prefill,prefill,decode,decode --print-config
has "--max-model-len 8192 reflected in config"  "8192" \
    start --model "$M" --max-model-len 8192 --print-config
has "--vllm-logging-level DEBUG normalized to debug" "debug" \
    start --model "$M" --vllm-logging-level DEBUG --print-config
has "--performance-mode explicit shown in config" "throughput (explicit" \
    start --model "$M" --performance-mode throughput --print-config
has "--data-parallel-size reflected in config"  "4" \
    start --model "$M" --data-parallel-size 4 --print-config
has "NixlConnector shown in PD config"          "NixlConnector" \
    start --model "$M" --num-prefill 2 --num-decode 2 --print-config
has "--nnodes 1 mode shown in config"           "0 on every node" \
    start --model "$M" --num-prefill 2 --num-decode 2 --nnodes 1 --print-config
has "kv-connector case-insensitive canonicalization in config" "P2pNcclConnector" \
    start --model "$M" --num-prefill 2 --num-decode 2 --kv-connector p2pncclconnector --print-config

# --------------------------------------------------------------------------
# Group 8: Service file content (--print-raw)  ← core correctness tests
# --------------------------------------------------------------------------
printf '\n=== Group 8: Service file content (--print-raw) ===\n'

# vllm serve migration: key assertions
has  "ExecStart uses 'vllm serve'"                   "vllm serve" \
    start --model "$M" --print-raw
no   "ExecStart does not use 'python -m vllm'"       "python -m vllm" \
    start --model "$M" --print-raw
no   "ExecStart does not use '--model' flag"         "--model " \
    start --model "$M" --print-raw
has  "model path is positional arg after 'serve'"    "serve ${M}" \
    start --model "$M" --print-raw

# Common service directives
has  "service file includes --enable-prefix-caching"  "--enable-prefix-caching" \
    start --model "$M" --print-raw
has  "service file includes --enable-chunked-prefill" "--enable-chunked-prefill" \
    start --model "$M" --print-raw
has  "service file includes --host 0.0.0.0"           "--host 0.0.0.0" \
    start --model "$M" --print-raw
has  "service file includes --port 8000"              "--port 8000" \
    start --model "$M" --print-raw
has  "Restart=always present"                         "Restart=always" \
    start --model "$M" --print-raw

# Environment variables
has  "VLLM_TARGET_DEVICE=cpu in service"              "VLLM_TARGET_DEVICE=cpu" \
    start --model "$M" --print-raw
has  "VLLM_CPU_KVCACHE_SPACE present"                 "VLLM_CPU_KVCACHE_SPACE" \
    start --model "$M" --print-raw
has  "custom --kv-cache-gb 80 in VLLM_CPU_KVCACHE_SPACE" "VLLM_CPU_KVCACHE_SPACE=80" \
    start --model "$M" --kv-cache-gb 80 --print-raw
has  "VLLM_LOGGING_LEVEL=debug in service env"        "VLLM_LOGGING_LEVEL=debug" \
    start --model "$M" --vllm-logging-level debug --print-raw

# Model parameters
has  "default --max-model-len 4096 in service"        "--max-model-len 4096" \
    start --model "$M" --print-raw
has  "custom --max-model-len 8192 in service"         "--max-model-len 8192" \
    start --model "$M" --max-model-len 8192 --print-raw
has  "default --max-num-seqs 16 in service"           "--max-num-seqs 16" \
    start --model "$M" --print-raw
has  "default --block-size 32 in service"             "--block-size 32" \
    start --model "$M" --print-raw

# TP / PP parallelism (non-PD)
has  "--tp 2 produces --tensor-parallel-size 2"       "--tensor-parallel-size 2" \
    start --model "$M" --tp 2 --print-raw
no   "--tp 1 omits --tensor-parallel-size flag"       "--tensor-parallel-size" \
    start --model "$M" --tp 1 --print-raw
has  "--pp 2 produces --pipeline-parallel-size 2"     "--pipeline-parallel-size 2" \
    start --model "$M" --pp 2 --print-raw
no   "--pp 1 omits --pipeline-parallel-size flag"     "--pipeline-parallel-size" \
    start --model "$M" --pp 1 --print-raw
has  "--data-parallel-size 4 in non-PD service"       "--data-parallel-size 4" \
    start --model "$M" --data-parallel-size 4 --print-raw

# Performance mode
has  "default non-PD --performance-mode is balanced"  "--performance-mode balanced" \
    start --model "$M" --print-raw
has  "explicit --performance-mode throughput all nodes" "--performance-mode throughput" \
    start --model "$M" --performance-mode throughput --print-raw
has  "PD prefill default --performance-mode is throughput" "--performance-mode throughput" \
    start --model "$M" --num-prefill 2 --num-decode 2 --print-raw
has  "PD decode default --performance-mode is interactivity" "--performance-mode interactivity" \
    start --model "$M" --num-prefill 2 --num-decode 2 --print-raw

# PD service file checks
has  "PD service includes --kv-transfer-config"       "--kv-transfer-config" \
    start --model "$M" --num-prefill 2 --num-decode 2 --print-raw
has  "PD service includes kv_producer role"           "kv_producer" \
    start --model "$M" --num-prefill 2 --num-decode 2 --print-raw
has  "PD service includes kv_consumer role"           "kv_consumer" \
    start --model "$M" --num-prefill 2 --num-decode 2 --print-raw
has  "PD NixlConnector present in JSON"               "NixlConnector" \
    start --model "$M" --num-prefill 2 --num-decode 2 --print-raw
has  "PD service includes --master-addr"              "--master-addr" \
    start --model "$M" --num-prefill 2 --num-decode 2 --print-raw
has  "PD service includes --distributed-executor-backend mp" "--distributed-executor-backend mp" \
    start --model "$M" --num-prefill 2 --num-decode 2 --print-raw
has  "PD service includes --attention-config"         "--attention-config" \
    start --model "$M" --num-prefill 2 --num-decode 2 --print-raw
has  "NixlConnector: prefill peer_ips contains decode RDMA IP" "100.0.0.143" \
    start --model "$M" --num-prefill 2 --num-decode 2 --print-raw
has  "NixlConnector: decode peer_ips contains prefill RDMA IP" "100.0.0.141" \
    start --model "$M" --num-prefill 2 --num-decode 2 --print-raw
has  "P2pNcclConnector (case-insensitive) canonicalized in service" "P2pNcclConnector" \
    start --model "$M" --num-prefill 2 --num-decode 2 --kv-connector p2pncclconnector --print-raw
has  "PD non-Nixl connector omits kv_connector_extra_config" \
    '"kv_role":"kv_producer","kv_connector":"P2pNcclConnector"}' \
    start --model "$M" --num-prefill 2 --num-decode 2 --kv-connector p2pncclconnector --print-raw
has  "--nnodes 1 appears in PD ExecStart"             "--nnodes 1" \
    start --model "$M" --num-prefill 2 --num-decode 2 --nnodes 1 --print-raw
has  "--data-parallel-size 4 in PD service file"      "--data-parallel-size 4" \
    start --model "$M" --num-prefill 2 --num-decode 2 --data-parallel-size 4 --print-raw

# --------------------------------------------------------------------------
# Group 9: Subcommands (stop / restart / status)
# --------------------------------------------------------------------------
printf '\n=== Group 9: Subcommands (stop/restart/status) ===\n'

ok  "stop exits 0 with stub SSH"    stop
ok  "restart exits 0 with stub SSH" restart
ok  "status exits 0 with stubs"     status

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
printf '\n======================================\n'
printf 'Results: %d passed, %d failed\n' "$PASS" "$FAIL"
if [[ "${#ERRORS[@]}" -gt 0 ]]; then
    printf '\nFailed tests:\n'
    for e in "${ERRORS[@]}"; do printf '  - %s\n' "$e"; done
fi
printf '======================================\n'

[[ "$FAIL" -eq 0 ]]
