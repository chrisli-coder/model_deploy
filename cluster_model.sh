#!/bin/bash

# ==========================================================
# Multi-node vLLM cluster controller (CPU, RDMA, PD split)
# Node controller host: 192.168.1.145
# Target nodes: 192.168.1.141-144 (management) with 100.0.0.1-4 (RDMA)
#
# All worker nodes must use a uv-managed virtual environment (not conda).
# Set PYTHON_BIN (and IOMP5_PATH under the same venv) to match `uv venv` layout on each node.
# ==========================================================

set -euo pipefail

# ------------------------------
# 1. Global configuration
# ------------------------------

# Management network IPs used for SSH and scp
MGMT_NODES=("192.168.1.141" "192.168.1.142" "192.168.1.143" "192.168.1.144")

# RDMA network IPs mapped positionally to MGMT_NODES
RDMA_IPS=("100.0.0.141" "100.0.0.142" "100.0.0.143" "100.0.0.144")

# Gateway URL for health checks (OpenAI-compatible endpoint)
GATEWAY_URL="http://192.168.1.145/v1/models"

# Python: uv virtualenv on each node (not conda). Sync deps with e.g. `uv sync` in that environment.
PYTHON_BIN="/home/labroot/vllm-cpu/bin/python"
MODELS_ROOT="/home/labroot/models"

# CPU tuning defaults
KV_CACHE_GB=40
MAX_LEN=4096
BLOCK_SIZE=32
MAX_SEQS=16
DEFAULT_MAX_BATCH_SIZE=16

OMP_NUM_THREADS=8
MKL_NUM_THREADS=8

TCMALLOC_PATH="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
# Intel OMP inside the same uv venv as PYTHON_BIN (if present).
IOMP5_PATH="/home/labroot/vllm-cpu/lib/libiomp5.so"

# Parallelism defaults
DEFAULT_TP_SIZE=1
DEFAULT_PP_SIZE=1

# PD / RDMA defaults
# By default PD is disabled unless roles / num-prefill / num-decode are provided.
PD_ENABLED=false
NIXL_INTERFACE="rxe0"
KV_PORT=20000

# Health check defaults (seconds)
HEALTH_TIMEOUT=300
HEALTH_INTERVAL=5


# ------------------------------
# 2. Helper: usage
# ------------------------------

print_usage() {
    cat <<EOF
Usage:
  $0 start   --model <name-or-path> [options]
  $0 stop
  $0 restart
  $0 status

Environment (required):
  vLLM on every worker node must run under a uv-managed virtual environment, not conda.
  Configure PYTHON_BIN (and IOMP5_PATH if used) in this script to the Python and libs from
  that uv venv so systemd ExecStart matches your cluster layout.

Subcommands:
  start      Start or switch the cluster to a given model.
  stop       Stop vLLM service on all nodes.
  restart    Restart vLLM service on all nodes.
  status     Show vLLM service status and basic load on all nodes.

Start options:
  --model <name-or-path>   Model directory name under ${MODELS_ROOT}
                           or absolute/relative path to the model.
  --alias <served-name>    Web alias for the model (default: basename of model path).

  # Prefill / decode (PD) split
  # By default PD is OFF. 
  # when --roles or --num-prefill/--num-decode are provided.
  --roles prefill,prefill,decode,decode
                           Explicit role per node (positionally mapped to nodes
                           192.168.1.141-144). Each value must be "prefill" or "decode".
  --num-prefill N          Number of prefill nodes starting from the first node.
  --num-decode M           Number of decode nodes after prefill nodes.
                           (N + M must equal number of nodes). Mutually exclusive with --roles.
  --no-pd                  Force disable PD split even if roles/counts are given;
                           all nodes run without VLLM_DIST_ROLE and RDMA KV config.

  # Parallelism
  --tp N                   Tensor parallel size (default: ${DEFAULT_TP_SIZE}).
  --pp M                   Pipeline parallel size (default: ${DEFAULT_PP_SIZE}).

  # vLLM limits (defaults from script header if omitted)
  --max-model-len N        Maps to vLLM --max-model-len (default: ${MAX_LEN}).
  --max-num-seqs N         Maps to vLLM --max-num-seqs (default: ${MAX_SEQS}).
  --block-size N           Maps to vLLM --block-size (default: ${BLOCK_SIZE}).
  --max-num-batched-tokens N
                           Maps to vLLM --max-num-batched-tokens (default: ${DEFAULT_MAX_BATCH_SIZE}).
  --kv-cache-gb N          CPU KV cache size in GB for VLLM_CPU_KVCACHE_SPACE (default: ${KV_CACHE_GB}).

Examples:
  # Default: PD OFF, all nodes homogeneous
  $0 start --model llama-7b --alias llama

  # Enable PD with explicit roles
  $0 start --model llama-7b --alias llama-pd \\
      --roles prefill,prefill,decode,decode --tp 2 --pp 2 --max-num-batched-tokens 32

  # Enable PD with counts (2 prefill, 2 decode)
  $0 start --model llama-7b --num-prefill 2 --num-decode 2 --tp 4 --max-num-batched-tokens 64

EOF
}


# ------------------------------
# 3. Helpers: small utilities
# ------------------------------

error_exit() {
    echo "ERROR: $*" >&2
    exit 1
}

require_command() {
    local cmd="$1"
    command -v "$cmd" >/dev/null 2>&1 || error_exit "Required command '$cmd' not found in PATH."
}

resolve_model_path() {
    local input="$1"

    if [[ -z "$input" ]]; then
        error_exit "Model path/name must not be empty."
    fi

    # If the path is absolute or relative and exists, use it directly
    if [[ -d "$input" ]]; then
        echo "$(cd "$input" && pwd)"
        return 0
    fi

    # Otherwise, treat it as a directory name under MODELS_ROOT
    local candidate="${MODELS_ROOT}/${input}"
    if [[ -d "$candidate" ]]; then
        echo "$candidate"
        return 0
    fi

    error_exit "Model directory not found: '$input' or '${candidate}'"
}


# ------------------------------
# 4. Service file generation
# ------------------------------

generate_service_file() {
    local role="$1"         # "prefill" | "decode" | "none"
    local rdma_ip="$2"      # 100.0.0.x, may be empty if PD disabled
    local model_path="$3"
    local model_alias="$4"
    local tp_size="$5"
    local pp_size="$6"
    local max_batch="$7"
    local max_len="$8"
    local max_seqs="$9"
    local block_size="${10}"
    local kv_cache_gb="${11}"

    # Build optional PD-specific environment and ExecStart flags
    local pd_env=""
    if [[ "$PD_ENABLED" == "true" && "$role" != "none" ]]; then
        pd_env+="Environment=VLLM_DIST_ROLE=${role}"$'\n'
        pd_env+="Environment=VLLM_KV_TRANSFER_PROTOCOL=nixl"$'\n'
        pd_env+="Environment=VLLM_NIXL_INTERFACE=${NIXL_INTERFACE}"$'\n'
        pd_env+="Environment=VLLM_KV_CONNECTOR_CFG={\"port\": ${KV_PORT}}"$'\n'
    fi

    # Optional TP/PP flags
    local tp_flag=""
    local pp_flag=""
    if [[ "$tp_size" -gt 1 ]]; then
        tp_flag="--tensor-parallel-size ${tp_size}"
    fi
    if [[ "$pp_size" -gt 1 ]]; then
        pp_flag="--pipeline-parallel-size ${pp_size}"
    fi

    # Write the temporary service file
    cat <<EOF > vllm.service.tmp
[Unit]
Description=vLLM CPU Cluster Node (role=${role})
After=network-online.target

[Service]
User=labroot
WorkingDirectory=/home/labroot

Environment=VLLM_TARGET_DEVICE=cpu
Environment=VLLM_CPU_KVCACHE_SPACE=${kv_cache_gb}
Environment=VLLM_CPU_OMP_THREADS_BIND=auto
Environment="OMP_NUM_THREADS=${OMP_NUM_THREADS}"
Environment="MKL_NUM_THREADS=${MKL_NUM_THREADS}"
Environment=LD_PRELOAD=${TCMALLOC_PATH}:${IOMP5_PATH}
${pd_env}Environment=VLLM_LOGGING_LEVEL=info

ExecStart=${PYTHON_BIN} -m vllm.entrypoints.openai.api_server \\
  --model ${model_path} \\
  --host 0.0.0.0 \\
  --port 8000 \\
  --served-model-name ${model_alias} \\
  --max-model-len ${max_len} \\
  --max-num-seqs ${max_seqs} \\
  --block-size ${block_size} \\
  --max-num-batched-tokens ${max_batch} \\
  --enable-prefix-caching \\
  ${tp_flag} \\
  ${pp_flag}

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
}


# ------------------------------
# 5. Cluster operations
# ------------------------------

do_start() {
    local model_input=""
    local alias=""
    local roles_arg=""
    local num_prefill=""
    local num_decode=""
    local tp_size="${DEFAULT_TP_SIZE}"
    local pp_size="${DEFAULT_PP_SIZE}"
    local max_batch="${DEFAULT_MAX_BATCH_SIZE}"
    local max_len="${MAX_LEN}"
    local max_seqs="${MAX_SEQS}"
    local block_size="${BLOCK_SIZE}"
    local kv_cache_gb="${KV_CACHE_GB}"
    local pd_flag="${PD_ENABLED}"
    local no_pd_set="false"

    # Parse options for "start"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)
                shift
                model_input="${1:-}"
                ;;
            --alias)
                shift
                alias="${1:-}"
                ;;
            --roles)
                shift
                roles_arg="${1:-}"
                ;;
            --num-prefill)
                shift
                num_prefill="${1:-}"
                ;;
            --num-decode)
                shift
                num_decode="${1:-}"
                ;;
            --no-pd)
                pd_flag="false"
                no_pd_set="true"
                ;;
            --tp)
                shift
                tp_size="${1:-}"
                ;;
            --pp)
                shift
                pp_size="${1:-}"
                ;;
            --max-num-batched-tokens)
                shift
                max_batch="${1:-}"
                ;;
            --max-model-len)
                shift
                max_len="${1:-}"
                ;;
            --max-num-seqs)
                shift
                max_seqs="${1:-}"
                ;;
            --block-size)
                shift
                block_size="${1:-}"
                ;;
            --kv-cache-gb)
                shift
                kv_cache_gb="${1:-}"
                ;;
            *)
                error_exit "Unknown option for start: $1"
                ;;
        esac
        shift || true
    done

    [[ -z "$model_input" ]] && error_exit "--model is required for start."

    # Validate numeric values
    [[ "$tp_size" =~ ^[0-9]+$ ]] || error_exit "--tp must be a positive integer."
    [[ "$pp_size" =~ ^[0-9]+$ ]] || error_exit "--pp must be a positive integer."
    [[ "$max_batch" =~ ^[0-9]+$ ]] || error_exit "--max-num-batched-tokens must be a positive integer."
    [[ "$max_len" =~ ^[0-9]+$ ]] || error_exit "--max-model-len must be a positive integer."
    [[ "$max_seqs" =~ ^[0-9]+$ ]] || error_exit "--max-num-seqs must be a positive integer."
    [[ "$block_size" =~ ^[0-9]+$ ]] || error_exit "--block-size must be a positive integer."
    [[ "$kv_cache_gb" =~ ^[0-9]+$ ]] || error_exit "--kv-cache-gb must be a positive integer."

    [[ "$tp_size" -ge 1 ]] || error_exit "--tp must be >= 1."
    [[ "$pp_size" -ge 1 ]] || error_exit "--pp must be >= 1."
    [[ "$max_len" -ge 1 ]] || error_exit "--max-model-len must be >= 1."
    [[ "$max_seqs" -ge 1 ]] || error_exit "--max-num-seqs must be >= 1."
    [[ "$block_size" -ge 1 ]] || error_exit "--block-size must be >= 1."
    [[ "$max_batch" -ge 1 ]] || error_exit "--max-num-batched-tokens must be >= 1."
    [[ "$kv_cache_gb" -ge 1 ]] || error_exit "--kv-cache-gb must be >= 1."

    # Decide PD enablement:
    # - If --no-pd was given, PD stays disabled even if roles/counts are provided.
    # - If any PD-related options are provided and --no-pd was NOT given,
    #   enable PD automatically.
    if [[ "$no_pd_set" == "false" && "$pd_flag" == "false" ]]; then
        if [[ -n "$roles_arg" || -n "$num_prefill" || -n "$num_decode" ]]; then
            pd_flag="true"
        fi
    fi

    # Propagate to global flag used in service generation
    PD_ENABLED="$pd_flag"

    # Resolve model path and alias
    local model_path
    model_path="$(resolve_model_path "$model_input")"
    if [[ -z "$alias" ]]; then
        alias="$(basename "$model_path")"
    fi

    echo "Starting cluster with:"
    echo "  Python (uv, not conda): ${PYTHON_BIN}"
    echo "  Model path : ${model_path}"
    echo "  Alias      : ${alias}"
    echo "  TP / PP    : ${tp_size} / ${pp_size}"
    echo "  PD enabled : ${pd_flag}"
    echo "  Max batched tokens : ${max_batch}"
    echo "  Max model len      : ${max_len}"
    echo "  Max num seqs       : ${max_seqs}"
    echo "  Block size         : ${block_size}"
    echo "  KV cache GB        : ${kv_cache_gb}"

    # Determine roles per node
    local roles=()
    local num_nodes="${#MGMT_NODES[@]}"

    if [[ "$pd_flag" == "false" ]]; then
        for ((i = 0; i < num_nodes; i++)); do
            roles+=("none")
        done
    else
        if [[ -n "$roles_arg" ]]; then
            IFS=',' read -r -a roles <<<"$roles_arg"
            if [[ "${#roles[@]}" -ne "$num_nodes" ]]; then
                error_exit "--roles must specify exactly ${num_nodes} entries."
            fi
            for r in "${roles[@]}"; do
                if [[ "$r" != "prefill" && "$r" != "decode" ]]; then
                    error_exit "Invalid role '$r' in --roles. Only 'prefill' or 'decode' allowed."
                fi
            done
        else
            # Use num-prefill / num-decode if provided
            if [[ -n "$num_prefill" || -n "$num_decode" ]]; then
                [[ "$num_prefill" =~ ^[0-9]+$ ]] || error_exit "--num-prefill must be a non-negative integer."
                [[ "$num_decode" =~ ^[0-9]+$ ]] || error_exit "--num-decode must be a non-negative integer."
                if [[ $((num_prefill + num_decode)) -ne "$num_nodes" ]]; then
                    error_exit "--num-prefill + --num-decode must equal ${num_nodes}."
                fi
                for ((i = 0; i < num_nodes; i++)); do
                    if [[ "$i" -lt "$num_prefill" ]]; then
                        roles+=("prefill")
                    else
                        roles+=("decode")
                    fi
                done
            else
                # Default PD layout: first half prefill, second half decode
                local half=$((num_nodes / 2))
                for ((i = 0; i < num_nodes; i++)); do
                    if [[ "$i" -lt "$half" ]]; then
                        roles+=("prefill")
                    else
                        roles+=("decode")
                    fi
                done
            fi
        fi
    fi

    echo "Node roles:"
    for i in "${!MGMT_NODES[@]}"; do
        echo "  ${MGMT_NODES[$i]} (RDMA ${RDMA_IPS[$i]}): role=${roles[$i]}"
    done

    # Deploy service to each node
    for i in "${!MGMT_NODES[@]}"; do
        local mgmt_ip="${MGMT_NODES[$i]}"
        local rdma_ip="${RDMA_IPS[$i]}"
        local role="${roles[$i]}"

        echo "Configuring node ${mgmt_ip} (RDMA ${rdma_ip}, role=${role})..."

        generate_service_file "$role" "$rdma_ip" "$model_path" "$alias" \
            "$tp_size" "$pp_size" "$max_batch" "$max_len" "$max_seqs" "$block_size" "$kv_cache_gb"

        scp vllm.service.tmp "labroot@${mgmt_ip}:/tmp/vllm.service" >/dev/null 2>&1 || \
            error_exit "Failed to copy service file to ${mgmt_ip}."

        ssh "labroot@${mgmt_ip}" "sudo mv /tmp/vllm.service /etc/systemd/system/vllm.service && sudo systemctl daemon-reload && sudo systemctl enable vllm && sudo systemctl restart vllm" >/dev/null 2>&1 || \
            error_exit "Failed to enable/restart vllm service on ${mgmt_ip}."
    done

    rm -f vllm.service.tmp

    # Health check via gateway
    echo "Waiting for model '${alias}' to appear at gateway ${GATEWAY_URL}..."
    local elapsed=0
    while [[ "$elapsed" -lt "$HEALTH_TIMEOUT" ]]; do
        if curl -s "${GATEWAY_URL}" | grep -qEi "\"id\"\s*:\s*\"${alias}\""; then
            echo "SUCCESS: Cluster is online and serving alias '${alias}'."
            echo ""
            echo "========== Effective configuration (all nodes) =========="
            echo "Python (uv):                       ${PYTHON_BIN}"
            echo "Model path (--model):               ${model_path}"
            echo "Served name (--served-model-name):  ${alias}"
            echo "Listen:                             host=0.0.0.0 port=8000"
            echo "Tensor parallel size:              ${tp_size}"
            echo "Pipeline parallel size:            ${pp_size}"
            if [[ "$tp_size" -gt 1 ]]; then
                echo "ExecStart --tensor-parallel-size: ${tp_size}"
            else
                echo "ExecStart --tensor-parallel-size:  (omitted, tp=1)"
            fi
            if [[ "$pp_size" -gt 1 ]]; then
                echo "ExecStart --pipeline-parallel-size: ${pp_size}"
            else
                echo "ExecStart --pipeline-parallel-size:  (omitted, pp=1)"
            fi
            echo "Prefill/decode (PD) enabled:       ${pd_flag}"
            if [[ "$pd_flag" == "true" ]]; then
                echo "NIXL interface:                    ${NIXL_INTERFACE}"
                echo "KV connector port:                 ${KV_PORT}"
            fi
            echo "VLLM_TARGET_DEVICE:                cpu"
            echo "VLLM_CPU_OMP_THREADS_BIND:        auto"
            echo "VLLM_LOGGING_LEVEL:                info"
            echo "Max model len:                      ${max_len}"
            echo "Max num seqs:                       ${max_seqs}"
            echo "Block size:                         ${block_size}"
            echo "Max num batched tokens:            ${max_batch}"
            echo "VLLM_CPU_KVCACHE_SPACE (GB):       ${kv_cache_gb}"
            echo "OMP_NUM_THREADS:                   ${OMP_NUM_THREADS}"
            echo "MKL_NUM_THREADS:                   ${MKL_NUM_THREADS}"
            echo "LD_PRELOAD (tcmalloc):             ${TCMALLOC_PATH}"
            echo "LD_PRELOAD (iomp5):                ${IOMP5_PATH}"
            echo "Prefix caching:                     enabled (--enable-prefix-caching)"
            echo "Gateway (health check):            ${GATEWAY_URL}"
            echo ""
            echo "Per-node role (VLLM_DIST_ROLE when PD on):"
            for i in "${!MGMT_NODES[@]}"; do
                echo "  ${MGMT_NODES[$i]}  RDMA ${RDMA_IPS[$i]}  role=${roles[$i]}"
            done
            echo "========================================================="
            return 0
        fi
        echo "  Not ready yet... (${elapsed}/${HEALTH_TIMEOUT}s)"
        sleep "$HEALTH_INTERVAL"
        elapsed=$((elapsed + HEALTH_INTERVAL))
    done

    echo "WARNING: Timed out waiting for alias '${alias}' to appear at gateway."
    echo "Please check logs on cluster nodes for details."
}


do_stop() {
    echo "Stopping vLLM service on all nodes..."
    for ip in "${MGMT_NODES[@]}"; do
        if ssh "labroot@${ip}" "sudo systemctl stop vllm" >/dev/null 2>&1; then
            echo "  Node ${ip} stopped."
        else
            echo "  Node ${ip}: failed to stop (maybe already stopped?)."
        fi
    done
}


do_restart() {
    echo "Restarting vLLM service on all nodes..."
    for ip in "${MGMT_NODES[@]}"; do
        if ssh "labroot@${ip}" "sudo systemctl restart vllm" >/dev/null 2>&1; then
            echo "  Node ${ip} restarted."
        else
            echo "  Node ${ip}: failed to restart."
        fi
    done
}


do_status() {
    echo -e "IP Address\t\tStatus\t\tLoad"
    echo "--------------------------------------------------------"
    for ip in "${MGMT_NODES[@]}"; do
        local st
        local load
        st=$(ssh "labroot@${ip}" "systemctl is-active vllm" 2>/dev/null || echo "unknown")
        load=$(ssh "labroot@${ip}" "uptime | awk -F'load average:' '{ print \$2 }'" 2>/dev/null || echo "n/a")
        echo -e "${ip}\t${st}\t${load}"
    done

    echo
    echo "Gateway models at ${GATEWAY_URL}:"
    if command -v curl >/dev/null 2>&1; then
        curl -s "${GATEWAY_URL}" | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin), indent=2))" 2>/dev/null || echo "  (failed to query gateway)"
        echo
    else
        echo "  curl not available; skipping gateway query."
    fi
}


# ------------------------------
# 6. Main entrypoint
# ------------------------------

main() {
    require_command ssh
    require_command scp
    require_command curl

    if [[ $# -lt 1 ]]; then
        print_usage
        exit 1
    fi

    local cmd
    cmd=$(echo "$1" | tr '[:upper:]' '[:lower:]')
    shift || true

    echo "Runtime: vLLM workers use a uv virtualenv (not conda). PYTHON_BIN=${PYTHON_BIN}"

    case "$cmd" in
        start|switch)
            do_start "$@"
            ;;
        stop)
            do_stop
            ;;
        restart)
            do_restart
            ;;
        status)
            do_status
            ;;
        *)
            print_usage
            exit 1
            ;;
    esac
}

main "$@"

