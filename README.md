# model_deploy

A shell-based controller for running a multi-node vLLM CPU cluster. It generates systemd service files and deploys them via SSH, supporting both homogeneous and prefill/decode (PD) split configurations.

## Cluster layout

| Role | Management IP | RDMA IP |
|---|---|---|
| Node 0 | 192.168.1.141 | 100.0.0.141 |
| Node 1 | 192.168.1.142 | 100.0.0.142 |
| Node 2 | 192.168.1.143 | 100.0.0.143 |
| Node 3 | 192.168.1.144 | 100.0.0.144 |
| Gateway | 192.168.1.145 | — |

The controller script is intended to run on the **gateway host** (192.168.1.145). It SSHes into each worker node as `labroot` to deploy and manage the vLLM systemd service.

## Prerequisites

**Controller host**
- `ssh`, `scp`, `curl` in PATH
- SSH key-based access to all worker nodes as `labroot`

**Each worker node**
- A `uv`-managed virtual environment at `/home/labroot/vllm-cpu/` with vLLM installed
- Models stored under `/home/labroot/models/`
- Intel tcmalloc (`libtcmalloc_minimal.so.4`) and Intel OMP (`libiomp5.so`) for CPU performance

## Quick start

```bash
# Start all nodes with a model (PD off, homogeneous)
./cluster_model.sh start --model llama-7b --alias llama

# Check cluster health
./cluster_model.sh status

# Stop all nodes
./cluster_model.sh stop
```

The `start` command prints a full configuration summary, deploys the systemd service to every node, then polls the gateway until the model alias appears (up to 300 s).

## Subcommands

| Subcommand | Description |
|---|---|
| `start` | Deploy a model and start vLLM on all nodes |
| `stop` | Stop the vLLM service on all nodes |
| `restart` | Restart the vLLM service on all nodes |
| `status` | Show service status, system load, and gateway model list |

## Start options

### Model

| Flag | Default | Description |
|---|---|---|
| `--model <name-or-path>` | required | Directory name under `/home/labroot/models/`, or an absolute path |
| `--alias <name>` | basename of model path | The `model_id` exposed by the OpenAI-compatible gateway |

### Prefill / decode (PD) split

PD is **disabled by default**. It activates automatically when `--roles`, `--num-prefill`, or `--num-decode` is provided (unless `--no-pd` is also set).

| Flag | Description |
|---|---|
| `--roles prefill,prefill,decode,decode` | Explicit role per node, positionally mapped to nodes 141–144 |
| `--num-prefill N` | First N nodes become prefill; remaining nodes become decode |
| `--num-decode M` | N + M must equal 4 (cluster size) |
| `--no-pd` | Force homogeneous mode even when roles/counts are provided |
| `--pd-master-addr IP` | Override `--master-addr` (default: 192.168.1.141) |
| `--kv-connector NAME` | vLLM v1 KVConnectorFactory name, case-insensitive (default: `NixlConnector`) |
| `--nnodes N` | Override vLLM `--nnodes`; must be 4 (cluster size) or 1 (every node gets `--node-rank 0`) |
| `--data-parallel-size N` | Add `--data-parallel-size` to every node (PD and non-PD) |

**Supported `--kv-connector` values:**
`NixlConnector`, `P2pNcclConnector`, `LMCacheConnectorV1`, `LMCacheMPConnector`,
`ExampleConnector`, `ExampleHiddenStatesConnector`, `MultiConnector`, `MoRIIOConnector`,
`OffloadingConnector`, `DecodeBenchConnector`, `MooncakeConnector`, `FlexKVConnectorV1`

**NixlConnector specifics:** the script automatically populates `kv_connector_extra_config` with `backends: ["UCX"]`, `side_channel_host` set to the node's RDMA IP, and `peer_ips` set to the RDMA IPs of all nodes in the **opposite** role (prefill ↔ decode). Other connectors receive only `kv_role` and `kv_connector` in the JSON.

### Parallelism (non-PD only)

| Flag | Default | Description |
|---|---|---|
| `--tp N` | 1 | Tensor parallel size; adds `--tensor-parallel-size N` when N > 1 |
| `--pp N` | 1 | Pipeline parallel size; adds `--pipeline-parallel-size N` when N > 1 |

`--tp` and `--pp` are ignored when PD is enabled.

### vLLM limits

| Flag | Default | Description |
|---|---|---|
| `--max-model-len N` | 4096 | `--max-model-len` passed to vLLM |
| `--max-num-seqs N` | 16 | `--max-num-seqs` passed to vLLM |
| `--block-size N` | 32 | `--block-size` passed to vLLM |
| `--max-num-batched-tokens N` | 16 | `--max-num-batched-tokens` passed to vLLM |
| `--kv-cache-gb N` | 40 | Sets `VLLM_CPU_KVCACHE_SPACE` (GB) on each node |

### Performance and logging

| Flag | Default | Description |
|---|---|---|
| `--performance-mode MODE` | per-role | `balanced`, `interactivity`, or `throughput`. When omitted: non-PD → `balanced`; PD prefill → `throughput`; PD decode → `interactivity` |
| `--vllm-logging-level LEVEL` | `info` | `debug`, `info`, `warning`, `error`, or `critical` (case-insensitive) |

### Dry-run flags

| Flag | Description |
|---|---|
| `--print-config` | Print the resolved configuration and exit — no SSH, no systemd changes |
| `--print-raw` / `--raw` | Print the full generated `vllm.service` unit file for every node, then exit |

Both flags can be combined: `--print-config --print-raw` prints the summary first, then the raw unit bodies.

## Examples

```bash
# Homogeneous cluster (PD off), inspect config only
./cluster_model.sh start --model llama-7b --alias llama --print-config

# Enable PD with explicit roles
./cluster_model.sh start \
  --model llama-7b \
  --alias llama-pd \
  --roles prefill,prefill,decode,decode

# Enable PD with counts
./cluster_model.sh start \
  --model llama-7b \
  --num-prefill 2 \
  --num-decode 2 \
  --max-num-batched-tokens 64

# PD with single-rank experiment (every node gets --node-rank 0)
./cluster_model.sh start \
  --model llama-7b \
  --num-prefill 2 \
  --num-decode 2 \
  --nnodes 1 \
  --data-parallel-size 4 \
  --print-config

# Non-PD with tensor and pipeline parallelism
./cluster_model.sh start \
  --model llama-7b \
  --tp 2 \
  --pp 2 \
  --max-num-batched-tokens 32 \
  --performance-mode throughput

# Use a custom kv-connector (case-insensitive)
./cluster_model.sh start \
  --model llama-7b \
  --num-prefill 2 \
  --num-decode 2 \
  --kv-connector p2pncclconnector

# Print generated systemd unit files without deploying
./cluster_model.sh start --model llama-7b --print-raw
```

## Configuration reference

Key variables at the top of `cluster_model.sh` that you may need to adjust for your environment:

| Variable | Default | Description |
|---|---|---|
| `MGMT_NODES` | 192.168.1.141–144 | Management IPs used for SSH/scp |
| `RDMA_IPS` | 100.0.0.141–144 | RDMA IPs, positionally mapped to `MGMT_NODES` |
| `GATEWAY_URL` | http://192.168.1.145/v1/models | Health check endpoint |
| `PYTHON_BIN` | /home/labroot/vllm-cpu/bin/python | Python in the uv venv on each node |
| `VLLM_BIN` | /home/labroot/vllm-cpu/bin/vllm | vLLM CLI binary in the same venv |
| `MODELS_ROOT` | /home/labroot/models | Root directory for model storage on each node |
| `KV_CONNECTOR` | NixlConnector | Default kv_connector for PD mode |
| `HEALTH_TIMEOUT` | 300 s | How long `start` polls the gateway before giving up |

## Running the tests

```bash
bash test_cluster_model.sh
```

The test suite (93 tests) runs entirely locally — no SSH or network access required. It uses `--print-config` and `--print-raw` to exercise argument validation, PD role logic, service file generation, and subcommand routing, with stub `ssh`/`scp`/`curl` commands injected via `PATH`.
