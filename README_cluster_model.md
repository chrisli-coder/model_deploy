### Cluster model launcher usage

This repository provides `cluster_model.sh` to control a multi-node vLLM CPU cluster across four nodes (`192.168.1.141-144`) with RDMA IPs (`100.0.0.141-144`) on the management/RDMA mapping used by the script.

#### Basic commands

- **Start or switch model**

```bash
./cluster_model.sh start --model llama-7b --alias llama
```

This assumes the model directory exists on the **machine running the script** under `/home/labroot/models/llama-7b` (or pass an absolute path). The alias `llama` is what the OpenAI-compatible gateway (`http://192.168.1.145/v1/models`) should list once workers are healthy.

By default, **PD (prefill/decode split) is OFF**: all nodes get the same `ExecStart` (no PD-specific vLLM flags).

Every start **prints a full “即将应用” configuration block** before any SSH/systemd changes. To **only print** that block and exit (no deploy, no gateway check):

```bash
./cluster_model.sh start --model llama-7b --alias llama --print-config
```

- **Stop, restart, status**

```bash
./cluster_model.sh stop
./cluster_model.sh restart
./cluster_model.sh status
```

#### Prefill / decode (PD) split

PD is **disabled by default** and turns on when you pass `--roles` or `--num-prefill` / `--num-decode` (unless you pass `--no-pd`).

When PD is on, workers are configured with vLLM CLI (not legacy env vars): `--kv-transfer-config`, `--attention-config` (`use_prefill_decode_attention`), `--master-addr`, `--nnodes`, `--node-rank`, and `--distributed-executor-backend mp`. **`--master-port` is not passed** (vLLM default); the pre-deploy summary mentions the documented default port for operators.

- **`kv_connector`** in `--kv-transfer-config` must be a **vLLM v1 `KVConnectorFactory` registry name** (exact JSON string), for example **`NixlConnector`**, **`P2pNcclConnector`**, **`LMCacheConnectorV1`**, etc. The script header `KV_CONNECTOR` defaults to **`NixlConnector`**; override with **`--kv-connector NAME`** (matching is case-insensitive). Legacy aliases **`gloo`** / **`nixl`** are removed—they were never valid factory names in recent vLLM.
- **Default `--master-addr`** is the first management IP in `MGMT_NODES`. Override with `--pd-master-addr IP`.
- **Optional PD tuning**: `--nnodes N` overrides vLLM `--nnodes` (default: cluster node count). Allowed values are the cluster size or **`1`**; with **`1`**, every node gets **`--node-rank 0`** so you can compare behavior against multi-rank mode. **`--data-parallel-size N`** adds vLLM `--data-parallel-size` on every node when set (PD or non-PD).
- **Tensor / pipeline parallel (`--tp` / `--pp`) are not applied when PD is enabled** (PD layout uses `mp` only in this script).

**`--performance-mode`**: if you omit it, non-PD clusters use `balanced` on every node; PD uses `throughput` on prefill nodes and `interactivity` on decode nodes. If you set `--performance-mode X` explicitly, **every node** uses `X`.

**`--enable-chunked-prefill`** is **enabled by default** on all nodes (PD and non-PD).

- **Explicit roles per node**

```bash
./cluster_model.sh start \
  --model llama-7b \
  --alias llama-pd \
  --roles prefill,prefill,decode,decode
```

Mapping:

- `192.168.1.141` → `prefill`
- `192.168.1.142` → `prefill`
- `192.168.1.143` → `decode`
- `192.168.1.144` → `decode`

- **Counts instead of a full role list**

```bash
./cluster_model.sh start \
  --model llama-7b \
  --num-prefill 2 \
  --num-decode 2
```

- **Force PD off** (homogeneous nodes, no PD CLI)

```bash
./cluster_model.sh start \
  --model llama-7b \
  --roles prefill,prefill,decode,decode \
  --no-pd
```

#### Parallelism and batch configuration (non-PD)

Use `--tp` and `--pp` only when **PD is off**. Example:

```bash
./cluster_model.sh start \
  --model llama-7b \
  --alias llama-tp2-pp2 \
  --tp 2 \
  --pp 2 \
  --max-num-batched-tokens 32 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --block-size 32 \
  --performance-mode throughput
```

Tune `--max-num-batched-tokens`, `--max-model-len`, `--max-num-seqs`, `--block-size`, and `--performance-mode` (`balanced`, `interactivity`, or `throughput`) as needed.
