### Cluster model launcher usage

This repository provides `cluster_model.sh` to control a multi-node vLLM CPU cluster across four nodes (`192.168.1.141-144`) with RDMA (`100.0.0.1-4`).

#### Basic commands

- **Start or switch model**

```bash
./cluster_model.sh start --model llama-7b --alias llama
```

This assumes the model is located at `/home/labroot/models/llama-7b` on all nodes. The alias `llama` is exposed at the OpenAI-compatible gateway (`http://192.168.1.145/v1/models`).

By default, **PD (prefill/decode split) is OFF**, so all nodes run the same role without `VLLM_DIST_ROLE` or RDMA KV transfer configuration.

- **Stop, restart, status**

```bash
./cluster_model.sh stop
./cluster_model.sh restart
./cluster_model.sh status
```

#### Prefill / decode (PD) split

PD is **disabled by default** and is automatically enabled when you pass `--roles` or `--num-prefill/--num-decode`. Use `--no-pd` to force PD off even if those options are present.

- **Explicit roles per node**

```bash
./cluster_model.sh start \
  --model llama-7b \
  --alias llama-pd \
  --roles prefill,prefill,decode,decode
```

The roles are mapped positionally to nodes:

- `192.168.1.141` → `prefill`
- `192.168.1.142` → `prefill`
- `192.168.1.143` → `decode`
- `192.168.1.144` → `decode`

- **Specify counts instead of full mapping**

```bash
./cluster_model.sh start \
  --model llama-7b \
  --num-prefill 2 \
  --num-decode 2
```

This yields the same layout as above (first two nodes prefill, last two decode).

- **Disable PD split**

```bash
./cluster_model.sh start \
  --model llama-7b \
  --roles prefill,prefill,decode,decode \
  --no-pd
```

In this example, `--no-pd` **forces PD off** even though roles are specified, so all nodes run without `VLLM_DIST_ROLE` or RDMA KV transfer configuration.

#### Parallelism and batch configuration

- **Configure tensor/pipeline parallelism and batch size**

```bash
./cluster_model.sh start \
  --model llama-7b \
  --alias llama-tp2-pp2 \
  --tp 2 \
  --pp 2 \
  --max-num-batched-tokens 32 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --block-size 32
```

Adjust `--tp` and `--pp` based on your vLLM deployment topology and hardware, and tune `--max-num-batched-tokens`, `--max-model-len`, `--max-num-seqs`, and `--block-size` (defaults apply when omitted) as needed for throughput and latency.

