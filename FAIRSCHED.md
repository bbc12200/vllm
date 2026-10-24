# fairsched：team-aware preemption for vLLM v0.31.0

This branch adds two scheduling policies to the V1 scheduler, for several teams
sharing one GPU through a gateway that tags each request with a team.

Base: `v0.31.0` (`db9527a468`). All changes are in `vllm/v1/core/sched/`.

| File | Change |
|---|---|
| `scheduler.py` | Extracts the inline victim choice into `_select_preemption_victim(request)` (no behavior change); adds seat preemption (`_pick_seat_victim`); NVTX ranges |
| `fair_scheduler.py` | `FairScheduler` / **`FairAsyncScheduler`**: team-aware KV preemption |
| `_nvtx.py` | NVTX helpers, no-op without CUDA |
| `tests/v1/core/test_fair_preemption.py`, `test_seat_preemption.py` | Unit tests (CPU only) |

## KV preemption: who pays, and which request

When a request cannot get KV blocks, stock priority mode evicts the worst
`(priority, arrival_time)`, regardless of how much KV each team holds.
`FairAsyncScheduler` picks in two steps:

1. **Team**: overuse = (team's share of held KV) / (team's share of weight).
   Pick the most over-used team above `1 + hysteresis`.
2. **Request**: within that team, the one with the fewest `num_computed_tokens`
   (V1 preemption drops KV and recomputes, so this is the cheapest to redo).

No over-used team → fall back to stock behavior.

Team and weight come from per-request custom args, set (and overwritten) by the gateway:

```json
{"priority": 0, "vllm_xargs": {"tenant": "engineering", "weight": 3}}
```

## Seat preemption

When seats (`max_num_seqs`) are full, a higher-priority waiting request that has
waited longer than `VLLM_SEAT_PREEMPT_WAIT_MS` (400) evicts a lower-priority running
request with the fewest computed tokens. A request is evicted at most
`VLLM_SEAT_PREEMPT_MAX_PER_REQ` (2) times.

## Usage

```
vllm serve <model> --scheduling-policy=priority \
  --scheduler-cls=vllm.v1.core.sched.fair_scheduler.FairAsyncScheduler
```

Use `FairAsyncScheduler`, not `FairScheduler`: a custom class is used as-is, and
subclassing `Scheduler` silently disables async scheduling.

## Tests

```
pytest tests/v1/core/test_scheduler.py tests/v1/core/test_fair_preemption.py -m cpu_test -q
pytest tests/v1/core/test_seat_preemption.py -q
```

## Results

Two A10G nodes, four teams (weights 3/2/1/1), multi-turn conversations that overflow
the KV cache: p95 TTFT down 27–44% for every team, throughput up 6–12%, averaged over
two paired runs. Nsight Systems: scheduler step p50 ~0.35 ms (0.7% of wall time),
GPU idle 0.2%, no GPU time spent waiting on the scheduler.
Full write-up, raw data and failure log are in the infrastructure repo.
