# SPDX-License-Identifier: Apache-2.0
"""Team-aware KV preemption.

When a request cannot get KV blocks, stock vLLM (priority policy) evicts the
lowest-priority request. A high-priority team can then grow without bound and
never pay. This scheduler splits the decision in two:

1. Who pays: the team whose KV usage most exceeds its weighted fair share,
   among teams currently running (work-conserving). A team only counts as
   over quota above 1 + hysteresis, so two teams near their share do not take
   turns being evicted.
2. Which request: that team's request with the fewest computed tokens, the
   cheapest to recompute (V1 preemption drops the KV and recomputes it).
   Requests already preempted max_preemptions times are skipped.

If no team qualifies, or requests carry no team info, fall back to stock.

Team info comes from the Gateway as `vllm_xargs={"tenant": str, "weight": num}`,
which vLLM puts in `request.sampling_params.extra_args`.

Use: --scheduling-policy priority \
     --scheduler-cls vllm.v1.core.sched.fair_scheduler.FairAsyncScheduler
"""

import os
from collections import defaultdict

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request


def team_of(req: Request) -> tuple[str, float] | None:
    """(tenant, weight) from the Gateway, or None if missing or malformed."""
    sp = req.sampling_params
    args = (sp.extra_args if sp is not None else None) or {}
    tenant = args.get("tenant")
    try:
        weight = float(args.get("weight", 1))
    except (TypeError, ValueError):
        return None
    if not isinstance(tenant, str) or weight <= 0:
        return None
    return tenant, weight


class FairPreemptionMixin:
    fair_hysteresis = float(os.environ.get("VLLM_FAIR_HYSTERESIS", "0.1"))
    fair_max_preemptions = int(os.environ.get("VLLM_FAIR_MAX_PREEMPTIONS", "2"))

    def team_overuse(self) -> dict[str, float]:
        """team -> (share of KV in use) / (share of weight), over running teams."""
        usage: dict[str, int] = defaultdict(int)
        weight: dict[str, float] = {}
        for r in self.running:
            t = team_of(r)
            if t is None:
                continue
            usage[t[0]] += r.num_computed_tokens
            weight[t[0]] = t[1]
        total_usage = sum(usage.values())
        total_weight = sum(weight.values())
        if total_usage == 0 or total_weight == 0:
            return {}
        return {
            t: (usage[t] / total_usage) / (weight[t] / total_weight) for t in usage
        }

    def _select_preemption_victim(self, request: Request) -> Request:
        if self.policy != SchedulingPolicy.PRIORITY:
            return super()._select_preemption_victim(request)
        overuse = self.team_overuse()
        over = sorted(
            (t for t, x in overuse.items() if x > 1 + self.fair_hysteresis),
            key=lambda t: overuse[t],
            reverse=True,
        )
        for tenant in over:
            candidates = [
                r
                for r in self.running
                if (t := team_of(r)) is not None
                and t[0] == tenant
                and r.num_preemptions < self.fair_max_preemptions
                and self._request_blocks_can_be_freed(r)
            ]
            if candidates:
                return min(candidates, key=lambda r: r.num_computed_tokens)
        return super()._select_preemption_victim(request)


class FairScheduler(FairPreemptionMixin, Scheduler):
    pass


class FairAsyncScheduler(FairPreemptionMixin, AsyncScheduler):
    pass
