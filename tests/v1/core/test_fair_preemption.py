# SPDX-License-Identifier: Apache-2.0
"""Team-aware KV preemption (vllm.v1.core.sched.fair_scheduler)."""

import os

import pytest

from vllm.v1.core.sched.fair_scheduler import FairScheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from . import test_scheduler
from .test_scheduler import create_requests_with_priority

pytestmark = pytest.mark.cpu_test

MODEL = os.environ.get("SEAT_TEST_MODEL", "facebook/opt-125m")


def fair_scheduler(**kw):
    # Reuse the stock helper, but have it build a FairScheduler.
    orig = test_scheduler.Scheduler
    test_scheduler.Scheduler = FairScheduler
    try:
        s = test_scheduler.create_scheduler_with_priority(model=MODEL, **kw)
    finally:
        test_scheduler.Scheduler = orig
    assert isinstance(s, FairScheduler)
    return s


def req(rid, tenant=None, weight=1, priority=0, computed=0, arrival=0.0,
        preemptions=0, num_tokens=16):
    r = create_requests_with_priority(
        1, [priority], [arrival], num_tokens=num_tokens, req_ids=[rid]
    )[0]
    if tenant is not None:
        r.sampling_params.extra_args = {"tenant": tenant, "weight": weight}
    r.num_computed_tokens = computed
    r.num_preemptions = preemptions
    return r


def pick(scheduler, running):
    scheduler.running = running
    return scheduler._select_preemption_victim(running[-1]).request_id


# 1. The over-quota team pays, not the lowest-priority one.
def test_over_quota_team_pays():
    s = fair_scheduler()
    # weights 3:1 -> fair share 75% : 25%.
    # engineering 9000 / 10000 = 90% -> 1.20x; hr 1000 -> 0.40x.
    running = [
        req("eng", "engineering", 3, priority=0, computed=9000, arrival=1),
        req("hr", "hr", 1, priority=2, computed=1000, arrival=2),
    ]
    assert pick(s, running) == "eng"


# 2. Within the paying team, evict the request cheapest to recompute.
def test_cheapest_request_in_team():
    s = fair_scheduler()
    running = [
        req("eng-big", "engineering", 1, computed=9000, arrival=1),
        req("eng-mid", "engineering", 1, computed=5000, arrival=2),
        req("eng-new", "engineering", 1, computed=300, arrival=3),
        req("hr", "hr", 1, priority=2, computed=1000, arrival=4),
    ]
    assert pick(s, running) == "eng-new"


# 3. Near fair share (below 1 + hysteresis): fall back to stock priority.
def test_hysteresis_falls_back_to_priority():
    s = fair_scheduler()
    # weights 1:1, usage 5200 : 4800 -> 1.04x and 0.96x, both below 1.1x.
    running = [
        req("eng", "engineering", 1, priority=0, computed=5200, arrival=1),
        req("hr", "hr", 1, priority=2, computed=4800, arrival=2),
    ]
    assert pick(s, running) == "hr"  # stock: highest priority number


# 4. A team whose requests hit the preemption cap is skipped.
def test_preemption_cap_skips_team():
    s = fair_scheduler()
    running = [
        req("eng", "engineering", 1, priority=0, computed=8000, preemptions=2),
        req("staff", "staff", 1, priority=1, computed=1500, arrival=1),
        req("hr", "hr", 1, priority=2, computed=500, arrival=2),
    ]
    # engineering 2.4x but capped; staff 0.45x; nobody else over -> stock.
    assert pick(s, running) == "hr"


def test_preemption_cap_moves_to_next_over_quota_team():
    s = fair_scheduler()
    # weights 1:1:2 -> shares 25 / 25 / 50 %.
    running = [
        req("eng", "engineering", 1, priority=0, computed=5000, preemptions=2),
        req("staff", "staff", 1, priority=1, computed=4000, arrival=1),
        req("hr", "hr", 2, priority=2, computed=1000, arrival=2),
    ]
    # engineering 2.0x (capped), staff 1.6x -> staff pays.
    assert pick(s, running) == "staff"


# 5. No team info (bypassed the Gateway): stock behaviour, no crash.
def test_no_tenant_falls_back():
    s = fair_scheduler()
    running = [
        req("a", priority=0, computed=9000, arrival=1),
        req("b", priority=2, computed=100, arrival=2),
    ]
    assert pick(s, running) == "b"


def test_malformed_weight_ignored():
    s = fair_scheduler()
    a = req("a", "engineering", priority=0, computed=9000, arrival=1)
    a.sampling_params.extra_args = {"tenant": "engineering", "weight": "x"}
    b = req("b", priority=2, computed=100, arrival=2)
    assert pick(s, [a, b]) == "b"


# 6. #59907: the victim was already scheduled earlier in this step.
def _decode(s, out):
    ids = list(out.num_scheduled_tokens)
    s.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={r: i for i, r in enumerate(ids)},
            sampled_token_ids=[[100] for _ in ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def test_already_scheduled_victim_is_unscheduled():
    """block_size 16, 7 blocks (6 usable).

    Step 1: eng1 32 tok (2 blocks), eng2 32 (2), hr1 16 (1) -> 1 free.
    Step 2: all three need one more block. eng1 takes the last one; eng2 must
            preempt. Weights 1:1, usage 64 : 16 -> engineering 1.6x pays,
            and its cheapest request is eng1, already scheduled this step.
            Stock priority would have evicted hr1 instead.
    """
    s = fair_scheduler(num_blocks=7, block_size=16, max_num_seqs=8)
    for rid, tenant, prio, n, t in [
        ("eng1", "engineering", 0, 32, 1.0),
        ("eng2", "engineering", 0, 32, 2.0),
        ("hr1", "hr", 2, 16, 3.0),
    ]:
        s.add_request(req(rid, tenant, 1, priority=prio, arrival=t, num_tokens=n))

    out = s.schedule()
    assert set(out.num_scheduled_tokens) == {"eng1", "eng2", "hr1"}
    _decode(s, out)

    out = s.schedule()
    by_id = s.requests
    assert by_id["eng1"].status == RequestStatus.PREEMPTED
    assert by_id["eng2"].status == RequestStatus.RUNNING
    assert by_id["hr1"].status == RequestStatus.RUNNING
    assert "eng1" not in out.num_scheduled_tokens
    assert set(out.num_scheduled_tokens) == {"eng2", "hr1"}
    assert "eng1" in out.preempted_req_ids
    assert out.total_num_scheduled_tokens == sum(out.num_scheduled_tokens.values())
