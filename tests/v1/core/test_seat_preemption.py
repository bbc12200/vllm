# SPDX-License-Identifier: Apache-2.0
"""Seat preemption: a high-priority waiting request evicts a low-priority
running request when all seats (max_num_seqs) are taken."""

import os
import time

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .test_scheduler import (
    create_requests_with_priority,
    create_scheduler_with_priority,
)

MODEL = os.environ.get("SEAT_TEST_MODEL", "facebook/opt-125m")


def _decode_once(scheduler, output):
    req_ids = list(output.num_scheduled_tokens)
    scheduler.update_from_output(
        output,
        ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={r: i for i, r in enumerate(req_ids)},
            sampled_token_ids=[[100] for _ in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )


def _fill_seats(scheduler):
    """Two low-priority requests take both seats: lo_big (64 tok), lo_small (16)."""
    lo_big = create_requests_with_priority(
        1, [5], [1.0], num_tokens=64, max_tokens=100, req_ids=["lo_big"]
    )[0]
    lo_small = create_requests_with_priority(
        1, [5], [2.0], num_tokens=16, max_tokens=100, req_ids=["lo_small"]
    )[0]
    scheduler.add_request(lo_big)
    scheduler.add_request(lo_small)
    _decode_once(scheduler, scheduler.schedule())
    assert len(scheduler.running) == 2
    return lo_big, lo_small


def _add_hi(scheduler, arrival_time):
    hi = create_requests_with_priority(
        1, [0], [arrival_time], num_tokens=16, max_tokens=100, req_ids=["hi"]
    )[0]
    scheduler.add_request(hi)
    return hi


def test_evicts_cheapest_low_priority_runner():
    scheduler = create_scheduler_with_priority(model=MODEL, max_num_seqs=2)
    lo_big, lo_small = _fill_seats(scheduler)
    hi = _add_hi(scheduler, arrival_time=3.0)  # waited "forever"

    output = scheduler.schedule()

    assert lo_small.status == RequestStatus.PREEMPTED  # fewer computed tokens
    assert lo_big.status == RequestStatus.RUNNING
    assert hi.status == RequestStatus.RUNNING
    assert "hi" in [r.req_id for r in output.scheduled_new_reqs]
    assert "lo_small" not in output.num_scheduled_tokens
    assert "lo_small" in output.preempted_req_ids
    assert scheduler.num_seat_preemptions == 1


def test_no_eviction_before_wait_threshold():
    scheduler = create_scheduler_with_priority(model=MODEL, max_num_seqs=2)
    _fill_seats(scheduler)
    hi = _add_hi(scheduler, arrival_time=time.time())  # just arrived

    scheduler.schedule()

    assert hi.status == RequestStatus.WAITING
    assert scheduler.num_seat_preemptions == 0


def test_already_preempted_requests_are_immune():
    scheduler = create_scheduler_with_priority(model=MODEL, max_num_seqs=2)
    lo_big, lo_small = _fill_seats(scheduler)
    lo_big.num_preemptions = 1  # K defaults to 1
    lo_small.num_preemptions = 1
    hi = _add_hi(scheduler, arrival_time=3.0)

    scheduler.schedule()

    assert hi.status == RequestStatus.WAITING
    assert lo_big.status == lo_small.status == RequestStatus.RUNNING


def test_equal_priority_never_evicts():
    scheduler = create_scheduler_with_priority(model=MODEL, max_num_seqs=2)
    _fill_seats(scheduler)
    same = create_requests_with_priority(
        1, [5], [3.0], num_tokens=16, max_tokens=100, req_ids=["same"]
    )[0]
    scheduler.add_request(same)

    scheduler.schedule()

    assert same.status == RequestStatus.WAITING
    assert scheduler.num_seat_preemptions == 0