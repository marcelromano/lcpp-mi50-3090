import argparse
import asyncio
import json
import os
import secrets
import signal
import socket
import subprocess
import uuid
from pathlib import Path

import aiohttp

MODEL = Path("/tmp/test-model.gguf")
EXPECTED_SIZE = 531066496
EXPECTED_SHA256 = "cc810521ad69df7e4d3b6ffdf5cd93f4f89d4cfb8f64299a5da7e92a6aa8abe6"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--scenario", required=True, choices=(
        "overcommit", "control", "task-zero-slot-zero", "nonunified", "parent", "restore", "boundaries", "parent-refusal"))
    parser.add_argument("--guard", action="store_true")
    return parser.parse_args()


async def event_lines(response):
    while True:
        line = await response.content.readline()
        if not line:
            return
        if line.startswith(b"data: "):
            payload = line[6:].strip()
            if payload == b"[DONE]":
                return
            yield json.loads(payload)


async def first_content_event(response):
    async for event in event_lines(response):
        if event.get("content"):
            return event
    raise AssertionError("stream ended before first content event")


async def drain_stream(response):
    events = []
    async for event in event_lines(response):
        events.append(event)
    return events


async def main():
    args = parse_args()
    if MODEL.stat().st_size != EXPECTED_SIZE:
        raise AssertionError("MinerU model size changed")
    import hashlib
    digest = hashlib.sha256(MODEL.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise AssertionError("MinerU model hash changed")

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=False)
    with socket.socket() as check:
        check.bind(("127.0.0.1", args.port))
    keyfile = work / "api-key"
    keyfile.write_text(secrets.token_urlsafe(32))
    keyfile.chmod(0o600)

    parallel = 3 if args.scenario in ("parent", "parent-refusal") else 2
    engine_id = str(uuid.uuid4())
    command = [
        args.binary, "-m", str(MODEL), "-ngl", "0", "-c", "512", "-np", str(parallel),
        "-b", "128", "-ub", "128", "-t", "4", "-tb", "4",
        "--host", "127.0.0.1", "--port", str(args.port), "--api-key-file", str(keyfile),
        "--slot-operation-owner", "proxy", "--slot-operation-engine-id", engine_id,
        "--props", "--slots", "--slot-save-path", str(work),
    ]
    if args.scenario == "nonunified":
        command.append("--no-kv-unified")
    else:
        command.append("--kv-unified")
    if args.guard:
        command.append("--kv-admission")

    log_path = work / "server.log"
    log = log_path.open("w")
    proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    first = None
    try:
        timeout = aiohttp.ClientTimeout(total=150)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            headers = {"Authorization": "Bearer " + keyfile.read_text()}
            base = f"http://127.0.0.1:{args.port}"
            for _ in range(180):
                if proc.poll() is not None:
                    raise AssertionError("CPU server exited during startup")
                try:
                    async with session.get(base + "/health", headers=headers) as response:
                        if response.status == 200:
                            break
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(0.5)
            else:
                raise AssertionError("CPU health deadline")

            async with session.get(base + "/props", headers=headers) as response:
                props = await response.json()
                descriptor = props.get("kv_admission")
                if args.guard:
                    assert descriptor is not None, props
                if descriptor is not None:
                    assert descriptor["version"] == 1
                    assert descriptor["enabled"] is args.guard
                    assert descriptor["engine_id"] == engine_id
                    assert descriptor["engine_epoch"]

            if args.scenario in ("overcommit", "control"):
                overcommit = args.scenario == "overcommit"
                first_prompt, first_output = (181, 128) if overcommit else (19, 32)
                first = await session.post(base + "/completion", headers=headers, json={
                    "id_slot": 1, "prompt": [19] * first_prompt, "n_predict": first_output,
                    "temperature": 0, "seed": 42, "ignore_eos": True,
                    "cache_prompt": True, "stream": True,
                })
                assert first.status == 200
                await first_content_event(first)
                async with session.get(base + "/slots", headers=headers) as response:
                    before = await response.json()
                    active = next(slot for slot in before if slot["id"] == 1)
                    assert active["is_processing"] and active["kv_capacity"]["seq_cells"] >= first_prompt, active
                second_request = {
                    "id_slot": 0, "prompt": [18] * 411, "n_predict": 48,
                    "temperature": 0, "seed": 42, "ignore_eos": True,
                }
                if overcommit and args.guard:
                    second_request["stream"] = True
                async with session.post(base + "/completion", headers=headers, json=second_request) as response:
                    body = await response.json()
                    if overcommit and args.guard:
                        assert response.status == 503, (response.status, body)
                        assert response.headers.get("Content-Type", "").startswith("application/json"), response.headers
                        assert isinstance(body.get("error"), dict), body
                        assert body["error"]["type"] == "kv_capacity_error", body
                        assert "KV admission refused" in body["error"]["message"]
                        server_log = log_path.read_text(errors="replace")
                        assert server_log.count("KV_ADMISSION_ACCEPTED task=") == 1, server_log
                        assert "id  0 | task" not in server_log or "launch_slot" not in "\n".join(
                            line for line in server_log.splitlines() if "id  0 | task" in line), server_log
                    else:
                        assert response.status == (500 if overcommit else 200), (response.status, body)
                old_events = await drain_stream(first)
                old_stream_error = any("error" in event for event in old_events)
                if args.guard or not overcommit:
                    completed = [event for event in old_events if event.get("stop") is True]
                    assert completed and completed[-1]["timings"]["predicted_n"] == first_output, old_events
                expected_old_stream_error = overcommit and not args.guard
                assert old_stream_error == expected_old_stream_error, old_events
                if not overcommit:
                    assert body["timings"]["prompt_n"] == 411
                    assert body["timings"]["predicted_n"] == 48
                result = {"scenario": args.scenario, "guard": args.guard,
                          "second_http": 503 if overcommit and args.guard else (500 if overcommit else 200),
                          "second_typed_json_before_sse": bool(overcommit and args.guard),
                          "old_stream_error": old_stream_error, "descriptor": descriptor}

            elif args.scenario in ("boundaries", "parent-refusal"):
                assert args.guard
                before_accept = log_path.read_text().count("KV_ADMISSION_ACCEPTED task=")
                cases = ([("parent_capacity", [19] * 200, 1, 3, 503, "kv_capacity_error")]
                         if args.scenario == "parent-refusal" else
                         [("negative_output", [19] * 8, -2, 1, 400, "invalid_request_error"),
                          ("output_exceeds_context", [19] * 8, 505, 1, 503, "kv_capacity_error"),
                          ("prompt_exceeds_context", [19] * 513, 1, 1, 503, "kv_capacity_error")])
                outcomes = []
                for label, prompt, output, count, expected, error_type in cases:
                    async with session.post(base + "/completion", headers=headers, json={
                        "id_slot": 0, "prompt": prompt, "n_predict": output, "n_cmpl": count,
                        "temperature": 0, "seed": 42, "ignore_eos": True, "stream": True,
                    }) as response:
                        body = await response.json()
                        assert response.status == expected and body["error"]["type"] == error_type, (label, response.status, body)
                    assert log_path.read_text().count("KV_ADMISSION_ACCEPTED task=") == before_accept
                    async with session.get(base + "/slots", headers=headers) as response:
                        state = await response.json()
                        assert all(not slot["is_processing"] for slot in state), state
                    outcomes.append({"case": label, "http": expected, "error_type": error_type})
                async with session.post(base + "/completion", headers=headers, json={
                    "id_slot": 0, "prompt": [19] * 8, "n_predict": 2,
                    "temperature": 0, "seed": 42, "ignore_eos": True,
                }) as response:
                    body = await response.json()
                    assert response.status == 200 and body["timings"]["predicted_n"] == 2, body
                assert log_path.read_text().count("KV_ADMISSION_ACCEPTED task=") == before_accept + 1
                result = {"scenario": args.scenario, "guard": True, "outcomes": outcomes,
                          "recovery_http": 200, "descriptor": descriptor}

            elif args.scenario == "nonunified":
                first = await session.post(base + "/completion", headers=headers, json={
                    "id_slot": 0, "prompt": [19] * 24, "n_predict": 16,
                    "temperature": 0, "seed": 42, "ignore_eos": True, "stream": True,
                })
                assert first.status == 200
                await first_content_event(first)
                async with session.post(base + "/completion", headers=headers, json={
                    "id_slot": 1, "prompt": [18] * 24, "n_predict": 16,
                    "temperature": 0, "seed": 42, "ignore_eos": True,
                }) as response:
                    body = await response.json()
                    assert response.status == 200, (response.status, body)
                await drain_stream(first)
                result = {"scenario": args.scenario, "guard": args.guard, "second_http": 200,
                          "independent_streams": True, "descriptor": descriptor}

            elif args.scenario == "parent":
                async with session.post(base + "/completion", headers=headers, json={
                    "prompt": [19] * 16, "n_predict": 4, "n_cmpl": 3,
                    "temperature": 0, "seed": 42, "ignore_eos": True,
                }) as response:
                    body = await response.json()
                    assert response.status == 200, (response.status, body)
                for _ in range(20):
                    if "KV_ADMISSION_ACCEPTED task=0 count=3" in log_path.read_text(errors="replace"):
                        break
                    await asyncio.sleep(0.1)
                else:
                    raise AssertionError("parent batch was not admitted atomically before launch")
                result = {"scenario": args.scenario, "guard": args.guard,
                          "http": 200, "parent_batch_count": 3, "descriptor": descriptor}

            else:
                async with session.post(base + "/completion", headers=headers, json={
                    "id_slot": 0, "prompt": [19] * 8, "n_predict": 1,
                    "temperature": 0, "seed": 42, "ignore_eos": True,
                }) as response:
                    body = await response.json()
                    assert response.status == 200, (response.status, body)
                async with session.get(base + "/slots", headers=headers) as response:
                    slots = await response.json()
                    slot0 = next(slot for slot in slots if slot["id"] == 0)
                    assert slot0["id_task"] == 0, slot0
                async with session.get(base + "/slot-operations", headers=headers) as response:
                    cap = await response.json()
                    assert response.status == 200 and cap["engine_id"] == engine_id, cap
                seq = cap["next_seq"]
                async def operation(action, filename=None):
                    nonlocal seq
                    claims = {k: cap[k] for k in ("engine_id", "engine_epoch", "owner")}
                    claims.update(op_id=str(uuid.uuid4()), owner_seq=seq)
                    if filename is not None:
                        claims["filename"] = filename
                    async with session.post(base + "/slots/0?action=" + action, headers=headers, json=claims) as response:
                        outcome = await response.json()
                        assert response.status == 200, (response.status, outcome)
                    for key in ("engine_id", "engine_epoch", "owner", "op_id", "owner_seq"):
                        assert outcome[key] == claims[key], (key, outcome, claims)
                    assert outcome["action"] == action and outcome["slot"] == 0, outcome
                    async with session.get(base + "/slot-operations/" + claims["op_id"], headers=headers) as response:
                        assert response.status == 200 and await response.json() == outcome
                    for _ in range(2):
                        async with session.post(base + "/slot-operations/" + claims["op_id"] + "/ack",
                                                headers=headers, json=claims) as response:
                            assert response.status == 204 and await response.read() == b"", response.status
                    seq += 1
                    return outcome
                saved = await operation("save", "b1-real.bin")
                assert saved["status"] == "completed" and saved["result"]["n_saved"] > 0, saved
                snapshot = work / "b1-real.bin"
                snapshot_digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
                erased = await operation("erase")
                assert erased["status"] == "completed" and erased["result"]["n_erased"] > 0, erased
                restored = await operation("restore", "b1-real.bin")
                if args.guard:
                    assert restored["status"] == "failed" and "B1 blocks slot restore" in restored["error"], restored
                    async with session.get(base + "/slots", headers=headers) as response:
                        after = await response.json()
                        assert all(not slot["is_processing"] for slot in after), after
                        assert next(slot for slot in after if slot["id"] == 0)["n_prompt_tokens"] == 0, after
                else:
                    assert restored["status"] == "completed", restored
                    assert restored["result"]["n_restored"] == saved["result"]["n_saved"], restored
                assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == snapshot_digest
                async with session.post(base + "/completion", headers=headers, json={
                    "id_slot": 0, "prompt": [19] * 8, "n_predict": 1,
                    "temperature": 0, "seed": 42, "ignore_eos": True,
                }) as response:
                    cold = await response.json()
                    assert response.status == 200 and cold["content"] == body["content"], cold
                result = {"scenario": args.scenario, "guard": args.guard, "task_zero": 0,
                          "slot_zero": 0, "saved": saved, "erased": erased, "restored": restored,
                          "snapshot_digest": snapshot_digest, "cold_equal": True, "descriptor": descriptor}

            result["gate"] = "PASS"
            (work / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result), flush=True)
    finally:
        if first is not None:
            first.close()
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        log.close()
        keyfile.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
