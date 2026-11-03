import argparse
import asyncio
import json
import secrets
import signal
import socket
import subprocess
from pathlib import Path

import aiohttp

MODEL = Path("/tmp/test-model.gguf")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    parser.add_argument("--work", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=False)
    with socket.socket() as check:
        check.bind(("127.0.0.1", args.port))
    keyfile = work / "api-key"
    keyfile.write_text(secrets.token_urlsafe(32))
    keyfile.chmod(0o600)
    command = [
        args.binary, "-m", str(MODEL), "-ngl", "0", "-c", "128", "-np", "1",
        "-b", "128", "-ub", "64", "-t", "2", "-tb", "2",
        "--host", "127.0.0.1", "--port", str(args.port), "--api-key-file", str(keyfile),
        "--spec-type", "draft-simple", "--spec-draft-model", str(MODEL),
        "--spec-draft-n-max", "4", "--kv-admission", "--props",
    ]
    log_path = work / "server.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            timeout = aiohttp.ClientTimeout(total=180)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                headers = {"Authorization": "Bearer " + keyfile.read_text()}
                base = f"http://127.0.0.1:{args.port}"
                for _ in range(180):
                    if proc.poll() is not None:
                        raise AssertionError("CPU SIMPLE server exited during startup\n" + log_path.read_text(errors="replace"))
                    try:
                        async with session.get(base + "/health", headers=headers) as response:
                            if response.status == 200:
                                break
                    except aiohttp.ClientError:
                        pass
                    await asyncio.sleep(0.5)
                else:
                    raise AssertionError("CPU SIMPLE health deadline\n" + log_path.read_text(errors="replace"))

                async with session.get(base + "/props", headers=headers) as response:
                    props = await response.json()
                descriptor = props.get("kv_admission")
                assert descriptor and descriptor["enabled"] is True and descriptor["supported"] is True, descriptor
                assert descriptor["scope"] == "simple_draft_typed_resources", descriptor
                async with session.post(base + "/completion", headers=headers, json={
                    "id_slot": 0, "prompt": [19] * 32, "n_predict": 8,
                    "temperature": 0, "seed": 42, "ignore_eos": True,
                }) as response:
                    body = await response.json()
                    assert response.status == 200 and body["timings"]["predicted_n"] == 8, (response.status, body)

                # A client disconnect is not a fence or an owner release by itself. Wait for
                # the native slot release, then prove the exact slot can be admitted again.
                cancelled = await session.post(base + "/completion", headers=headers, json={
                    "id_slot": 0, "prompt": [19] * 16, "n_predict": 100,
                    "temperature": 0, "seed": 43, "ignore_eos": True, "stream": True,
                })
                assert cancelled.status == 200
                saw_event = False
                while True:
                    line = await cancelled.content.readline()
                    if not line:
                        break
                    if line.startswith(b"data: ") and line[6:].strip() not in (b"", b"[DONE]"):
                        saw_event = True
                        break
                assert saw_event, "SIMPLE cancel stream produced no event"
                cancelled.close()
                for _ in range(100):
                    async with session.get(base + "/slots", headers=headers) as response:
                        slot_rows = await response.json()
                    if not next(row for row in slot_rows if row["id"] == 0)["is_processing"]:
                        break
                    await asyncio.sleep(0.1)
                else:
                    raise AssertionError("client cancellation did not reach native slot release")
                async with session.post(base + "/completion", headers=headers, json={
                    "id_slot": 0, "prompt": [19] * 16, "n_predict": 2,
                    "temperature": 0, "seed": 44, "ignore_eos": True,
                }) as response:
                    recovery = await response.json()
                    assert response.status == 200 and recovery["timings"]["predicted_n"] == 2, (response.status, recovery)
                text = log_path.read_text(errors="replace")
                assert text.count("KV_ADMISSION_ACCEPTED task=") == 3, text
                assert "draft acceptance =" in text, text
                result = {"gate": "PASS", "scenario": "simple-two-context-control",
                          "predicted_n": body["timings"]["predicted_n"], "descriptor": descriptor,
                          "accepted_count": 3, "simple_draft_ran": True, "cancel_released_native_slot": True,
                          "post_cancel_recovery_n": recovery["timings"]["predicted_n"]}
                (work / "result.json").write_text(json.dumps(result, indent=2) + "\n")
                print(json.dumps(result), flush=True)
        finally:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            keyfile.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
