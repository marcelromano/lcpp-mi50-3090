#!/usr/bin/env python3
"""B3-R1-S3 end-to-end HTTP driver (CPU, synthetic hybrid gguf). stdlib only.

usage: b3_http_driver.py --binary BIN --model GGUF --work DIR --port N --scenario S
scenarios:
  on        admission + restore ON: save/erase/restore, D1 replacement credit, per-key ledger
  off       admission OFF: legacy path works, planner untouched (0 reservations)
  b1        --kv-admission without --kv-admission-restore: restore blocked (unchanged B1 behavior)
  fixture   build with LLAMA_B3_LIMIT_FIXTURE: forced limit_rec=1 accepted, 0 refused; attention 0 refused
"""
import argparse, json, os, signal, socket, subprocess, sys, time, urllib.request, urllib.error, uuid
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--binary", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--scenario", required=True, choices=("on", "off", "b1", "fixture", "fullpool"))
args = ap.parse_args()

work = Path(args.work)
work.mkdir(parents=True, exist_ok=True)
base = f"http://127.0.0.1:{args.port}"
engine_id = str(uuid.uuid4())
results = []
failures = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name + ((" :: " + str(detail)) if (detail and not cond) else ""), flush=True)
    if not cond:
        failures.append(name)


def http(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, raw


cmd = [args.binary, "-m", args.model, "-ngl", "0", "-c", "512", "-np", "2", "-b", "64", "-ub", "64", "-t", "2", "-tb", "2",
       "--host", "127.0.0.1", "--port", str(args.port), "--slot-operation-owner", "b3s3",
       "--slot-operation-engine-id", engine_id, "--props", "--slots", "--slot-save-path", str(work) + "/",
       "--no-warmup", "--cache-ram", "0"]
if args.scenario in ("on", "fixture", "fullpool"):
    cmd += ["--kv-admission", "--kv-admission-restore"]
elif args.scenario == "b1":
    cmd += ["--kv-admission"]
log_path = work / f"server-{args.scenario}.log"
logf = open(log_path, "w")
proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)

try:
    for _ in range(240):
        if proc.poll() is not None:
            print("server exited during startup\n" + log_path.read_text(errors="replace")[-3000:])
            sys.exit(2)
        try:
            if http("GET", "/health")[0] == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        print("health deadline")
        sys.exit(2)

    st, props = http("GET", "/props")
    desc = props.get("kv_admission") if isinstance(props, dict) else None
    print("props.kv_admission =", json.dumps(desc), flush=True)

    Q = [1] + [(i * 37) % 300 + 3 for i in range(20)]

    def complete(slot, prompt, n):
        st, b = http("POST", "/completion", {"id_slot": slot, "prompt": prompt, "n_predict": n, "temperature": 0, "seed": 7,
                                              "ignore_eos": True, "return_tokens": True, "cache_prompt": True,
                                              "logit_bias": [[i, False] for i in range(0, 259)]})
        return st, b

    st, cap = http("GET", "/slot-operations")
    check("slot-operations capability", st == 200, cap)
    seq = [cap["next_seq"]]

    def op(action, slot, filename=None):
        claims = {k: cap[k] for k in ("engine_id", "engine_epoch", "owner")}
        claims.update(op_id=str(uuid.uuid4()), owner_seq=seq[0])
        if filename is not None:
            claims["filename"] = filename
        st, out = http("POST", f"/slots/{slot}?action={action}", claims)
        if st == 200 and isinstance(out, dict):
            http("POST", f"/slot-operations/{claims['op_id']}/ack", claims)
            seq[0] += 1
        return st, out

    def metrics():
        return http("GET", "/slot-operations")[1].get("b3_restore", {})

    def slot_ntok(slot):
        rows = http("GET", "/slots")[1]
        r = next(r for r in rows if r["id"] == slot)
        return r.get("next_token", [{}])[0].get("n_decoded", None), r

    if args.scenario == "fullpool":
        # no restore involved: does the (pre-existing) completion admission refuse a new prompt on an occupied slot
        # when both recurrent cells of the pool are in use?
        st0, a = complete(0, Q, 8)
        st1, b = complete(1, Q, 8)
        st2, c = complete(1, Q + [5, 6, 7], 8)
        st3, d = complete(0, [1] + [9, 8, 7, 6], 8)
        print("fullpool statuses (slot0 Q, slot1 Q, slot1 Q+3, slot0 other):", st0, st1, st2, st3, flush=True)
        print("fullpool metrics", json.dumps(metrics()), flush=True)
        sys.exit(0)

    # 1. generate on slot 0 (real hybrid forward), save
    st, c1 = complete(0, Q, 24)
    check("completion slot0", st == 200 and len(c1.get("tokens", [])) == 24, c1 if st != 200 else "")
    gen1 = c1.get("tokens", [])
    st, saved = op("save", 0, "hyb.bin")
    check("save slot0", st == 200 and saved.get("status") == "completed" and saved["result"]["n_saved"] > 0, saved)
    fsize = (work / "hyb.bin").stat().st_size if (work / "hyb.bin").exists() else 0
    print("snapshot bytes", fsize, "n_saved", saved.get("result"), flush=True)

    # continuation request used for the live-vs-restore comparison
    Q2 = Q + gen1[:8]
    st, live = complete(0, Q2, 48)
    check("live continuation", st == 200 and len(live.get("tokens", [])) == 48)

    import hashlib
    print("OFFPROOF snapshot_sha256", hashlib.sha256((work / "hyb.bin").read_bytes()).hexdigest(), "gen1_tokens_sha256",
          hashlib.sha256(json.dumps(gen1).encode()).hexdigest(), "live_tokens_sha256",
          hashlib.sha256(json.dumps(live.get("tokens")).encode()).hexdigest(), flush=True)
    m0 = metrics()
    print("metrics before restore", json.dumps(m0), flush=True)

    if args.scenario == "off":
        st, er = op("erase", 0)
        check("OFF erase", st == 200 and er["status"] == "completed", er)
        st, rs = op("restore", 0, "hyb.bin")
        check("OFF restore completed", st == 200 and rs["status"] == "completed" and rs["result"]["n_restored"] == saved["result"]["n_saved"], rs)
        m1 = metrics()
        check("OFF planner untouched (0 reservations, 0 releases, 0 clears, 0 refusals)",
              m1.get("reservations") == 0 and m1.get("releases") == 0 and m1.get("executor_clears") == 0 and m1.get("plan_refusals") == 0, m1)
        check("OFF /props kv_admission disabled or absent", not desc or desc.get("enabled") in (False, None), desc)
        st, after = complete(0, Q2, 48)
        check("OFF restore continuation == live", st == 200 and after.get("tokens") == live.get("tokens"))
    elif args.scenario == "b1":
        st, rs = op("restore", 0, "hyb.bin")
        check("B1 restore blocked (unchanged legacy gate)", st == 200 and rs["status"] == "failed" and "B1 blocks slot restore" in rs.get("error", ""), rs)
    else:
        check("admission supported for hybrid", bool(desc) and desc.get("enabled") is True and desc.get("supported") is True, desc)
        if args.scenario == "on":
            pr = (desc or {}).get("plan_restore", {})
            print("plan_restore descriptor:", json.dumps(pr), flush=True)
            check("declared keys att+rec", sorted(map(str, pr.get("declared_keys", []))) != [] and len(pr.get("declared_keys", [])) == 2, pr)

            # restore onto slot 0 itself (D1: occupied dest replacement; slot 1 idle)
            st, rs = op("restore", 0, "hyb.bin")
            check("ON restore slot0 (occupied dest, replace)", st == 200 and rs["status"] == "completed" and rs["result"]["n_restored"] == saved["result"]["n_saved"], rs)
            m1 = metrics()
            check("ledger: +1 reservation, +1 release, 0 active", m1["reservations"] == m0["reservations"] + 1 and m1["releases"] == m0["releases"] + 1 and m1["active_restore_tokens"] == 0, m1)
            check("limits by key present (att+rec)", set(m1.get("last_target_limit_by_key", {})) == {"attention", "recurrent"}, m1)
            st, after = complete(0, Q2, 48)
            check("ON restore continuation == live (HTTP, greedy)", st == 200 and after.get("tokens") == live.get("tokens"))

            # slot 1 idle: pool has 2 cells, slot 0 holds 1 -> limit_rec == 1
            st, rs = op("restore", 1, "hyb.bin")
            check("ON restore slot1 (idle dest, other slot occupied)", st == 200 and rs["status"] == "completed", rs)
            m2 = metrics()
            print("metrics after slot1 restore", json.dumps(m2), flush=True)
            check("slot1 limit_rec == 1", m2["last_target_limit_by_key"]["recurrent"] == 1, m2)
            # NOTE: completions on a full recurrent pool are refused by the pre-existing completion admission (see fullpool
            # scenario), so slot 1 is verified by re-saving it: recurrent tail + size must equal the original snapshot.
            st, s1 = op("save", 1, "hyb1.bin")
            orig = (work / "hyb.bin").read_bytes(); got = (work / "hyb1.bin").read_bytes() if (work / "hyb1.bin").exists() else b""
            check("slot1 re-save: same size and recurrent tail (16 KiB) as the original snapshot",
                  st == 200 and len(orig) == len(got) and orig[-16384:] == got[-16384:], (st, len(orig), len(got)))

            # both slots occupied: pool full; restore replaces own cell thanks to D1 credit
            st, rs = op("restore", 1, "hyb.bin")
            m3 = metrics()
            print("metrics full-pool replace", json.dumps(m3), flush=True)
            check("full pool: replacement restore admitted with D1 credit", st == 200 and rs["status"] == "completed", rs)
            check("full pool: limit_rec == credit == 1", m3["last_target_limit_by_key"]["recurrent"] == 1, m3)
            check("ledger drained", m3["active_restore_tokens"] == 0 and m3["plan_refusals"] == m0["plan_refusals"], m3)

            # restore of a corrupted snapshot is refused without breaking the server
            raw = (work / "hyb.bin").read_bytes()
            (work / "hyb-trunc.bin").write_bytes(raw[: len(raw) // 2])
            st, rs = op("restore", 1, "hyb-trunc.bin")
            check("truncated snapshot restore fails (terminal failed)", st == 200 and rs["status"] == "failed", rs)
            m4 = metrics()
            check("ledger drained after failed restore", m4["active_restore_tokens"] == 0, m4)
            check("server healthy after failure", http("GET", "/health")[0] == 200)
            st, s1b = op("save", 1, "hyb1b.bin")
            got2 = (work / "hyb1b.bin").read_bytes() if (work / "hyb1b.bin").exists() else b""
            check("slot1 intact after refused truncated restore (re-save == earlier re-save)", st == 200 and got2 == got)
            st, er = op("erase", 1)
            check("erase slot1", st == 200 and er["status"] == "completed", er)
            # erase + restore round
            st, er = op("erase", 0)
            check("erase slot0", st == 200 and er["status"] == "completed", er)
            st, rs = op("restore", 0, "hyb.bin")
            check("restore into erased slot0", st == 200 and rs["status"] == "completed", rs)
            st, a3 = complete(0, Q2, 48)
            check("erase+restore continuation == live", st == 200 and a3.get("tokens") == live.get("tokens"), (st, a3.get("tokens") if st==200 else a3, live.get("tokens")))
        else:  # fixture
            def forced(env_name):
                return env_name

            st, rs = op("restore", 0, "hyb.bin")
            # the env var is set by the caller per run; report what happened
            m1 = metrics()
            print("fixture result:", st, json.dumps(rs)[:300], "metrics", json.dumps(m1), flush=True)
            exp = os.environ.get("B3_EXPECT", "")
            if exp == "accept":
                check("fixture: forced limit accepted", st == 200 and rs["status"] == "completed", rs)
                check("fixture: ledger drained, 1 clear", m1["active_restore_tokens"] == 0 and m1["executor_clears"] == m0["executor_clears"] + 1, m1)
                st, a = complete(0, Q2, 48)
                check("fixture accept: continuation == live", st == 200 and a.get("tokens") == live.get("tokens"))
            elif exp == "refuse":
                check("fixture: limit+1 refused (terminal failed)", st == 200 and rs["status"] == "failed" and "refused" in rs.get("error", ""), rs)
                check("fixture: zero clear, 1 refusal, ledger 0", m1["executor_clears"] == m0["executor_clears"] and m1["plan_refusals"] == m0["plan_refusals"] + 1
                      and m1["active_restore_tokens"] == 0, m1)
                st, a = complete(0, Q2, 48)
                check("fixture refuse: prior slot intact (continuation == live)", st == 200 and a.get("tokens") == live.get("tokens"))
            elif exp == "report":
                print("FIXTURE_OUTCOME", rs.get("status"), (rs.get("error") or "")[:120], flush=True)
                st, a = complete(0, Q2, 48)
                m5 = metrics()
                print("FIXTURE_AFTER clears_delta", m5["executor_clears"] - m0["executor_clears"], "ledger_active", m5["active_restore_tokens"],
                      "continuation_equals_live", st == 200 and a.get("tokens") == live.get("tokens"), flush=True)
            else:
                check("B3_EXPECT set", False)
finally:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    logf.close()

print(json.dumps({"scenario": args.scenario, "pass": len(results) - len(failures), "fail": len(failures), "failed": failures}))
sys.exit(1 if failures else 0)
