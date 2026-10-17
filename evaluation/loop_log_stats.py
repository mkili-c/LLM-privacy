"""
C2: statistics of the loop's PUPA-SD run, read from its job logs (pu_loop_sweeps/pu_loop_sweep_<job>.out).

The logs hold one section per prompt, from "sampling the reference answers" to the "final (...)"
line, written by privacy_utility_loop.py and privacy_rewrite.rewrite. A section is matched to its
prompt id by its "original:" text; a section without a "final" line (a job cut off mid-prompt) is
dropped, and of several finished sections of one id the last counts, as in the rewrites CSV. Only
the eval set's ids count, and every final text is checked against results/rewrites/pupa_sd_loop.csv.

Per prompt (results/evaluation/loop_stats.csv):
    rounds          privacy rounds, i.e. iterations that scored spans (at most --privacy_max_iters)
    stop            done (every P <= tau) | stalled (a round changed nothing) | max_iters | no_span
    converged, leak, R, ref_s, privacy_s     from the "final" line
    per picked span, one outcome:
        accepted    the first rewrite passed the utility check (R < 100%)
        retry_in    refused, and the generalizing retry passed
        protected   refused, and the retry failed too: the span stays and is protected
        noop        the rewriter returned the message unchanged
        moved       an earlier rewrite in the round changed the span's words: blamed again next round
        resolved    nothing it is blamed for leaks any more: skipped
    ineffective     accepted rewrites that did not lower the hinge sum_a max(0, P_a - tau)
    stalls          [stall] lines: a leaking attribute whose best span blame is < 0.01 (saturated P)
    final_leaking   attributes with max P > tau at the end, and which attacker holds each
    words_orig, words_final, similarity      the edit's size (difflib ratio over words)
The summary is printed, including a cross-check with the judge: for the pairs the judge still
recovers from the loop's output, how the loop ended on that prompt.

    python evaluation/loop_log_stats.py
"""
import argparse
import ast
import csv
import difflib
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "evaluation"))

from read_judge_answers import LOOP_ATTRIBUTE, THRESHOLD                  # noqa: E402

LOGS = [REPO / "pu_loop_sweeps" / f"pu_loop_sweep_{job}.out" for job in (665, 674)]
# the loop's attributes, in the order its logs print them (attribute_probe.ATTRIBUTES)
ATTRIBUTES = ["gender", "age", "geographic location", "profession", "health condition",
              "education level", "socioeconomic status", "relationship status",
              "family situation", "nationality"]
OUTCOMES = ["accepted", "retry_in", "protected", "noop", "moved", "resolved"]
STOPS = {"  all attributes <= 0.5: done": "done",
         "  no rewrite changed the message: stop": "stalled",
         "  max_iters reached": "max_iters",
         "  no unprotected span lowers the leakage when deleted: stop": "no_span"}
BIG_CUT = 0.5              # a final text with under this share of the original's words lost most of it

SECTION = "  sampling the reference answers"
REFERENCE = re.compile(r"^  \d+ answers, D\(∅\) = ([\d.]+) \((\d+) s\)")
ITERATION = re.compile(r"^=== iteration (\d+): ")
OCCLUSION = re.compile(r"^  occlusion: (\d+) spans scored in (\d+) s")
STALL = re.compile(r"^  \[stall\] (.+?): best span blame")
# a span line: '    <repr of the span> (<attributes>)' with an optional ': <outcome>'
SPAN = re.compile(r"^    (?P<span>(['\"]).*\2) \((?P<attrs>[a-z ,]+)\)(?P<rest>: .*)?$")
SKIP = re.compile(r"^    (?P<span>(['\"]).*\2): (?P<why>an earlier rewrite changed it|nothing it is blamed for leaks any more)")
# one picked span on the "picked spans" line: <repr> (<attribute> <blame>)
PICK = re.compile(r"((['\"]).*?\2) \([a-z ]+ ([+-]\d\.\d{3})\)")
ACCEPT = re.compile(r"^      utility R (\d+)%: in$")
REFUSE = re.compile(r"^      utility R (\d+)% >= \d+%: request lost; (?P<how>.*)$")
STILL_LOST = re.compile(r"^      utility R \d+%: still lost")
PROTECT = re.compile(r"^      span kept and protected: (.*)$")
HINGE = re.compile(r"^      hinge ([\d.]+) -> ([\d.]+)")
FINAL = re.compile(r"^final \((?P<status>.*?), leak (?P<leak>[\d.]+), utility R (?P<R>[\d.]+)%; "
                   r"reference (?P<ref>\d+) s, privacy step (?P<priv>\d+) s\): (?P<text>.*)$")
P_VALUES = re.compile(r"\d\.\d{3}")


def get_args():
    p = argparse.ArgumentParser(description="Statistics of the loop's run from its job logs")
    p.add_argument("--data", default=str(REPO / "data" / "pupa_sd" / "pupa_sd_eval.csv"))
    p.add_argument("--rewrites", default=str(REPO / "results" / "rewrites" / "pupa_sd_loop.csv"))
    p.add_argument("--eval_dir", default=str(REPO / "results" / "evaluation"))
    return p.parse_args()


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def new_section():
    return {"rounds": 0, "stop": None, "spans_scored": 0, "occl_s": 0, "stalls": [],
            "picks": 0, "outcomes": Counter(), "by_attr": Counter(), "blames": [], "ineffective": 0,
            "refusals": [], "protected_spans": [], "d_empty": None}


def parse_log(path):
    """-> finished sections of one log, each a dict with the original text and its statistics."""
    sections, s = [], None
    span_attrs, span_blame, blame_of, retry, final_p = [], None, {}, False, []
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    for line in lines:
        if line == SECTION:
            s = new_section()
            continue
        if s is None:
            continue
        if m := REFERENCE.match(line):
            s["d_empty"], s["ref_s"] = float(m.group(1)), int(m.group(2))
        elif m := OCCLUSION.match(line):
            s["rounds"] += 1
            s["spans_scored"] += int(m.group(1))
            s["occl_s"] += int(m.group(2))
        elif m := STALL.match(line):
            s["stalls"].append(m.group(1))
        elif line.startswith("  picked spans: "):
            blame_of = {m.group(1): float(m.group(3)) for m in PICK.finditer(line)}   # span repr -> blame
            s["picks"] += len(re.findall(r"\([a-z ]+ [+-]\d\.\d{3}\)", line))
        elif line in STOPS:
            s["stop"] = STOPS[line]
        elif m := SKIP.match(line):
            outcome = "moved" if m.group("why").startswith("an earlier") else "resolved"
            count(s, outcome, [], blame_of.get(m.group("span")))
        elif m := SPAN.match(line):
            span_attrs, span_blame, retry = m.group("attrs").split(", "), blame_of.get(m.group("span")), False
            if m.group("rest") and "rewriter" in m.group("rest"):
                count(s, "noop", span_attrs, span_blame)
        elif ACCEPT.match(line):
            count(s, "retry_in" if retry else "accepted", span_attrs, span_blame)
        elif m := REFUSE.match(line):
            s["refusals"].append(m.group("how").rstrip(":"))
            retry = True
        elif STILL_LOST.match(line):
            pass                                # its "span kept and protected" line follows
        elif m := PROTECT.match(line):
            count(s, "protected", span_attrs, span_blame)
            s["protected_spans"].append((ast.literal_eval(m.group(1)), span_attrs))
        elif m := HINGE.match(line):
            s["ineffective"] += float(m.group(2)) >= float(m.group(1))
        elif line.startswith("original: "):
            s["original"], final_p = ast.literal_eval(line[len("original: "):]), []
        elif line.startswith("  qwen ") or line.startswith("  laya "):
            final_p.append([float(v) for v in P_VALUES.findall(line)])      # [10] per attacker
        elif m := FINAL.match(line):
            s.update(converged=not m.group("status").startswith("NOT"), leak=float(m.group("leak")),
                     R=float(m.group("R")) / 100, privacy_s=int(m.group("priv")),
                     final=ast.literal_eval(m.group("text")), final_qwen=final_p[0], final_laya=final_p[1])
            sections.append(s)
            s = None
    return sections


def count(s, outcome, attrs, blame):
    """One span outcome, overall and per attribute it was rewritten for, with the span's blame."""
    s["outcomes"][outcome] += 1
    s["blames"].append((outcome, blame))
    for a in attrs:
        s["by_attr"][(a, outcome)] += 1


def final_leaking(s):
    """{attribute: 'qwen' | 'laya' | 'both'} for the attributes with max P > tau at the end."""
    out = {}
    for a, q, l in zip(ATTRIBUTES, s["final_qwen"], s["final_laya"]):
        if q > THRESHOLD or l > THRESHOLD:
            out[a] = "both" if q > THRESHOLD and l > THRESHOLD else ("qwen" if q > THRESHOLD else "laya")
    return out


def match_rows(sections, data_path, rewrites_path):
    """{id: section} for the eval set's ids (the last finished section counts), and the ids whose
    final text differs from the rewrites CSV."""
    by_text = {r["original"]: r["id"] for r in read_csv(data_path)}
    rows = {}
    for s in sections:
        if s["original"] in by_text:
            rows[by_text[s["original"]]] = s
    csv_final = {r["id"]: r["rewrite"] for r in read_csv(rewrites_path) if r.get("rewrite")}
    mismatched = [i for i, s in rows.items() if csv_final.get(i) != s["final"]]
    return rows, mismatched


def row_record(i, s):
    words_o, words_f = s["original"].split(), s["final"].split()
    leaking = final_leaking(s)
    return {"id": i, "rounds": s["rounds"], "stop": s["stop"], "converged": s["converged"],
            "leak": s["leak"], "R": s["R"], "ref_s": s["ref_s"], "privacy_s": s["privacy_s"],
            "occl_s": s["occl_s"], "spans_scored": s["spans_scored"], "picks": s["picks"],
            **{o: s["outcomes"][o] for o in OUTCOMES}, "ineffective": s["ineffective"],
            "stalls": "; ".join(sorted(set(s["stalls"]))),
            "final_leaking": "; ".join(f"{a} ({who})" for a, who in leaking.items()),
            "words_orig": len(words_o), "words_final": len(words_f),
            "similarity": round(difflib.SequenceMatcher(None, words_o, words_f).ratio(), 3)}


def share(k, n):
    return f"{k}/{n} ({k / n:.1%})" if n else f"{k}/0"


def describe(xs):
    return f"median {st.median(xs):g}, mean {st.mean(xs):.1f}, range {min(xs):g}-{max(xs):g}"


def print_summary(rows, records):
    n = len(records)
    print(f"\n=== how the {n} prompts ended")
    for stop, k in Counter(r["stop"] for r in records).most_common():
        print(f"  {stop:<10} {share(k, n)}")
    print(f"  converged (max P <= {THRESHOLD} at the end): {share(sum(r['converged'] for r in records), n)}")

    print("\n=== privacy rounds per prompt")
    rounds = Counter(r["rounds"] for r in records)
    print("  " + "   ".join(f"{k} rounds: {rounds[k]}" for k in sorted(rounds)))
    for stop in ("done", "stalled"):
        xs = [r["rounds"] for r in records if r["stop"] == stop]
        if xs:
            print(f"  {stop:<8} {describe(xs)}")

    print("\n=== picked spans and what happened to them")
    total = Counter()
    for s in rows.values():
        total.update(s["outcomes"])
    picks = sum(r["picks"] for r in records)
    print(f"  {picks} spans picked in {sum(r['rounds'] for r in records)} rounds")
    for o in OUTCOMES:
        print(f"  {o:<10} {share(total[o], picks)}")
    blames = [ob for s in rows.values() for ob in s["blames"]]
    print("  blame of the picked span (its P drop when deleted), median per outcome: "
          + ", ".join(f"{o} {st.median(b for oo, b in blames if oo == o and b is not None):.2f}"
                      for o in OUTCOMES if any(oo == o for oo, _ in blames)))
    accepted = total["accepted"] + total["retry_in"]
    print(f"  accepted rewrites that did not lower the hinge: {share(sum(r['ineffective'] for r in records), accepted)}")

    print("\n  per attribute the span was rewritten for (a span can serve several):")
    by_attr = Counter()
    for s in rows.values():
        by_attr.update(s["by_attr"])
    print(f"  {'attribute':<22}" + "".join(f"{o:>11}" for o in OUTCOMES[:4]))
    for a in ATTRIBUTES:
        print(f"  {a:<22}" + "".join(f"{by_attr[(a, o)]:>11}" for o in OUTCOMES[:4]))

    print("\n=== utility check")
    refusals = [h for s in rows.values() for h in s["refusals"]]
    print(f"  refusals: {len(refusals)} on {sum(1 for s in rows.values() if s['refusals'])} prompts; "
          f"retries: " + ", ".join(f"{k} x '{h}'" for h, k in Counter(refusals).items()))
    for i, s in rows.items():
        for span, attrs in s["protected_spans"]:
            print(f"  protected  id {i:>5}: {span!r} ({', '.join(attrs)})")
    Rs = [r["R"] for r in records]
    print(f"  final R: {describe([round(100 * x) for x in Rs])} (%)")

    print("\n=== saturated attributes ([stall]: best blame < 0.01)")
    stalled_attr = Counter(a for r in records for a in r["stalls"].split("; ") if a)
    print(f"  prompts with a stall: {sum(1 for r in records if r['stalls'])}; by attribute: "
          + ", ".join(f"{a} {k}" for a, k in stalled_attr.most_common()))

    print("\n=== what still leaks at the end (our attackers, max P > tau), not converged prompts")
    holder = Counter()
    for s in rows.values():
        for a, who in final_leaking(s).items():
            holder[(a, who)] += 1
    print(f"  {'attribute':<22}{'qwen only':>11}{'laya only':>11}{'both':>8}")
    for a in ATTRIBUTES:
        if any(holder[(a, w)] for w in ("qwen", "laya", "both")):
            print(f"  {a:<22}{holder[(a, 'qwen')]:>11}{holder[(a, 'laya')]:>11}{holder[(a, 'both')]:>8}")

    print("\n=== edits")
    ratios = [r["words_final"] / r["words_orig"] for r in records]
    print(f"  final / original words: median {st.median(ratios):.2f}; under {BIG_CUT:.0%} of the words: "
          f"{sum(x < BIG_CUT for x in ratios)}; unchanged: {sum(r['similarity'] == 1.0 for r in records)}")
    print(f"  word similarity to the original: median {st.median(r['similarity'] for r in records):.2f}")

    print("\n=== runtime per prompt (s)")
    for key in ("ref_s", "privacy_s", "occl_s"):
        print(f"  {key:<10} {describe([r[key] for r in records])}")
    occl = sum(r["occl_s"] for r in records) / sum(r["privacy_s"] for r in records)
    print(f"  occlusion's share of the privacy step: {occl:.0%}; "
          f"seconds per round: median {st.median(r['occl_s'] / r['rounds'] for r in records if r['rounds']):.0f} (occlusion)")


def judge_cross(rows, eval_dir):
    """For the pairs the judge recovers from the original: the judge's outcome on the loop's output
    against how the loop ended and whether our attackers still flag the attribute."""
    outcomes = read_csv(eval_dir / "judge_outcomes.csv")
    counted = {(r["id"], r["attribute"]) for r in outcomes if r["version"] == "original" and r["outcome"] == "match"}
    loop = {(r["id"], r["attribute"]): r["outcome"] for r in outcomes if r["version"] == "loop"}
    table = Counter()
    for i, a in counted:
        s = rows[i]
        flagged = LOOP_ATTRIBUTE[a] in final_leaking(s)
        table[(loop[(i, a)], s["stop"], flagged)] += 1
    print(f"\n=== judge outcome on the loop's output x how the loop ended ({len(counted)} counted pairs)")
    print(f"  {'judge':<14}{'stop':<10}{'still flagged by ours':>24}{'pairs':>8}")
    for (o, stop, flagged), k in sorted(table.items(), key=lambda kv: (-kv[1])):
        print(f"  {o:<14}{stop:<10}{str(flagged):>24}{k:>8}")
    occ = Counter((loop[(i, a)], rows[i]["stop"], LOOP_ATTRIBUTE[a] in final_leaking(rows[i]))
                  for i, a in counted if a == "occupation")
    print("  occupation only: " + ", ".join(f"{o}/{stop}/flagged={f}: {k}" for (o, stop, f), k in occ.most_common()))


def main():
    args = get_args()
    sections = [s for log in LOGS for s in parse_log(log)]
    rows, mismatched = match_rows(sections, args.data, args.rewrites)
    n_eval = len(read_csv(args.data))
    print(f"{len(sections)} finished sections in {len(LOGS)} logs; {len(rows)} of {n_eval} eval ids found; "
          f"final text differs from the rewrites CSV for {len(mismatched)} {mismatched[:5]}")
    records = [row_record(i, s) for i, s in sorted(rows.items(), key=lambda kv: int(kv[0]))]
    out = Path(args.eval_dir) / "loop_stats.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print_summary(rows, records)
    judge_cross(rows, Path(args.eval_dir))
    print(f"\nper prompt -> {out}")


if __name__ == "__main__":
    main()
