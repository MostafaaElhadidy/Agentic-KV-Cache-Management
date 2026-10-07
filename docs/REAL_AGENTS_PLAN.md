# Plan: a REAL multi-agent workload (branch `real-agents`)

Goal: complement the synthetic traces with sessions in which real agents (Qwen2.5-1.5B-Instruct) talk to each
other, call real tools and solve real GSM8K problems, then measure CacheScout vs vanilla on that workload.
**Nothing existing is deleted or changed in behaviour.** The plugin, simulator, synthetic traces, results, and
the 86 tests stay as they are. New code is added next to them; small extensions only add new options.

Safety rules unchanged:
- every GPU job runs through `scripts/gpu_run.sh`, one at a time, with nothing CPU-heavy alongside;
- vllm/torch/transformers are never touched;
- every step is logged in `docs/WORK_LOG.md`;
- no fabricated or hand-edited numbers;
- I commit before each campaign, so each result.json records a clean commit (`git_dirty=false`).

## 0. Facts checked before planning (read-only)
- The Qwen2.5 tokenizer and chat template load offline from the local HF cache.
- The chat template puts 3 header tokens (`<|im_start|>system\n`) before the system text.
- If no system message is given, Qwen inserts its default "You are Qwen, created by Alibaba Cloud...". Every
  prompt will therefore carry an explicit system message, and each agent's system text starts with its
  **name** (e.g. `PLANNER.`), so the plugin's 2-block (32-token) fingerprint differs per agent from token ~4.
- `datasets` is not installed and is not needed: GSM8K is two small JSONL files (see §3).

## 1. Agents, tools, topologies: `src/cachescout/agents/`
| Id | Agent | Role | Tools |
|---|---|---|---|
| P | PLANNER | breaks the problem into steps, decides who acts | scratchpad |
| A | ANALYST | extracts the given quantities and the question | scratchpad |
| C | CALCULATOR-AGENT ("coder") | computes intermediate results | calculator, scratchpad |
| T | TESTER | re-checks a computation independently | calculator |
| R | REVIEWER | critiques the current solution, flags errors | none |
| D | DECIDER (judge) | states `FINAL ANSWER: <number>` or sends work back | none |

- **Anchors (system prompt + tool definitions):** 150–300 tokens per agent. Each has a distinct opening within
  the first 32 tokens, and the test suite checks that the 6 fingerprints differ with the real tokenizer.
- **Prompt for each call** (built with `tokenizer.apply_chat_template`, then sent as token IDs so prefixes are
  exact):
  - `system`: the agent's anchor;
  - `user`: `Task: <GSM8K question>`;
  - then the shared group-chat history, as alternating user messages `[<AGENT>]: <text>` and tool results
    `[tool:<name>] <result>`.

  This is the AutoGen group-chat structure: anchor + shared history.
- **Tools really execute:**
  - `calculator`: safe arithmetic through Python's `ast` (+ − × ÷ ** %, parentheses, decimals; no names or
    calls).
  - `scratchpad`: per-session notes, `write`/`read`.
  - An agent calls a tool by writing `CALL calculator: <expr>` or `CALL scratchpad: write <text>` / `read`.
  - The driver executes the tool, appends `[tool:...] <result>` to the history, and calls the **same** agent
    again (a multi-call invocation), at most 2 tool calls per invocation.
- **Next-agent rule per topology:**
  - `pipeline`: fixed chain P→A→C→T→R→D.
  - `random`: uniform over the other 5 agents (seeded RNG); the session ends when D answers or the cap is hit.
  - `debate`: P→C→R→D. D either ends with a final answer or replies `REVISE`, which sends the work back to C,
    so the loop is C→R→D.
  - `selector`: every agent must end with `NEXT: <AGENT>`, which is parsed. If it's missing or invalid, a
    fixed fallback is used (back to PLANNER, or DECIDER after the turn cap). **Fallbacks are counted** and
    reported.
- **Termination:** D outputs `FINAL ANSWER: <number>`, or the session hits a cap of 14 LLM calls. If the cap is
  reached, D is forced to answer, and the session is marked `capped`.
- **History trimming:** prompt tokens + `max_tokens` must be ≤ `max_model_len` (1584, which fits the 100-block
  budget). If too long, the oldest history messages are dropped; system, task and the latest messages are kept.
  Trimming is counted, since it changes prefixes and reduces reuse.
- **Decoding:** temperature 0, `max_tokens` 128 per call (stops at EOS, so real output lengths vary), seed fixed.

## 2. Driver: `drive_agents` in `run.py` (`mode: agents`)
- Same in-process `AsyncLLM` as now, so the `--system` switch, `scheduler_cls`, `CACHESCOUT_CONFIG` and
  `num_gpu_blocks_override` all apply unchanged.
- Poisson session arrivals (seeded). `session_id` is passed to `engine.generate`. Tool execution time is real;
  a fixed optional "think" delay between agent invocations defaults to 0.2 s.
- Output per call: the same `TurnRecord` as now (`agent_id` = the agent letter, `turn_idx`). Per session:
  completion time, final answer, gold answer, correct (numeric match), number of calls, fallbacks, trims,
  capped flag.
- WarmupCoordinator: unchanged. It sees the real prompt tokens, exactly as with synthetic traces.
- `compare.py`: add `--workload <name>` so it finds `results/<exp>/<workload>/...` without a trace file. Its
  existing behaviour stays the default.

## 3. Tasks: GSM8K (download needs your OK, part of approving this plan)
- **Source:** the official GSM8K files from https://github.com/openai/grade-school-math (MIT license):
  `grade_school_math/data/test.jsonl` (~0.75 MB, 1,319 problems) and `train.jsonl` (~4 MB, 7,473 problems).
- Download with `curl` into `data/gsm8k/` (git-ignored). SHA-256 checksums are recorded in
  `data/gsm8k/SHA256SUMS`, plus `scripts/get_gsm8k.sh` to re-download. No Python package is installed.
- **Disjoint sets:**
  - tuning problems from `train.jsonl` (indices chosen with seed 101);
  - evaluation problems from `test.jsonl` (seeds 1/2/3 select 3 disjoint problem sets).
- Scoring: the gold answer is the number after `####`, compared numerically with the parsed `FINAL ANSWER`.

## 4. Transcripts: `results/real/`
- Every session is saved as JSON: problem, gold answer, every call (agent, prompt text, prompt token count,
  output text, tool call + result, routing decision and why, fallback/trim flags, timings), final answer and
  correctness.
- `scripts/show_session.py <run_dir> [--session ID]` prints one session as a readable conversation, for your
  instructor.

## 5. Record-and-replay (the controlled comparison)
- **Record:** the live **vanilla** run at 100 blocks is the recording for each topology × seed. It stores the
  exact prompt token IDs, output lengths, arrival times and inter-call gaps.
- **Replay:** `mode: replay` turns a recording into a trace object that `drive_online` already accepts, so the
  existing replay path is reused. The same prompts are sent with `max_tokens` = the recorded output length and
  `ignore_eos`, under vanilla and CacheScout at 100/150/200 blocks. This removes all randomness from routing,
  answers and lengths, so only the cache policy differs.
- **Live closed-loop** runs are reported separately. There, each system generates its own outputs (temperature
  0, but batching can still change outputs slightly), so routing, answers and lengths may differ between
  systems. GSM8K accuracy should match across systems in replay by construction; live accuracy differences will
  be reported, not hidden.
- Bonus at no GPU cost: the simulator can replay the recordings too (`compare.py --sim`).

## 6. Tests (no GPU, fake engine)
- Calculator: correct results, and rejection of unsafe expressions.
- Scratchpad behaviour, `CALL` parsing, `NEXT:` parsing and the fallback counter, `FINAL ANSWER` parsing and
  GSM8K scoring.
- History trimming respects the budget and keeps system + task.
- Each topology rule: pipeline order, debate loop, random excludes self, selector fallback.
- Fingerprint distinctness of the 6 anchors: with the real tokenizer (skipped if not cached) and with a fake
  tokenizer.
- A full fake-engine session (canned outputs → tool call → routing → final answer → TurnRecords) and a
  record→replay round trip.
- All 86 existing tests keep passing.

## 7. Runs (in this order; each GPU job via gpu_run.sh, one at a time)
1. **Smoke:** 1 session per topology, vanilla, then one CacheScout run. Read the transcripts and check routing,
   tools and scoring by eye.
2. **Tuning (tuning problems only):**
   - record vanilla sessions on train problems (e.g. 20 per topology);
   - re-tune the constants in the simulator on those recordings (same grid as before);
   - adopt new constants only if the tuning objective improves. Constants are never chosen on test problems.
3. **Campaign:**
   - live: 4 topologies × {vanilla, cachescout} × {100, 150, 200} blocks × 3 seeds = 72 runs;
   - replay: 4 × 2 × 3 × 3 = 72 runs;
   - about 20 sessions per run (~200–280 LLM calls).

   Estimated ~2–3 min per run, ~6 h total. If time or memory gets tight, I'll cut sessions per run first and log
   it. The order is replay first (the controlled result), then live.

## 8. Report: `docs/REPORT_REAL_AGENTS.md`
- Measured routing predictability R per topology, engine next-agent prediction accuracy, selector fallback
  rate, trimming rate, and GSM8K accuracy per system (live and replay).
- Hit rate, TTFT, per-turn latency and session completion time (live and replay separately).
- Every number links to a result file. An honest comparison with the synthetic results and with the paper.
  No tuning to match the paper.

## 9. README and commits
- Keep the synthetic results, labelled "synthetic traces".
- Add a "Real multi-agent workload" section with the new results, an updated Mermaid diagram, and an updated
  scope box.
- Remove only the caveats that become false: "no real agents / no GSM8K" goes; "small model" and "single GPU"
  stay.
- Commit on `real-agents` before each campaign and at the end. **No push and no merge without your approval.**

## Risks
- **Qwen2.5-1.5B may not follow the formats** (`CALL`, `NEXT:`, `FINAL ANSWER`). The parsers are tolerant,
  fallbacks are counted, and the cap prevents endless sessions. A high fallback rate will be reported as a
  finding, not hidden.
- **Real prompts may be longer than the synthetic ones,** so more trimming and less cache headroom at 100
  blocks. All of it is measured and reported.
- **Live runs are nondeterministic across systems,** which is why replay is the primary comparison.
- **GPU time is long (~6 h).** Runs are resumable (existing results are skipped) and committed between stages.

## Amendments approved by the user (2026-10-06)
1. **Pilot gate:** selector topology, 100 blocks, ~5 sessions per system (live; vanilla and CacheScout). Then
   STOP and report: selector fallback rate, tool-call parse failures, FINAL ANSWER rate, GSM8K accuracy, hit rate
   and trim rate. No campaign until the user says go.
2. **Before tuning:** cross-check the simulator against real vLLM (sequential mode) on recorded real sessions. If
   they don't match, tune with real GPU runs on tuning problems only.
3. **Trimming** keeps anchor + task and drops the oldest messages after them; the trim rate is reported per run.
4. `scripts/fetch_gsm8k.sh` (curl + checksum verification), mentioned in the README quickstart.
5. Different seeds use different GSM8K problems and arrival times; the problem ids are reported.
6. The selector deviation (the agent's own NEXT line instead of AutoGen's separate selector call) is documented
   in docs/decisions.md.
7. Before the final report: verify `git log --format='%an <%ae>' | sort -u` shows only the noreply address.
Resume rule: on a usage limit, error or crash, append the current state and the exact next step to
docs/WORK_LOG.md first. Commit as I go on `real-agents`. No campaign, no push, no merge without approval.

## Status (2026-10-07)
Done: agents/tools/routing/driver/record-replay, 119 tests, GSM8K fetch, pilots, router pilot (rejected per
criterion), sim cross-check on real sessions (155/155 exact), λ re-tuning on train recordings, 12 recordings,
72 replay runs, 60 more live runs, docs/REPORT_REAL_AGENTS.md, README. Not pushed/merged (awaiting approval).
