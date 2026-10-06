# Paper notes: CacheScout

**Paper:** "Learning Agent Execution for KV-Cache Management in Agentic Serving" (Research Full),
Zhang, Kim, Feng, Du, Liu, Zhong, Ching, Jiang, Hu. arXiv:2608.14624v1 [cs.AI], 16 Jul 2026, 14 pages.
File: `paper/cachescout.pdf`.

Conventions: "Sec. X" = paper section, "Eq. N" = paper equation, "Fig. N" / "Tab. N" = paper figure/table,
"p. N" = PDF page. `[UNCERTAIN]` = the paper does not state this clearly; see `docs/open_questions.md`.
Notes were made from the PDF's extracted text; figure values were read from the figure text labels
(the figures were not rendered as images), so figure-only numbers carry a small transcription risk.

---

## 1. One-paragraph summary

In multi-agent LLM applications (planner + specialist agents), every invocation of the same agent starts with
the same fixed prompt prefix (system prompt, tool definitions, skills, few-shot examples), which the paper calls
the **agent anchor**. Anchors make up 53–62% of all prompt tokens in the four evaluated workloads (Fig. 2), so
their KV-cache is highly reusable, even across user sessions. But serving engines like vLLM manage the KV-cache
*reactively*: content-hashed blocks with LRU eviction. An agent's anchor is evicted while other agents run and must
be recomputed when that agent is called again (Sec. 1, Fig. 1). **CacheScout** is a lightweight runtime layer
on vLLM that (1) learns, online, a first-order Markov chain of which agent follows which (Sec. 3.2), (2) turns
that into a per-agent "survival score" via BFS hop distance on a thresholded transition graph, blended with
recency and block size to rank eviction candidates (Sec. 3.3, Eqs. 7–9), and (3) issues background "warmup"
requests for the predicted next agent's anchor between requests, gated on predictability (Sec. 3.4, Eqs. 10–11).
The core claim: future KV reuse is governed by agent execution semantics, not recency alone.

---

## 2. Background (simple explanations)

- **KV-cache.** When an LLM processes a prompt ("prefill"), every layer computes Key and Value tensors for every
  token. Storing them lets later tokens attend to earlier ones without recomputation. Prefill is the expensive part
  of time-to-first-token (TTFT).
- **Prefix caching (Sec. 2.1).** If a new request starts with exactly the same tokens as an earlier one, the stored
  KV for that shared prefix can be reused. vLLM splits prompts into fixed-size **blocks** (16 tokens by default),
  identifies each block by a hash of its content *and all preceding content*, and reuses matching blocks. SGLang
  uses a radix tree instead. Either way, reuse is identified purely by prompt content.
- **Eviction.** GPU memory holds a limited number of KV blocks. Blocks no longer used by a running request become
  "free but cached". When a new block is needed, vLLM reuses the least recently used free cached block (LRU), and
  its cached content is lost (unless offloaded to CPU).
- **vLLM V1 internals relevant here** (vLLM v0.11 per Sec. 4; our installed vLLM is 0.31.0):
  `BlockPool` (holds blocks, the hash→block map, and `touch()`), `FreeKVCacheBlockQueue` (LRU-ordered free list,
  in `kv_cache_utils.py`), `get_new_blocks()` (takes from the head of the free queue, evicting its cached content),
  the scheduler, the engine core, and the CPU-offload KV connector (scheduler- and worker-side halves).
- **Agentic serving.** A user request becomes a *session* of several *turns*; each turn is one LLM call made by some
  agent. In AutoGen's **SelectorGroupChat**, an LLM picks the next speaking agent at runtime, so no fixed workflow
  graph exists ahead of time (Sec. 2.3, Challenge #2).
- **Agent anchor (Sec. 2.1).** The reusable fixed prompt prefix of an agent: system prompt + tool defs + skills +
  optional few-shot examples.
- **Reactive vs. proactive.** Reactive = decisions based only on past accesses (LRU/LFU/ARC/LIRS). CacheScout is
  proactive: decisions use a prediction of which agent runs next.

---

## 3. System components (Sec. 3.1, Fig. 6)

CacheScout sits **between the agent framework and the serving engine**. Sec. 3.1 claims it "introduces no changes
to either component", but Sec. 4 says it is an 800-line patch to 5 vLLM files (see §9, Paper inconsistencies).

| Component | Section | Input | Output | Connects to |
|---|---|---|---|---|
| Transition Learner | 3.2 | Stream of agent dispatches (agent IDs, identified from prompt-prefix fingerprints, Sec. 4) | Counts `C_ij`, matrix `P` (Eq. 3), next-agent distribution (Eq. 4), argmax (Eq. 5) | Feeds Scorer (②) and Prefetch (③) |
| Survival Scorer | 3.3 | `P`, current agent `a_t`, per-block agent mapping and last-access time, block size | Score per cached block (Eq. 9); lowest evicted first | Hooked into vLLM eviction path (④) |
| Background Prefetch Coordinator | 3.4 | `P`, `a_t`, predictability `R` (Eq. 2) | A warmup request (anchor only, `max_tokens=1`) for `a* = argmax P(a|a_t)` if `R ≥ R_min` | Sent through the normal serving API (⑤) |

Numbered flow (Fig. 6 caption, Alg. 1): ① each dispatch observed → transition matrix updated; ② matrix feeds the
scorer; ③ matrix feeds prefetch; ④ scorer converts hop distances to survival scores guiding GPU KV eviction;
⑤ coordinator issues a background warmup for the predicted agent's anchor, "reloading it from the external KV pools
when present" (CPU memory / disk KV pools shown in Fig. 6).

**Runtime primitives (Sec. 4):** `ObserveTouch()`, `Predict()`, `ScoreBlock()`, `Warmup()`.
Alg. 1 and Fig. 15b also mention `PredictSurvival`.

**Runtime data structures (Sec. 4):** online transition matrix; sparse execution graph with shortest-path
distances; hop-distance table; block-to-agent mapping. Total < 25 KB.

**Three vLLM hooks (Sec. 4):**
1. Block pool calls `ObserveTouch()` **at prefix-matching time**. "Each request is fingerprinted from its own
   prefix block hashes, so a fingerprint change signals an agent dispatch", which updates transition statistics and
   the block-to-agent map "without any framework cooperation". [UNCERTAIN: how many prefix blocks form the fingerprint.]
2. Eviction path calls `ScoreBlock()` (Eq. 9) instead of LRU ordering, "additionally preferring blocks whose
   contents are already resident in the CPU tier". Survival probabilities computed once per scheduler step and cached.
3. `Warmup()` runs between agent turns via the standard serving API: after the `R ≥ R_min` gate and **a rate limit**
   [UNCERTAIN: value], it issues an inference request with only the predicted agent's anchor and `max_tokens=1`;
   the engine recognizes it and excludes it from transition learning [UNCERTAIN: how it is recognized].

Other implementation facts (Sec. 4): ~2,300 lines of runtime logic + 800-line patch; patched files = block pool,
engine core, scheduler, and the scheduler- and worker-side halves of the CPU-offload connector. Enabled/disabled by a
single environment variable [UNCERTAIN: name]. Reuses vLLM's LMCache and CPU-offloading infrastructure unmodified.
Transition counts update on each dispatch; graph + hop table refreshed only when the current agent changes; eviction
hot path does a single cached score evaluation.

---

## 4. Algorithms and equations (transcribed)

**Eq. 1, agent anchor ratio (Sec. 2.2.1):**
```
φ = (system prompts + tool definitions + skills + few-shot) / total prompt tokens
```

**Eq. 2, relative entropy reduction / predictability (Sec. 2.2.2):**
```
R = 1 − H(A_{t+1} | A_t) / H(A_{t+1})
```
R = 0: next agent independent of current; R = 1: fully determined.

**Transition counting (Sec. 3.2):** for each observed transition A_t = i → A_{t+1} = j:  `C_ij ← C_ij + 1`.

**Eq. 3, smoothed transition probability:**
```
P_ij = P(A_{t+1}=j | A_t=i) = (C_ij + ε) / Σ_k (C_ik + ε)
```
ε = small smoothing constant [UNCERTAIN: value].

**Eq. 4:** `p_{t+1} = P_{i,:}` (row i of the matrix).

**Eq. 5:** `Â_{t+1} = argmax_j P_ij`.

**Eq. 6, survival probability (ideal, not computed):**
```
p_surv(a; a_0, K) = Pr( ∃ k ∈ [1, K] : A_{t+k} = a | A_t = a_0 )
```

**Eq. 7, sparse execution graph:** `(a, b) ∈ E  ⇔  P(b | a) ≥ τ`, with τ a confidence threshold
[UNCERTAIN: value; Fig. 7 illustrates τ = 0.5].

**BFS (Sec. 3.3, Fig. 7):** a single BFS from the current agent over E gives the minimum hop distance E[a]
to each reachable agent (current agent: 0; unreachable: ∞).

**Eq. 8, approximate survival score:**
```
p̃_surv(a) = 1 − min(E[a], E_max) / E_max
```
So: hop 0 → 1; hop 1 → 1 − 1/E_max; hop 2 → 1 − 2/E_max; hop ≥ E_max (or unreachable) → 0 (Fig. 7 table).
E_max bounds the prediction horizon [UNCERTAIN: value].

**Eq. 9, block eviction score (lower = evicted first):**
```
Score(b) = ( p̃_surv(a_b) + δ ) · ( e^{−λ·age(b)} + δ ) · |b|
```
a_b = anchor (agent) associated with block b; age(b) = scheduler steps since last access; λ = recency decay rate;
|b| = number of tokens the block caches; δ = small floor keeping both factors positive (cold start).
[UNCERTAIN: λ, δ values.] Interpretation given by the paper: expected prefill work lost by evicting b. When
execution is unpredictable, survival scores flatten, recency dominates, and the policy degrades toward LRU.

**Eq. 10, prefetch target:** `a* = argmax_a P(a | a_t)`. Warm up only that agent's anchor.

**Eq. 11, prefetch gate:** prefetch enabled iff `R ≥ R_min` (R from Eq. 2, evaluated continuously on the learned
model) [UNCERTAIN: R_min value and how R is estimated online].

**Algorithm 1, CacheScout runtime workflow (p. 8), transcribed:**
```
State: transition counts n(·,·); transition matrix P̂; current agent a_t ← ⊥; execution graph G;
       hop-count table E[·]; block-to-agent map agentOf[·]; last-access table lastAccess[·];
       decay rate λ, floor δ, threshold R_min.
 1: loop
 2:   r ← framework.NextRequest()            ▷ agent dispatch
 3:   engine.Serve(r)                         ▷ normal serving path
 4:   BetweenStep                             ▷ ③ off critical path
 5: end loop
During Serve, the block pool raises two events:
 6: on engine.pool.BlockTouch(b, s_b):
 7:   ObserveTouch(b, s_b)                    ▷ ① learn transition
 8: on engine.pool.Evict:
 9:   return argmin_{b ∈ B_evict} ScoreBlock(b)   ▷ ④ survival-guided eviction
10: procedure ObserveTouch(b, s_b)
11:   a ← AgentId(s_b)
12:   agentOf[b] ← a; lastAccess[b] ← now()
13:   if a ≠ a_t then
14:     if a_t ≠ ⊥ then
15:       n(a_t, a) ← n(a_t, a) + 1
16:       update P̂ from n(·,·)
17:     end if
18:     G ← ThresholdGraph(P̂, τ)              ▷ ② refresh scorer state
19:     E[·] ← Bfs(G, a)
20:     a_t ← a
21:   end if
22: end procedure
23: function ScoreBlock(b)
24:   a ← agentOf[b]
25:   age ← now() − lastAccess[b]
26:   return (p̃_surv(a) + δ) · (e^{−λ·age} + δ) · |b|
27: end function
28: procedure BetweenStep
29:   if R(P̂) ≥ R_min then
30:     a* ← argmax_a P̂(a | a_t)
31:     IssueWarmupRequest(a*)                ▷ ⑤ background warmup
32:   end if
33: end procedure
```
Observations (facts derived from the pseudocode, not stated in prose): (i) a transition is counted only when
the agent changes (`a ≠ a_t`), so self-transitions are never counted, consistent with the empty diagonals in
Fig. 5; (ii) `a_t` is a single global variable, not per session [UNCERTAIN: how concurrent interleaved sessions
are handled]; (iii) `s_b` (the argument used to derive the agent ID) is not defined [UNCERTAIN].

---

## 5. The learning component

- **Formulation:** online first-order Markov chain over agent IDs (Sec. 3.2). Not a neural model.
- **Features:** only the identity of the current agent (from the prompt-prefix fingerprint, Sec. 4).
- **Model:** count matrix `C` → smoothed row-normalized `P` (Eq. 3).
- **Training data:** the live stream of dispatches. "No offline training, no workflow annotations, and no replay of
  historical traces" (Sec. 3.2).
- **Training procedure:** increment one counter per observed agent change; O(1) update, bounded state (Sec. 3.2).
- **Online accuracy:** most-frequent-successor prediction reaches 76–86% top-1 within 50 dispatches (Sec. 2.2.2, Fig. 4b).
- **Window:** Fig. 6 labels the learner input "Sliding window" / "Dispatch history"; Sec. 3.2 describes cumulative
  counts [UNCERTAIN: whether counts are windowed or decayed].
- **Comparison point:** PBKV [51] uses an offline-trained neural predictor per workload (Sec. 6).

---

## 6. Evaluation setup (Sec. 5.1)

- **Hardware (main):** server with 8× NVIDIA RTX PRO 6000 Blackwell, 96 GB each [UNCERTAIN: how many GPUs a single run uses].
- **Model (main):** Llama-3.1-8B-Instruct. All systems use the same vLLM configuration, GPU memory budget,
  decoding parameters, and prefix-cache settings [UNCERTAIN: their values].
- **Model scale (Sec. 5.4):** Qwen3-235B-A22B-FP8, tensor parallelism over 4× H200 (141 GB, NVLink). Compared
  against vanilla vLLM only (Continuum's fork does not support this model).
- **Workloads:** GSM8K [5], MT-Bench [52], GAIA [27], SWE-bench [15], all run on "the same six-agent supervisor
  framework with tools and AutoGen's SelectorGroupChat for dynamic LLM-selected routing".
  [UNCERTAIN: agent roles/prompts/tools; how many sessions/problems; turns per session; arrival process.]
  Fig. 5 axis labels for the six agents: P, A, C, T, R, D [UNCERTAIN: full names; text mentions Planner, Coder,
  Reviewer, and a judge].
- **Coordination topologies (Sec. 2.2.3, Fig. 5):** Pipeline, Debate (Coder–Reviewer proposer–challenger exchange
  periodically adjudicated by a judge), Selector (SelectorGroupChat), Random. Used in the motivation study; the main
  evaluation uses SelectorGroupChat.
- **Baselines:** vanilla vLLM (reactive prefix caching, LRU over block-aligned KV); Continuum [20] (TTL-based pinning
  around tool-call boundaries, TTL = 0.3 s, "following the observed inter-turn latency in AutoGen").
- **Ablation (Sec. 5.3):** vanilla vLLM, eviction only, eviction + prefetch.
- **Metrics (Sec. 5.1):**
  - KV-cache hit rate = total_cached_tokens / total_prompt_tokens.
  - TTFT = request arrival → first generated token.
  - Per-turn latency = end-to-end latency of one agent invocation (prefill + decode).
  - Throughput = completed agent turns per second.
  - Also reported: median/P90/P99 TTFT and latency, max load within a mean-latency budget.
- **Load:** session arrival rate swept 0.2–50 sessions/s (Fig. 11; SWE-bench axis goes to 20); 0.2–2.0 sessions/s for
  the Qwen3 study (Fig. 13).
- **Cache budget study:** GPU cache budget 100–200 blocks (Fig. 14); budgets below the workload's ~93-block maximum
  request footprint are excluded [UNCERTAIN: which workload Fig. 14 uses].
- **Overhead study:** in-process microbenchmarks with 6, 12, 24 agents (Fig. 15).

---

## 7. Reported results to reproduce (trends first)

Motivation (Sec. 2.2):
- Recurring fixed context share of prompt tokens (Fig. 2): GSM8K 54%, MT-Bench 53%, GAIA 53%, SWE-bench 62%.
- Mean reuses per block (Fig. 3a): anchors 49–173, session history 12–15; anchor/history ratio 4×, 4×, 5×, 13×
  (GSM8K, MT-Bench, GAIA, SWE-bench, in figure order).
- φ stays 43–60% even a dozen turns into a session (Fig. 3b).
- R (Fig. 4a): Pipeline 1.0, Debate 0.78, Selector 0.57, Random 0.12.
- Online top-1 next-agent accuracy 76–86% within 50 dispatches (Fig. 4b).

Main results (Sec. 5.2, Llama-3.1-8B), values in figure order GSM8K / MT-Bench / GAIA / SWE-bench:
- Hit rate gain vs vLLM (Fig. 8a): +18 / +18 / +18 / +10 pp; CacheScout reaches 81–85% (Sec. 1).
- Mean TTFT reduction (Fig. 8b): 26% / 18% / 27% / 45%.
- Median TTFT reduction (Fig. 9): 52% / 14% / 51% / 25%; e.g. GAIA 231→114 ms, GSM8K 239→115 ms.
- P99 TTFT reduced 21–52% on three of four workloads; SWE-bench 711→342 ms.
- Mean per-turn latency reduction (Fig. 10a): 38% / 32% / 29% / 32%.
- Peak throughput gain (Fig. 10b): 57% / 40% / 30% / 19%.
- Within the same mean-latency budget: 1.7–12× the arrival rate of vLLM, 4.2–16× that of Continuum (Fig. 11).

Ablation (Sec. 5.3, Fig. 12):
- Eviction only: +18–22 pp hit rate vs vLLM (figure label "eviction alone recovers +20pp"); mean TTFT cut up to 46%.
- Prefetch alone adds at most 1 pp; "inert" alone.
- Full runtime on GAIA: per-turn latency 251 ms vs 347 ms eviction-only (−28%). Throughput +8–19% turns/s (label).

Scale (Sec. 5.4):
- Qwen3-235B: hit rate +8 / +8 / +7 / +13 pp (text: 7–13 pp). SWE-bench: mean TTFT −33–54% and per-turn latency
  −26–36% at every rate from 0.2–2.0 sess/s; throughput 5.2 → 7.2 turns/s (+37%).
- Cache budget (Fig. 14a): CacheScout 86–87% hit rate across 100–200 blocks; vLLM 64.4% → 76.6%.
  Label: "87% @ 100 blocks; vLLM: 77% at 2× memory".
- Load: CacheScout 86–87%, vLLM 65–68% across 0.2–50 sess/s.
- Prefetch gate off: up to 22% higher per-turn latency at small cache sizes (Fig. 14b, label "1.2×").

Overhead (Sec. 5.5, Fig. 15):
- Coordinator state < 25 KB with 24 agents (labels 25 / 19 / 17 KB, presumably 24 / 12 / 6 agents [UNCERTAIN mapping]).
- ObserveTouch ≈ 1 µs per block touch; PredictSurvival ≤ 6 µs at 24 agents; label "< 8 µs P99", "O(1)".

**Most reproducible at small scale:** the hit-rate gap vs LRU under tight block budgets (Fig. 14a), the eviction-only
ablation dominance (Fig. 12a), degradation toward LRU under Random routing (Sec. 3.3), R values per topology (Fig. 4a),
and online prediction accuracy (Fig. 4b).

---

## 8. Every number/setting the paper specifies

| Setting | Value | Source |
|---|---|---|
| vLLM version | v0.11, V1 engine | Sec. 4 |
| Runtime size | ~2,300 lines + 800-line patch | Sec. 4 |
| Patched vLLM files | 5: block pool, engine core, scheduler, CPU-offload connector (scheduler + worker halves) | Sec. 4 |
| vLLM block size | 16 tokens (default) | Sec. 2.1 |
| Warmup `max_tokens` | 1 | Sec. 4 |
| Warmup content | predicted agent's system prompt + tool defs + minimal user prompt | Sec. 3.4 |
| Prefetch target count | top-1 agent only | Eq. 10 |
| τ (graph threshold) | not given; Fig. 7 example uses 0.5 | Eq. 7, Fig. 7 |
| ε, E_max, λ, δ, R_min, K, warmup rate limit | **not given** | Eqs. 3, 6, 8, 9, 11; Sec. 4 |
| age unit | scheduler steps (Eq. 9 text); `now() − lastAccess` (Alg. 1) | Sec. 3.3, Alg. 1 |
| Main model | Llama-3.1-8B-Instruct | Sec. 5.1 |
| Main hardware | 8× RTX PRO 6000 Blackwell 96 GB | Sec. 5.1 |
| Scale model | Qwen3-235B-A22B-FP8, TP=4 on H200 141 GB, NVLink | Sec. 5.1 |
| Number of agents | 6 (supervisor framework); overhead study 6/12/24 | Sec. 5.1, Fig. 15 |
| Routing | AutoGen SelectorGroupChat | Sec. 5.1 |
| Workloads | GSM8K, MT-Bench, GAIA, SWE-bench | Sec. 5.1 |
| Continuum TTL | 0.3 s | Sec. 5.1 |
| Arrival rate range | 0.2–50 sessions/s (main); 0.2–2.0 (Qwen3) | Figs. 11, 13 |
| GPU cache budget range | 100–200 blocks | Fig. 14 |
| Max request footprint | ~93 blocks | Sec. 5.4 |
| Hit rate metric | total_cached_tokens / total_prompt_tokens | Sec. 5.1 |
| Coordinator state | < 25 KB (24 agents) | Sec. 4, 5.5 |
| Hot-path latency | ObserveTouch ~1 µs; PredictSurvival ≤ 6 µs | Sec. 5.5 |
| Decoding params, max_model_len, gpu_memory_utilization, sessions per run, seeds, repetitions | **not given** | n/a |

---

## 9. Paper inconsistencies

Check each against the PDF yourself; page numbers are PDF pages.

1. **Leftover text from other papers in figures.**
   - **Fig. 1 (p. 1):** the figure's embedded text includes "Without CacheSage" / "With CacheSage" (not
     "CacheScout"), plus several overlapping versions of the same panel (labels "missing tier", "Planner anchor",
     "high survival score", and the travel panels repeated three times).
   - **Fig. 6 (p. 6):** the figure's embedded text includes labels unrelated to this paper: "§3.1
     Criticality-Tiered EC Layout", "Structural Criticality Scoring", "Binary Tier Selection", "EC(kc, mc)",
     "§3.3 Dual-Track Recovery Engine", "Failure Detection", "Star Recovery / Line / Tree Recovery",
     "KV loss event", "Distributed KV Cache Pool".
   - These were found in the extracted text layer. They may be hidden layers that are not visible in the rendered
     figure, so **open the figure and check visually**.
2. **Ablation gain exceeds main-result gain.**
   - Sec. 5.3 / Fig. 12a (p. 9, p. 11): **eviction alone** gives **+18–22 pp** over vLLM.
   - Sec. 5.2 / Fig. 8a (p. 9) and Sec. 1 (p. 2): the **full system** gives **+10–18 pp** (SWE-bench only +10).
   - The full system should be at least as good as eviction-only, so these numbers likely come from different
     configurations (cache budget, load), and the paper does not say which.
3. **Different CacheScout hit-rate ranges.**
   - **81–85%** across all workloads: Sec. 1, p. 2.
   - **86–87%**: Sec. 5.4 / Fig. 14a, p. 10, and the load sweep "across the full arrival-rate sweep of Fig. 11".
   - The load-sweep statement implies the main experiments also reach 86–87%, which contradicts 81–85%.
4. **"No changes to vLLM" vs. a patch.**
   - Sec. 3.1 (p. 6): CacheScout "introduces no changes to either component".
   - Sec. 4 (p. 8): an 800-line patch modifies five vLLM files, including the **scheduler**.
   - Sec. 4 (p. 8) then says the implementation works "without modifying attention kernels, memory allocators,
     **schedulers**, or block hashing".
5. **Prefetch-only result without a prefetch-only configuration.**
   - Sec. 5.3 (p. 9) lists three configurations: vLLM, eviction only, full.
   - It then reports that "background prefetch alone adds at most one percentage point", which would need a
     prefetch-only run that isn't in that list.
6. **age units.** Eq. 9 text says age counts *scheduler steps* (p. 7); Alg. 1 line 25 uses `now() − lastAccess[b]`,
   which reads as wall-clock time (p. 8).
7. **τ undeclared in Alg. 1.** The State line lists λ, δ, R_min but not τ, which line 18 uses (p. 8).
8. **Edge threshold boundary.** Fig. 7 legend (p. 7): "Kept edge (P(v|u) ≥ τ)" and "Pruned edge (P(v|u) ≤ τ)".
   Both include equality. Eq. 7 uses ≥ for kept edges.
9. **Sliding window.** Fig. 6 shows a "Sliding window" feeding the Transition Learner (p. 6), but Sec. 3.2 describes
   plain cumulative counts.
10. **Eviction rule beyond Eq. 9.** Sec. 4 (p. 8) says eviction "additionally" prefers blocks already resident in the
   CPU tier. This term does not appear in Eq. 9 or Alg. 1.
