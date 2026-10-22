# DeepSeek-V4.1-Flash engine: work log, flags, and measurements

This page is the record of the work on the DeepSeek engine (`src/ds4`, `maya-ds4` here, `strata-ds4` in the
standalone Strata-DeepSeek tree): every flag and environment knob added, what each one changes, and the
measurements and dead ends behind them. The development machine is a normal desktop: 2 x RTX 5060 8 GB
(CUDA 1 and 2) and 2 x RTX 3060 12 GB (CUDA 0 and 3, the displays), PCIe Gen 3 (x16 on the 3060s, x8 on the
5060s), 247 GiB RAM over two NUMA nodes, and the pack on `/mnt/nvme`. The model is
DeepSeek-V4.1-Flash-1017-UD-IQ3_XXS.

**The rule every feature here follows:** each one is opt-in behind its own flag or environment variable, and
off by default. New names only: no existing flag, environment variable, config key, KV-cache format or layout was
renamed or changed (the KV cache was not touched at all). With a feature's flag off, the engine takes the path it
took before. The two prompt paths (chunked and row-split) are alternatives by construction - a prompt takes one
or the other - and inside the chunked path the decoder tail and expert-parallel are mutually exclusive by their
own gate. Everything else composes: telemetry, placement, chunk sizing and MMQ do not fight each other.

The live configuration this work is measured on:

```json
"args": ["--pack", "<model>/maya-ds4", "--max-context", "262144",
         "--prefill-chunk-max", "8192,8192,8192,8192", "--reserve-pct", "2,1,1,2", "--verbose"],
"env":  { "DS4_SPLIT": "5,5,6,24", "DS4_TAIL_PREFILL": "1", "DS4_FLOW_STATS": "1", "DS4_PROF": "1" }
```

`DS4_SPLIT`, `DS4_FLOW_STATS` and `DS4_PROF` predate this work (upstream PRs #271/#275); the values are this
work's, and none of their names changed. The GPUs are `[0,2,3,1]`, so `5,5,6,24` puts 5 encoder layers on each of
the two 3060s, 6 on a 5060, and all 19 decoder layers plus the head on the last 3060.

---

## 1. Flags added by this work

| Flag | Env mirror | Default | Scope | What it does |
|---|---|---|---|---|
| `--reserve-pct P0,P1,...` | - | none | placement, all paths | VRAM kept free on each card as a per cent of its total, in the split's (CUDA) order; one value applies to every card, a short list repeats its last, floor 64 MiB. `--reserve-mib` still wins when given. The live split is `2,1,1,2` (2% on the display 3060s) |
| `--no-expert-lend` | - | lending on | placement, all paths | Keeps the prompt's chunk buffers and the decoder's expert banks in separate allocations - the pre-lending placement, for A/B and byte checks. Lending is what makes the same bytes prompt scratch during prefill and resident experts during decode |
| `--prefill-chunk-max N0,...` | `DS4_CHUNK_MAX` | none | chunked prompt only | A per-part ceiling on the prompt chunk, one entry per part in the split's order; 0/absent = no ceiling. A part's step is the largest whole multiple of its feeder's step at most its ceiling, so the parts' chunks nest and can differ. The live value is `8192,8192,8192,8192` |
| `--ram-resident` | - | off | CPU-lane tier only | Bare form of `--mmap-experts off`: every CPU-lane expert is copied into anonymous RAM up to the copy budget, the coldest staying in the file when RAM runs out. No value to write |
| `--ep-prefill` | `DS4_EP=1` | off | chunked prompt, single-chunk encoder pass | Expert-parallel prefill: every card streams and computes a share of *every* layer's non-resident experts, the layer's owner summing the shares over host-mapped memory. Gated to an encoder pass of one chunk with plain (non-MTP, `nm==1`) layers and no decoder tail |
| `--row-split-prefill` | `DS4_ROW_SPLIT=1` | off | a prompt path of its own | Every card holds the same K-layer group (K computed at load from the room the expert slots leave) and runs its own chunk of the prompt's rows through it, group by group, a wavefront ordering the (chunk, layer) cells. Any prompt at least `DS4_ROW_SPLIT_MIN` (8,192) rows runs, in rounds of the sum of the cards' chunk caps when it is larger |
| `--chunk-mmq` | `DS4_MMQ=1` | off | chunked prompt expert GEMMs | The prompt's routed experts through llama.cpp's MMQ kernels (int8 tensor cores on Ampere, the native FP4 product on Blackwell) instead of the F32 GGUF kernels; a window of `DS4_MMQ_ROWS` (1,024) activation rows at a time. Only a piece whose gate, up and down matrices are all MXFP4 on 17-byte blocks is taken; anything else keeps the GGUF kernels |

Environment-only knobs added with them (all off/unset in normal use):

| Env | Default | What it does |
|---|---|---|
| `DS4_TAIL_PREFILL=1` | off | The decoder tail: layers after the candidate-source layer only attend over the last `tail_rows()` rows (2,560 here), so a prompt runs in two passes - layers 0-20 over every token, then 21-39 over the tail, seeded with layer 20's saved hidden rows and its KV/index cache |
| `DS4_EP_ALLOC=1` | off | Allocate EP's staging/shares without running the path (the check that keeps placement equal between the two arms) |
| `DS4_EP_SELF=1` | off | Check arm: the owner keeps every expert and the partners none, so the expert arithmetic is the non-EP one |
| `DS4_EP_STATS=1` | off | Per-layer share/read lines for EP |
| `DS4_ROW_SPLIT_MIN` | `8192` | The least rows a prompt needs for the row-split path; `DS4_ROW_SPLIT_FORCE=1` takes any |
| `DS4_ROW_SPLIT_K`, `DS4_ROW_SPLIT_MIN_K` | plan, `4` | Layers a group; the planner starts at `K` and walks down to `MIN_K` |
| `DS4_ROW_SPLIT_BUDGET_MB`, `DS4_ROW_SPLIT_FRAC` | plan, `0.5` | The replica budget per card: a fixed MB, or a share of (free - reserve) the slot arenas may take. The default plan here is K=3, 14 groups, 1,244 MB a card; `FRAC=1.0` gives K=9, 5 groups (9,9,9,9,4), 2,464 MB |
| `DS4_ROW_SPLIT_W=d0,d1,d2,d3` | even | Start row weights per card (clamped [0.25, 4]); `DS4_ROW_SPLIT_BALANCE=0` holds them instead of learning |
| `DS4_ROW_SPLIT_EARLY_PUB=0` | on | Publish a layer's ring/compressor/key data when the compressor finishes (the default) instead of after the whole layer |
| `DS4_MMQ_ROWS` | `1024` | Activation rows an MMQ product takes |
| `DS4_RP_CHECK=1` | off | Temporary diagnostics (`ds4 rpok:`/`rpbad:`/`rpcomp:`) for the row-split wavefront; inert unless set |

---

## 2. What each work is, and its status

### 2.1 Expert lending from the prompt arenas (`d092e7bd`)

The prompt's chunk buffers and the decoder's expert banks now share one arena per card, re-described at each
phase switch: while decoding, the prompt's space holds lent expert slots; while a prompt runs, the drafter's
banks are evicted and the arena is zeroed back into chunk buffers. `--reserve-pct` makes the reserve per card and
`--no-expert-lend` reproduces the base placement for byte checks. Mixed-role layers take MXFP4 and low-bit types
side by side (the UD-IQ3_XXS pack's down matrix), and the chunk chain follows the nested prompt path.
**Status: live.**

### 2.2 The arena takes only what the prompt needs (`95cbd953`)

`enter_prefill` sizes each arena buffer's touched bytes from the prompt's row count (fixed + per-row x rows, plus
the 512-row kernel guard) and evicts only the lent experts that overlap them; the rest stay resident through the
prompt, hidden from its slot map. `exit_prefill` refills just the evicted slots. Measured on 2,282 tokens:
628 experts / 9.4 GB / 1,631 ms -> 384 / 5.8 GB / 1,031 ms (-39%). An agent turn of 12 new rows: nothing
evicted, refill 0, 0.84 s wall. Prompt tok/s is unchanged - this is the time around the prompt. **Status: live.**

### 2.3 The decoder tail (`6fd82c9a`)

In the chunked path only layers 0..cand_src (20) run over every token; the decoder layers 21-39 only ever see
the last `win + (nl - 1 - cand_src) * win` = 2,560 rows. With `DS4_TAIL_PREFILL=1` a prompt runs in two passes:
the encoder layers over every token, then the decoder layers over the tail alone, seeded from layer 20's saved
rows and its KV/index cache. The 5,5,6,24 split makes the encoder the pacing work (5/5/6/5 across the cards) and
puts all 19 decoder layers on one card, where the 2,560-row tail costs about two seconds. Measured: a
111,124-token prompt in about 193 s = **576 tok/s end to end** (settled chunks ~639) against **344.4 tok/s**
before. Two first-runtime crashes were found and fixed (see section 4, items 12-13). **Status: live.**

### 2.4 Expert-parallel prefill (`--ep-prefill`, uncommitted at this writing)

Every card streams and computes a share of every layer's non-resident experts; the layer's owner sums the shares
through host-mapped buffers (`ep_add`). The shares are index-strided over the non-resident list; the owner's own
resident slots stay pinned and are never in a share. Two implementations were measured:

1. **Whole-share streaming** (the first): the static `Streamer`/`Batch` plan copied each device's entire
   per-layer share at run start - 51.6 GB a card, 206 GB total, 20,655 ms for the 2,282-token prompt against
   16,311 ms for the normal path.
2. **Picked-range streaming** (the second): `copy_ahead` skips a batch whose bounds say no row picks it, EP has
   no cross-layer lookahead, the read-ahead thread is off, and the ring halves are allocated by a copied-batch
   counter with an in-flight window so a skipped batch takes no slot. Result: 17,035 ms and 18.7/18.6/19.0/17.9 GB
   per card (74.2 GB total) against the normal path's 15,711 ms and 18.5 GB total.

**Status: correct, still 8.4% slower on this pack**; keep it off. The reason and the remaining lever are in
section 4, item 3.

### 2.5 Row-split prefill (`--row-split-prefill`, uncommitted at this writing)

The user's scheme: give every card the same layer group instead of splitting the 40 layers across the cards,
split the prompt's rows one chunk a card, and order the (chunk, layer) cells with a wavefront. A chunk at layer L
needs only its predecessor's last 128 raw rows, its compressed-key rows and index keys - not its attention
output or experts - so the wavefront waits for bytes, not for a whole layer (the early-publish change,
`DS4_ROW_SPLIT_EARLY_PUB`). The plan is automatic: `rp_plan` takes the largest K whose slot arenas fit
`DS4_ROW_SPLIT_FRAC` x (free - reserve) on every card. Row shares are learned from measured per-card layer time
(EMS-style: weight = 1/2 old + 1/2 target, clamped [0.5, 2], renormalized) and persist for the process's life.
Prompts larger than the cards' caps run in rounds; the last chunk of the last round takes the logits and the
drafter taps.

**Status: works, correct (identical tokens, 0 faults), not a speed win on this host**: 16,797 tokens in
130,162 ms (129.0 tok/s) against today's 52,465 ms (312.3 tok/s) on the same binary - about 2.5x slower -
because every card streams all 384 experts of every layer. 40,935 tokens took 288,589 ms (141.8 tok/s) in 2
rounds. Its value is balance, completeness (any prompt size) and the wavefront machinery, not fewer bytes. The
next lever is the 5-group geometry (`FRAC=1.0`), measured only on 512 rows so far.

### 2.6 The MMQ prompt path (`--chunk-mmq`, uncommitted at this writing)

`src/ds4/ds4_mmq.cu` reads the same packed MXFP4 blocks the GGUF kernels read and multiplies them through
llama.cpp's MMQ kernels: q8_1 activations on Ampere, the native block-scaled FP4 MMA on Blackwell (the build
compiles `strata_mmq`'s 12.0 target as `120a-real` under CUDA >= 12.8). It applies only to a piece whose gate,
up and down matrices are all MXFP4 on 17-byte blocks; anything else is declined with one line and the GGUF
kernels run. **Status: built, guarded, no win on this pack** - every layer of UD-IQ3_XXS mixes types (IQ2_XS /
IQ2_XXS / IQ2_S gate/up with an MXFP4 down, or IQ3/Q3_K), so every piece is declined. Checked against a
4,564-row logits dump: top-1 agreement 4,564/4,564, prompt 33.9 s vs 33.6 s (the guard's arm, both effectively
the GGUF path). It is a capability for an all-MXFP4 pack; a per-role generalization is the open item.

### 2.7 The Monitor

The engine's `LIVE` per-device lines (stage, layer, chunk, position, loan) feed the Monitor's pipeline rows and
its Prefill/Loan windows; the server intercepts them, so a request's stream never sees them. The Prefill tab
gained the row-split wavefront strip (`renderWavefront` in `serve/web/app.js`): one row a card, cells for the
layers finished/current, the group (`g/G`), the round, and an explicit `wait=1` state from the engine. A stale
row (a busy card's last line older than the freshness window) keeps showing the layer it is on and is never
drawn as waiting - only the engine's own `wait=1` says held. `serve/test_dashboard.py` and
`serve/test_server.py` cover the new rows and facts. **Status: live.**

---

## 3. Measurements

| Scenario | Result |
|---|---|
| 111k prompt, before the tail (10/10/10/10 split, 16k chunks) | 113,302 tokens in 329,003 ms = **344.4 tok/s** |
| 111k prompt, tail on, split 5,5,6,24 | 111,124 tokens in ~193 s = **576 tok/s** end to end (settled chunks ~639) |
| 16k prompt, chunked path | 16,797 in 52,465 ms = 312.3 tok/s (new binary); 56,746 ms (288.7) on the pre-change binary |
| 16k prompt, row-split (K=3, 14 groups) | 130,162 ms = 129.0 tok/s (~2.5x slower) |
| 40,935-token prompt, row-split, 2 rounds | 288,589 ms = 141.8 tok/s |
| 111k prompt on today's path (row-split contrast) | 167,332 ms = 661.6 tok/s |
| 2,282-token prompt, normal path | 15,711 ms = 145.25 tok/s; 18.5 GB streamed (1.4 / 2.5 / 2.5 / 12.1 GB by card) |
| 2,282-token prompt, EP picked-range | 17,035 ms = 133.96 tok/s; 18.7 / 18.6 / 19.0 / 17.9 GB a card; exchange 4.21 / 4.21 / 4.31 / 5.99 GB |
| 2,282-token prompt, EP whole-share (before the fix) | 20,655 ms = 110.5 tok/s; 51.6 GB a card, 206 GB total |
| 2,282-token prompt, refill around it | 628 experts / 9.4 GB / 1,631 ms -> 384 / 5.8 GB / 1,031 ms |
| 2,282-token prompt, profile-cold probe | 176 GB in 11,992 experts (~77 MB/token) - the page cache at the PCIe cap, not a software limit |
| 4,564-row logits arm, MMQ guard | top-1 4,564/4,564, 33.9 s vs 33.6 s |

Every A/B in this table held everything else fixed: same ids, same split, same copied expert profile, only the
flag changed. Both arms produced identical greedy tokens in the EP and row-split checks (0 fault lines, rc=0).

---

## 4. Research findings

1. **The prompt is byte-bound at the PCIe link, not reader-bound.** All four cards run PCIe Gen 3: the 3060s at
   x16 calibrate to 10.6 GB/s DMA, the 5060s at x8 to ~7.9 GB/s. A profile-cold 2k prompt streams 176 GB and
   the per-card read waits equal bytes / link (15.6/10.6, 18.4/7.0, 23.2/7.7, 119.1/11.0 GB/s): the stream runs
   at the cap, and there is no software headroom. Eight reader threads reach 33 GB/s over page-cached data;
   `DS4_READERS`, `DS4_RING_MB` and `DS4_READAHEAD_MB` buy nothing. Raw disk is 0.6-0.8 GB/s, so the page cache
   is what keeps prompts at link speed - a hot-set eviction would collapse throughput. The only levers are
   compute/stream overlap and splitting the expert bytes across cards.
2. **The encoder/decoder insight is the big win.** In the chunked path only layers 0-20 see every token; layers
   21-39 can only attend over the last 2,560 rows. Running them over the tail alone, on one card, took the
   111k prompt from 344.4 to 576 tok/s. The tail gate is exact and refuses rather than approximating: it needs
   `n > tail_rows()`, a candidate source, every decoder layer's KV source equal to it, a head group with a
   decoder layer, and chunk steps at least `tail_rows()` on every group with decoder layers.
3. **The live path's advantage is its resident experts, not a smaller model.** The expert profile keeps the hot
   experts in each owner card's VRAM slots, so a 2,282-token prompt streams only the ~18.5 GB its routing misses;
   it is compute-bound, not byte-bound. EP gives every device a share of every layer and makes each one read its
   quarter of the *whole* model from the files, ignoring the residents the other cards already hold: 206 GB in
   the first implementation. Picked-range streaming cut that to 74.2 GB total, but a 32-expert batch around every
   pick still costs ~4x the picked bytes, leaving EP 8.4% behind. EP can only pay if its share planner lends and
   borrows the owners' resident slots (as decode does), or if the pack is profile-cold; finer-than-batch reads
   are the remaining lever.
4. **Row-splitting multiplies expert bytes by the number of cards.** Every card holds every layer of its group,
   and its chunk routes to ~all 384 experts of each layer, so the same prompt reads about 4x the expert bytes on
   four cards. Balance buys latency, not bandwidth; on this host the duplication cancels the parallelism (16k:
   130 s vs 52 s). The wavefront itself works and is correct; it is the byte duplication that loses.
5. **The wavefront's wait floor is bytes, not a layer.** A chunk at layer L needs only its predecessor's ring
   (128 rows, ~67.6 KB), the boundary compressor group (2 rows, ~8 KB), and once per source layer its compressed
   rows plus index keys (~1.2 MB + ~0.3 MB). Publishing them at the compressor instead of after the whole layer
   (`DS4_ROW_SPLIT_EARLY_PUB`, default on) removed the attention + MoE wait; the remaining critical path is the
   K/V projections, the indexer and the D2H copy, with the selector still ahead of the ckv finish - the next
   lever, not done.
6. **Row shares are measurable and move as expected.** With `DS4_ROW_SPLIT_W` unset, the first prompt uses an
   even split; the learned weights on the 512-row check moved to 0.93 / 0.93 / 0.96 / 1.18 toward CUDA3 (GPU1),
   which finishes its rows fastest, and the tokens were unchanged. Weights persist across rounds and prompts.
7. **Row-split geometry is a memory-budget answer, not a preference.** `rp_plan` takes the largest K with
   sum(per-layer slot arenas) <= `DS4_ROW_SPLIT_FRAC` x (free - reserve) on every card: frac 0.5 gives K=3,
   14 groups, 1,244 MB a device; frac 1.0 gives K=9, 5 groups (9,9,9,9,4), 2,464 MB; K=10 does not fit. The
   price is decode's expert slots (e.g. CUDA1 271 -> 92, CUDA2 246 -> 57, CUDA0 663 -> 393, CUDA3 184 -> 134).
   Wavefront overhead scales as (cards - 1)/K, so 14 -> 5 groups is the speed lever still to be measured at 16k.
8. **UD quants mix expert types inside one layer.** In this pack, gate/up are IQ2_XS (L0-1), IQ2_XXS (L2-4),
   IQ2_S (L5-9), IQ3_XXS (L15-24), Q3_K (L25-34), IQ3_XXS (L35-37, 39) and Q3_K (L38); the down matrix is
   MXFP4 only for L0-9 and L15-34, else Q3_K or IQ3_S. Expert blobs are 12.6-15.3 MB, not the 18,800,640 B of
   an all-MXFP4 blob. `lay.fmt` is `kFmtMXFP4` for every expert layer, so it alone does *not* mean the matrices
   are MXFP4 - `gguf::expert_type(lay, role)` does. A kernel that assumes MXFP4 reads past its slot (this was
   the MMQ crash of 2026-10-21: illegal access after ~23 layers of a 2.3k prompt). The MMQ path now guards on
   all three role types and declines the piece.
9. **Prompt logits are not bit-identical between runs.** Two runs of the same binary with the same flags differ
   in 590,033,832 of 590,033,920 values (max abs diff 11.7) because the prompt path's reductions are not
   order-stable, while top-1 agreement is 4,564/4,564. Judge a prompt-path change by top-1 agreement across all
   dumped rows plus prompt ms, never by max abs difference; byte identity is meaningful only where the changed
   path provably never runs.
10. **The profiler's event pool was one pool for all four device threads.** An event created on one GPU was
    recorded on another's stream, `cudaEventRecord` failed with "invalid resource handle", and every subsequent
    launch check printed it - 38,330 lines in the live log. Output was never affected; per-stage profiler
    numbers were. Fixed with a per-device pool (`prof_pool[d.id % 32]`) and a device set in
    `prof_collect` before `cudaEventElapsedTime`; a full 16k row-split run after the fix printed 0 lines.
11. **The row-split staging bug that blocked multi-layer groups**: `load_replica` saved the slot's compressor
    state (`c_kv`/`c_score`) out of the replica `Layer` it was about to overwrite, but a staged layer without a
    compressor nulled those pointers while the slot's allocation stayed; the next staged layer with a
    compressor restored null and `compress_win` wrote through it (illegal access, surfaced as
    `cublasGemmEx status 13`). `RpRep` now keeps the pointers itself and `load_replica` re-attaches them only
    when the source layer is a KV source.
12. **Tail pass crash 1: an index into the wrong list.** `tail_sync_src` indexed `srcs[g.cand_src]` (20) into
    the compact 4-entry source list (layers 2, 8, 14, 20) - an out-of-bounds read that produced a garbage
    `Src` and crashed inside `halloc`. Fixed by looking the entry up by `.layer`.
13. **Tail pass crash 2: a default-constructed run.** Pass 1 builds one run per group, and the encoder-only
    groups' runs are empty (`T` empty, `stage = 0`). The flow branch read `run.T[0]` -> a null-vector read,
    three worker threads at the same ip, fault at address 0. Fixed with `if (run.T.empty()) return true;` at the
    top of `run_dev_chunks`; `pass_devs` now counts only devices with a layer in the pass. Both crashes appeared
    only at the *last* chunk of a prompt, because pass 1 starts after the whole prompt's pass 0.
14. **The reserve is eaten by the driver at startup (wider engine finding, applies here).** The pool is sized
    exactly as free - reserve, and then startup takes ~285 MiB a card (in 20-45 MiB steps, then stops) for the
    CUDA context, lazily loaded fatbin and the driver - so the reserve must be ~285 MiB plus the free VRAM you
    actually want. The live `2,1,1,2` is a deliberate trade: decode's expert slots get the rest.
15. **The first prompt of a session is slow on every version.** It carries every kernel's first launch, the
    prompt path's first allocations and the tiers' first reads; no later prompt pays it. Compare like-for-like
    (second prompt onwards) before calling a version slower.

---

## 5. How the paths compose

`Model::feed` routes a prompt in this order:

1. **Row-split** if `--row-split-prefill` is on, the prompt is at least `DS4_ROW_SPLIT_MIN` rows, the logits are
   window-sized, and `rp_ready` finds a plan and a flow on every device. Any prompt size, in rounds.
2. **Chunked path** if `--prefill-chunk > 0` and the prompt is at least `chunk_min`; inside it, `pick_path`
   chooses between the flow (chunked pipeline, streamed picks) and 16-token windows.
   - **Decoder tail** (`DS4_TAIL_PREFILL=1`) engages when the prompt exceeds 2,560 rows and its exact gate holds.
   - **EP** (`--ep-prefill`) engages only in a single-chunk encoder pass with plain layers and no tail - i.e.
     prompts at most 2,560 rows on this split.
   - **MMQ** (`--chunk-mmq`) replaces the expert GEMMs of every all-MXFP4 piece, independently of the other two.
3. **Windows of 16** otherwise.

So: row-split and the chunked path are alternatives; within the chunked path tail and EP are mutually exclusive
by design (an EP partner would wait for a layer the owner never runs in pass 1); MMQ, telemetry, placement and
chunk sizing are independent of all of them.

---

## 6. Open items

- The flag-off byte check (`tools/ds4/bytecheck.sh windows chunked` against the saved pre-change binary) for the
  row-split/EP/MMQ tree is still unrun; it is the plan's remaining gate.
- Row-split at 16k with the 5-group geometry (`DS4_ROW_SPLIT_FRAC=1.0`) is measured only on 512 rows; the
  14 -> 5 group A/B would say what the wavefront overhead is worth.
- EP with lend/borrow resident slots in its shares, and finer-than-batch reads for its picks.
- MMQ per-role generalization (llama.cpp has IQ2/IQ3/Q3_K MMQ configs; each plane offset and the ring pitch would
  have to be a whole number of that type's blocks) or an all-MXFP4 pack.
- Moving the row-split selector off the wavefront's critical path (the ckv finish before `if (l.ix_src)`).
- Single-device row-split (Phase B of the plan) and the decode-slot cost of the 5-group geometry.

---

## 7. Files and commits

| Commit | What |
|---|---|
| `d092e7bd` | Expert lending from the prompt arenas, mixed-role expert types, nested prompt chunks |
| `41f0b424` | UD-IQ3_XXS pack support (IQ2_XS experts, Q6_K engram_kv and Engram rows) |
| `ce774f8a` | The Monitor, sandal-white theme, NVML indices from the LEND PCI addresses |
| `2df50dd9` | v1.1.17's layer-major path on the chunked runner's signatures |
| `1e3edd0a` | The Monitor's colour keys and pipeline row order |
| `cd0aff66` | `--ram-resident` |
| `6fd82c9a` | The decoder tail and the LIVE Monitor lines - 576 tok/s |
| `95cbd953` | The arena takes only what the prompt needs (-39% refill) |
| this work | `--ep-prefill`, `--row-split-prefill`, `--chunk-mmq`, the MMQ build, the wavefront Monitor, the tests, this page |

The unchanged pieces this work relies on: the decode path, the KV-cache format and layout, the model files, the
server protocol, and every pre-existing flag and config key (`--expert-profile`, `--no-expert-cache`,
`--vram-experts`, `--cpu-threads`, `--prefill-chunk`, `--tp`, `--threads`, `--dspark*`, `--vision`,
`--mmap-experts`, `DS4_SPLIT`, `DS4_FLOW`, `DS4_FLOW_STATS`, `DS4_PROF`, `DS4_SPEC_Q`, `DS4_READAHEAD_MB`,
`DS4_RING_MB`, `DS4_READERS`).
