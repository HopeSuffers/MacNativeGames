/* Included only by nvn_mtl.m (before mtl_sub_split). TOTK_OVERLAP=1 (experiment, off by default): the game's CPU frame
   N+1 overlaps GPU frame N.

   The game records frame N+1, waits for frame N's fence (nvnSyncWait), then submits N+1. With the wait released only when
   the GPU has finished N, the GPU idles from the end of N until N+1's first command buffer arrives (wake-up, game code,
   submit call, first chunk translated: ~2.2 ms a frame at Lookout Landing, plus ~1.8 ms of translation starvation later).
   Here the wait for a submission's fence is satisfied once every command buffer of it but the last ("the tail") has
   completed and the tail is committed (g_early_done); everything the CPU could see differently is made to look as if
   the submission had completed (TOTK_OVERLAP=2 is a control: all of the below except the early release itself):
   - CPU-visible results posted by the submission (event signals, counter reports, occlusion query sums) are written at
     the release, and the replay's pending-counter records of the fence are retired then (cond_eval reads memory). A
     query reported inside the tail cannot be known yet: such a submission is not released early.
   - Guest memory the tail still reads or writes is held from the CPU until the GPU is done (nvn_dirty.c page holds):
     PROT_READ over what it reads (vertex / index / storage / live uniform buffers, sources of copies) so CPU writes wait,
     PROT_NONE over what it writes (storage buffers, copy targets) so every CPU access waits; the end-of-submission uniform
     write-back (upd_flush) gets PROT_READ. TOTK_OVERLAP_HOLD=0 drops the holds (unsafe, for measuring their cost only).
   - Small storage buffers that compute shaders write in the tail at a stable address (the depth statistics the game
     copies on the CPU right after its frame wait: cs 4c8f872d -> cd68063a -> 65a96cf0, read by main+0xcc1d7c) are
     redirected ("snapshots", TOTK_OVERLAP_SNAP=1, default on): every GPU binding of the range uses a host buffer H, so
     the GPU timeline (including 65a96cf0's temporal smoothing that reads its own previous output) is unchanged; after
     each submission H is copied to a ring slot, and when a frame wait returns (or the game invalidates the pool range)
     the newest completed slot is written into guest memory. A frame released early therefore hands the game the
     statistics of the frame before (they were already one frame old by design). CPU writes into the range are merged
     into H at the next submission; a binding or copy that overlaps the range only partly retires the snapshot.
   - The game's next recording (nvnCommandBufferBeginRecording, TOTK_OVERLAP_GATE=1 default) and every operation that
     destroys or reads back Metal state (x_drain) first wait until the early-released submission has really completed.
   The tail is the last chunk of the ordinary mid-submission commits (TOTK_CHUNK_DRAWS / TOTK_CHUNK_FIRST), or with
   TOTK_OVERLAP_TAIL=<draws> the last <draws> draws, predicted from the previous submission with the same handle count
   (the ordinary chunking stops at that split). TOTK_OVERLAP_LOG=1 logs the first releases in detail. */

#define OVL_MAX_HOLDS 192   // merged hold ranges per submission (nvn_dirty.c has 512 slots, about two frames' worth)
#define OVL_SLOTS 64
#define OVL_SNAPS 8
#define OVL_SNAP_MAX (64u << 10)
#define OVL_RING 4
static int g_ovl_tail, g_ovl_hold = 1, g_ovl_snap = 1, g_ovl_gate = 1, g_ovl_log, g_snap_log;
typedef struct { uint64_t lo, hi; int prot; } ovl_rng;
/* The command buffer being recorded (reset at each mid-submission commit: at the commit of the submission it is the tail). */
static struct {
    ovl_rng *r; int n, cap, over;
    uint64_t cand_lo[16], cand_hi[16]; int ncand;   // storage ranges compute shaders write in it (snapshot candidates)
    int npost0;                                      // g_npost when it began
    int tail_started, key;                           // TOTK_OVERLAP_TAIL split done; handle count of the submission
    uint64_t draws0;                                 // g_stat_draw when the submission began
} g_ot;
static struct { int key; uint64_t draws; } g_ovl_pred[8];
typedef struct { uint64_t fence; int total, done, committed, state, released, nops, nholds; post_op *ops; uint64_t gpu_f, t_rel; ovl_rng *holds; } ovl_slot;
static ovl_slot g_ovs[OVL_SLOTS];   // under g_mu. state: 0 none, 1 parked (eligible), 2 being released, 3 posted results applied, 4 taken by the completion handler
static struct {
    uint64_t subs, eligible, released, why[5], tail_draws, tail_n, hold_ranges, hold_fail;
    uint64_t lead_ns, lead_n, lead_max, gate_n, gate_ns;
    uint64_t snap_pub, snap_merge, snap_conflict, snap_skip, snap_redirects, snap_activated;
} g_ovs_stat;
static void post_apply(post_op *ops, int n, uint64_t fence, uint64_t gpu_f);
static uint64_t mtl_done_now(void) { return __atomic_load_n(&g_done, __ATOMIC_ACQUIRE); }
static uint64_t ovl_now(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }

static void ovl_init(void) {   // replay thread, first submission
    if (g_ovl >= 0) return;
    const char *e = getenv("TOTK_OVERLAP"); g_ovl = e ? atoi(e) : 0;
    if (g_ovl != 1 && g_ovl != 2) { g_ovl = 0; return; }
    e = getenv("TOTK_OVERLAP_TAIL"); g_ovl_tail = e ? atoi(e) : 0;
    e = getenv("TOTK_OVERLAP_HOLD"); g_ovl_hold = !(e && !strcmp(e, "0"));
    e = getenv("TOTK_OVERLAP_SNAP"); g_ovl_snap = !(e && !strcmp(e, "0"));
    e = getenv("TOTK_OVERLAP_GATE"); g_ovl_gate = !(e && !strcmp(e, "0"));
    e = getenv("TOTK_OVERLAP_LOG"); g_ovl_log = e && atoi(e) > 0 ? 40 : 0;
    e = getenv("TOTK_OVERLAP_SNAP_LOG"); g_snap_log = e && atoi(e) > 0;
    nvn_hold_set_done(mtl_done_now);
    if (g_ovl == 2) MLOG("TOTK_OVERLAP=2 (control): snapshots and bookkeeping as with 1, but no frame wait ends early");
    MLOG("TOTK_OVERLAP: frame waits end when only the last command buffer of a submission still runs (tail %s%s, holds %s, snapshots %s, gate %s; chunks %s)",
         g_ovl_tail > 0 ? "= last " : "= last ordinary chunk", g_ovl_tail > 0 ? [NSString stringWithFormat:@"%d draws", g_ovl_tail].UTF8String : "",
         g_ovl_hold ? "on" : "OFF (unsafe)", g_ovl_snap ? "on" : "off", g_ovl_gate ? "on" : "off", g_chunk_size ? "on" : "OFF: nothing is released early");
}
int mtl_overlap_gate_enabled(void) { return g_ovl > 0 && g_ovl_gate; }
int mtl_overlap_gate_needed(void) { return g_ovl > 0 && __atomic_load_n(&g_done, __ATOMIC_RELAXED) < __atomic_load_n(&g_early_done, __ATOMIC_RELAXED); }

// ---------- what the current command buffer touches ----------
static void ovl_note(uint64_t a, uint64_t n, int prot) {   // replay thread
    if (!a || !n) return;
    if (n > (1u << 20)) { uintptr_t plo, phi; if (mtl_addr_range(a, 1, &plo, &phi) && a + n > phi) n = phi - a; }   // a binding past its pool's end (whole-pool sizes)
    uint64_t z = a + n;
    if (g_ot.n) { ovl_rng *l = &g_ot.r[g_ot.n - 1]; if (l->prot == prot && a <= l->hi && z >= l->lo) { if (a < l->lo) l->lo = a; if (z > l->hi) l->hi = z; return; } }
    if (g_ot.n == g_ot.cap) {
        if (g_ot.cap >= 16384) { g_ot.over = 1; return; }
        g_ot.cap = g_ot.cap ? g_ot.cap * 2 : 256; g_ot.r = realloc(g_ot.r, sizeof *g_ot.r * (size_t)g_ot.cap); if (!g_ot.r) MDIE("overlap ranges");
    }
    g_ot.r[g_ot.n++] = (ovl_rng){a, z, prot};
}
static void ovl_seg_reset(void) { g_ot.n = 0; g_ot.over = 0; g_ot.ncand = 0; g_ot.npost0 = g_npost; }
static int rng_cmp(const void *x, const void *y) { const ovl_rng *a = x, *b = y; return a->prot != b->prot ? (a->prot < b->prot ? -1 : 1) : a->lo < b->lo ? -1 : a->lo > b->lo; }
/* The tail's ranges in host pages, overlapping / touching ones of the same protection merged. Ranges of different
   protections may overlap: the page holds take the strictest one per page. */
static int ovl_merge(ovl_rng **out) {
    *out = NULL; if (!g_ot.n) return 0;
    uint64_t pg = (uint64_t)getpagesize();
    ovl_rng *v = malloc(sizeof *v * (size_t)g_ot.n); if (!v) MDIE("overlap ranges");
    for (int i = 0; i < g_ot.n; i++) v[i] = (ovl_rng){g_ot.r[i].lo & ~(pg - 1), (g_ot.r[i].hi + pg - 1) & ~(pg - 1), g_ot.r[i].prot};
    qsort(v, (size_t)g_ot.n, sizeof *v, rng_cmp);
    int k = 0;
    for (int i = 1; i < g_ot.n; i++) {
        if (v[i].prot == v[k].prot && v[i].lo <= v[k].hi) { if (v[i].hi > v[k].hi) v[k].hi = v[i].hi; }
        else v[++k] = v[i];
    }
    *out = v; return k + 1;
}

// ---------- tail split (TOTK_OVERLAP_TAIL) ----------
static uint64_t ovl_pred_get(int key) { for (int i = 0; i < 8; i++) if (g_ovl_pred[i].key == key) return g_ovl_pred[i].draws; return 0; }
static void ovl_pred_put(int key, uint64_t draws) {
    for (int i = 0; i < 8; i++) if (g_ovl_pred[i].key == key) { g_ovl_pred[i].draws = draws; return; }
    static int next; g_ovl_pred[next++ & 7] = (typeof(g_ovl_pred[0])){key, draws};
}
/* chunk_maybe_split: -1 no split (the tail runs to the end), 1 split now (the tail starts here), 0 the ordinary rule */
static int ovl_split_rule(void) {
    if (g_ovl <= 0 || g_ovl_tail <= 0) return 0;
    if (g_ot.tail_started) return -1;
    uint64_t pred = ovl_pred_get(g_ot.key), done = g_stat_draw - g_ot.draws0;
    if (!pred || !done || done + (uint64_t)g_ovl_tail < pred) return 0;
    g_ot.tail_started = 1; return 1;
}

// ---------- release ----------
static ovl_slot *ovs_get(uint64_t fence) {   // under g_mu
    ovl_slot *s = &g_ovs[fence % OVL_SLOTS];
    if (s->fence != fence) { free(s->holds); *s = (ovl_slot){.fence = fence, .total = -1}; }
    return s;
}
static void ovl_try_release(uint64_t fence) {   // translation thread (commit) or feedback queue (a chunk completed)
    pthread_mutex_lock(&g_mu);
    ovl_slot *s = &g_ovs[fence % OVL_SLOTS];
    int go = s->fence == fence && s->state == 1 && s->committed && s->total > 0 && s->done >= s->total;
    post_op *ops = NULL; int n = 0, nh = 0; uint64_t gf = 0; ovl_rng *h = NULL;
    if (go) { s->state = 2; ops = s->ops; n = s->nops; gf = s->gpu_f; h = s->holds; nh = s->nholds; s->ops = NULL; }
    pthread_mutex_unlock(&g_mu);
    if (!go) return;
    post_apply(ops, n, fence, gf);   // frees ops
    /* the counter records are in memory now: the replay of the next submission must read memory, not the values still
       pending for this fence (the game re-arms its conditional passes by storing 1 over a type-16 report right after
       its frame wait; with the pending 0 the next frame's pass would be skipped) */
    cpend_done(fence);
    int fail = 0;
    for (int i = 0; i < nh; i++) if (!nvn_hold_begin((const void *)(uintptr_t)h[i].lo, (size_t)(h[i].hi - h[i].lo), fence, h[i].prot)) fail++;
    pthread_mutex_lock(&g_mu);
    s->state = 3;
    if (!fail) { s->released = 1; s->t_rel = ovl_now(); if (fence > g_early_done) g_early_done = fence; g_ovs_stat.released++; }
    else g_ovs_stat.hold_fail++;
    pthread_cond_broadcast(&g_cv);
    pthread_mutex_unlock(&g_mu);
    if (!fail) totk_tl(TL_EARLY, fence, (uint64_t)nh);
    if (g_ovl_log > 0) { g_ovl_log--; MLOG("[overlap] fence %llu released early%s: %d chunks done, %d posted results applied, %d hold ranges", (unsigned long long)fence, fail ? " FAILED (hold slots full)" : "", s->total, n, nh); }
}
static void ovl_chunk_done(uint64_t fence) {   // feedback queue: a mid-submission command buffer of that submission completed
    pthread_mutex_lock(&g_mu); ovs_get(fence)->done++; pthread_mutex_unlock(&g_mu);
    ovl_try_release(fence);
}
/* mtl_sub_commit, before the tail is committed: may this submission be released early? 1 = its slot is parked. */
static int ovl_prepare(uint64_t fence, post_op *ops, int n, uint64_t gpu_f) {
    int why = 0; ovl_rng *h = NULL; int nh = 0;
    g_ovs_stat.subs++;
    ovl_pred_put(g_ot.key, g_stat_draw - g_ot.draws0);
    if (g_ovl == 2) why = 4;   // control mode: never released early
    else if (!g_sub_chunks) why = 1;   // not split: nothing completes before the end
    else {
        g_ovs_stat.tail_draws += g_stat_draw - g_chunk_mark; g_ovs_stat.tail_n++;
        for (int i = g_ot.npost0; i < n; i++) if (ops[i].kind == 3) { why = 2; break; }   // an occlusion query in the tail
    }
    if (!why && g_ovl_hold) {
        if (g_ot.over) why = 3;
        else if ((nh = ovl_merge(&h)) > OVL_MAX_HOLDS) { why = 3; free(h); h = NULL; nh = 0; }
    }
    if (why) { g_ovs_stat.why[why]++; free(h); return 0; }
    g_ovs_stat.eligible++; g_ovs_stat.hold_ranges += (uint64_t)nh;
    pthread_mutex_lock(&g_mu);
    ovl_slot *s = ovs_get(fence);
    s->total = g_sub_chunks; s->state = 1; s->ops = ops; s->nops = n; s->gpu_f = gpu_f; s->holds = h; s->nholds = nh; s->committed = 0;
    pthread_mutex_unlock(&g_mu);
    return 1;
}
static void ovl_committed(uint64_t fence) {   // the tail is committed
    pthread_mutex_lock(&g_mu); ovs_get(fence)->committed = 1; pthread_mutex_unlock(&g_mu);
    ovl_try_release(fence);
}
/* Completion handler of the tail: 1 = the posted results were already applied at the release (and freed). */
static int ovl_final_take(uint64_t fence) {
    int applied = 0;
    pthread_mutex_lock(&g_mu);
    ovl_slot *s = &g_ovs[fence % OVL_SLOTS];
    if (s->fence == fence) {
        if (s->state == 1) { s->state = 4; s->ops = NULL; }
        while (s->state == 2) pthread_cond_wait(&g_cv, &g_mu);
        applied = s->state == 3;
    }
    pthread_mutex_unlock(&g_mu);
    return applied;
}
static void ovl_completed(uint64_t fence) {   // completion handler, after g_done moved to fence
    nvn_hold_end(fence);
    pthread_mutex_lock(&g_mu);
    ovl_slot *s = &g_ovs[fence % OVL_SLOTS];
    if (s->fence == fence) {
        if (s->released && s->t_rel) { uint64_t d = ovl_now() - s->t_rel; g_ovs_stat.lead_ns += d; g_ovs_stat.lead_n++; if (d > g_ovs_stat.lead_max) g_ovs_stat.lead_max = d; }
        free(s->holds); s->holds = NULL; s->nholds = 0;
    }
    pthread_mutex_unlock(&g_mu);
}
/* The game's frame wait (nvnSyncWait): satisfied by a real completion or an early release of that fence or a later one. */
int mtl_wait_frame(uint64_t fence, uint64_t timeout_ns) {
    if (g_ovl <= 0) return mtl_wait(fence, timeout_ns);
    struct timespec dl; clock_gettime(CLOCK_REALTIME, &dl);
    uint64_t ns = (uint64_t)dl.tv_nsec + (timeout_ns > 3600000000000ull ? 3600000000000ull : timeout_ns);
    dl.tv_sec += ns / 1000000000ull; dl.tv_nsec = ns % 1000000000ull;
    pthread_mutex_lock(&g_mu);
    while (g_done < fence && g_early_done < fence) if (pthread_cond_timedwait(&g_cv, &g_mu, &dl)) break;
    int ok = g_done >= fence || g_early_done >= fence;
    pthread_mutex_unlock(&g_mu);
    return ok;
}
/* Before the game builds its next frame / before Metal state is destroyed or read back: an early-released submission has
   really completed. */
void mtl_overlap_gate(void) {
    if (g_ovl <= 0) return;
    pthread_mutex_lock(&g_mu);
    if (g_done < g_early_done) {
        uint64_t t0 = ovl_now(); totk_tl(TL_GATE_IN, g_early_done, 0);
        while (g_done < g_early_done) pthread_cond_wait(&g_cv, &g_mu);
        g_ovs_stat.gate_n++; g_ovs_stat.gate_ns += ovl_now() - t0; totk_tl(TL_GATE_OUT, g_done, 0);
    }
    pthread_mutex_unlock(&g_mu);
}

// ---------- snapshots of small storage buffers written in the tail ----------
typedef struct {
    int state;   // 0 free, 1 candidate, 2 active, 3 retiring (whole bindings still use H; at the next submission H goes back to memory), 4 dead (buffers kept until dead_fence)
    int hits, used_rw, base_valid, gone;
    uint64_t lo, hi, last_sub, base_fence, pub_fence, dead_fence;
    __strong id<MTLBuffer> H, S[OVL_RING]; uint64_t s_fence[OVL_RING];
    uint64_t *base, *mval; uint8_t *mmask;   // guest words at the last sync; CPU words still to merge into H
} ovl_snap;
static ovl_snap g_snap[OVL_SNAPS];
static os_unfair_lock g_snap_lk = OS_UNFAIR_LOCK_INIT;   // publish (game thread) / pool finalize vs the replay thread
static _Atomic int g_snap_live;   // entries in state 2 or 3
static uint64_t g_ovl_sub_serial;
static void ovl_snap_conflict(ovl_snap *s, const char *what, uint64_t a, uint64_t n) {   // replay thread
    os_unfair_lock_lock(&g_snap_lk); if (s->state == 2) s->state = 3; os_unfair_lock_unlock(&g_snap_lk);
    if (++g_ovs_stat.snap_conflict <= 8) MLOG("【红】[overlap] snapshot %#llx+%llu retired: %s %#llx+%llu overlaps it only partly (this access uses memory)",
                                              (unsigned long long)s->lo, (unsigned long long)(s->hi - s->lo), what, (unsigned long long)a, (unsigned long long)n);
}
/* A GPU binding of [a, a+n): inside a live snapshot it uses H (1, *buf / *gpu set); partly over one retires it. */
static int ovl_snap_redirect(uint64_t a, uint64_t n, int rw, void **buf, uint64_t *gpu) {
    if (!atomic_load_explicit(&g_snap_live, memory_order_relaxed) || !a) return 0;
    if (!n) n = 1;
    for (int i = 0; i < OVL_SNAPS; i++) {
        ovl_snap *s = &g_snap[i];
        if ((s->state != 2 && s->state != 3) || a + n <= s->lo || a >= s->hi) continue;
        if (a >= s->lo && a + n <= s->hi) { *buf = (__bridge void *)s->H; *gpu = s->H.gpuAddress + (a - s->lo); if (rw) s->used_rw = 1; g_ovs_stat.snap_redirects++; return 1; }
        ovl_snap_conflict(s, "binding", a, n);
    }
    return 0;
}
static void ovl_snap_touch(uint64_t a, uint64_t n, const char *what) {   // a copy / update / built-in kernel reaches [a, a+n) through memory
    if (!atomic_load_explicit(&g_snap_live, memory_order_relaxed) || !a || !n) return;
    for (int i = 0; i < OVL_SNAPS; i++) { ovl_snap *s = &g_snap[i]; if ((s->state == 2 || s->state == 3) && a < s->hi && a + n > s->lo) ovl_snap_conflict(s, what, a, n); }
}
static void ovl_cand(uint64_t a, uint64_t n) {   // a compute shader writes [a, a+n) in the current command buffer
    if (!g_ovl_snap || !a || !n || n > OVL_SNAP_MAX || g_ot.ncand >= 16) return;
    for (int i = 0; i < g_ot.ncand; i++) if (g_ot.cand_lo[i] == a && g_ot.cand_hi[i] == a + n) return;
    g_ot.cand_lo[g_ot.ncand] = a; g_ot.cand_hi[g_ot.ncand++] = a + n;
}
static void ovl_snap_free(ovl_snap *s) {   // replay thread; buffers no longer used by the GPU
    if (s->H) res_remove(s->H);
    for (int k = 0; k < OVL_RING; k++) if (s->S[k]) res_remove(s->S[k]);
    free(s->base); free(s->mval); free(s->mmask);
    *s = (ovl_snap){0};
}
/* mtl_sub_commit: the tail's compute-written storage ranges become candidates; seen in two tails within 8 submissions
   they are activated at the next submission (ovl_snap_begin). */
static void ovl_snap_candidates(void) {
    g_ovl_sub_serial++;
    for (int c = 0; c < g_ot.ncand; c++) {
        uint64_t lo = g_ot.cand_lo[c], hi = g_ot.cand_hi[c]; ovl_snap *f = NULL, *s = NULL;
        for (int i = 0; i < OVL_SNAPS && !s; i++) if (g_snap[i].state == 1 && g_snap[i].lo == lo && g_snap[i].hi == hi) s = &g_snap[i];
        for (int i = 0; i < OVL_SNAPS && !s && !f; i++) if (g_snap[i].state == 0) f = &g_snap[i];
        if (!s && !f) continue;
        if (!s) { s = f; *s = (ovl_snap){0}; s->state = 1; s->lo = lo; s->hi = hi; }
        s->hits = s->last_sub && g_ovl_sub_serial - s->last_sub <= 8 ? s->hits + 1 : 1; s->last_sub = g_ovl_sub_serial;   // other submissions may come between two frames' tails
    }
}
static void ovl_snap_merge_into_h(ovl_snap *s) {   // replay thread: CPU words written since the last sync go into H (GPU timeline)
    size_t nw = (size_t)(s->hi - s->lo) / 8; const volatile uint64_t *g = (const volatile uint64_t *)(uintptr_t)s->lo;
    os_unfair_lock_lock(&g_snap_lk);
    if (!s->base_valid && mtl_done_now() >= s->base_fence) { for (size_t w = 0; w < nw; w++) s->base[w] = g[w]; s->base_valid = 1; }   // no direct GPU write of the range is in flight any more
    if (s->base_valid) for (size_t w = 0; w < nw; w++) { uint64_t v = g[w]; if (v != s->base[w]) { s->mval[w] = v; s->mmask[w] = 1; s->base[w] = v; } }
    size_t w = 0; int runs = 0;
    while (w < nw) {
        if (!s->mmask[w]) { w++; continue; }
        size_t w0 = w; while (w < nw && s->mmask[w]) { s->mmask[w] = 0; w++; }
        id<MTLBuffer> sb; size_t so = arena_alloc((w - w0) * 8, &sb);
        memcpy((uint8_t *)sb.contents + so, &s->mval[w0], (w - w0) * 8);
        id<MTL4ComputeCommandEncoder> b = blit(); enc_b_copy_buffer(b, sb, so, s->H, w0 * 8, (w - w0) * 8); runs++;
    }
    os_unfair_lock_unlock(&g_snap_lk);
    if (runs) { enc_b_end(g_blit); g_blit = nil; g_ovs_stat.snap_merge += (uint64_t)runs; }
}
/* mtl_sub_begin (replay thread, command buffer open, nothing encoded yet): activate, merge, retire, free. */
static void ovl_snap_begin(void) {
    for (int i = 0; i < OVL_SNAPS; i++) {
        ovl_snap *s = &g_snap[i];
        if (s->state == 4 && mtl_done_now() >= s->dead_fence) ovl_snap_free(s);
        else if (s->state == 2) ovl_snap_merge_into_h(s);
        else if (s->state == 3) {   // retire: H (with the CPU's words) back to memory, then plain bindings again
            if (!s->gone) ovl_snap_merge_into_h(s);   // a finalized pool's memory is not read any more
            void *mb; uint64_t mo;
            if (!s->gone && mtl_addr(s->lo, s->hi - s->lo, &mb, &mo)) { id<MTL4ComputeCommandEncoder> b = blit(); enc_b_copy_buffer(b, s->H, 0, (__bridge id<MTLBuffer>)mb, mo, s->hi - s->lo); enc_b_end(g_blit); g_blit = nil; }
            os_unfair_lock_lock(&g_snap_lk); s->state = 4; s->dead_fence = g_fence + 1; os_unfair_lock_unlock(&g_snap_lk);
            atomic_fetch_sub(&g_snap_live, 1);
        } else if (s->state == 1) {
            if (g_ovl_sub_serial - s->last_sub > 8) { s->state = 0; continue; }   // not in a recent tail: stale
            if (s->hits < 2) continue;
            uint64_t lo = s->lo & ~15ull, hi = (s->hi + 15) & ~15ull;   // union with the overlapping candidates (one shader binds a part of another's range)
            for (int j = 0; j < OVL_SNAPS; j++) { ovl_snap *o = &g_snap[j]; if (o != s && o->state == 1 && o->lo < hi && o->hi > lo) { if (o->lo < lo) lo = o->lo & ~15ull; if (o->hi > hi) hi = (o->hi + 15) & ~15ull; } }
            int clash = 0; for (int j = 0; j < OVL_SNAPS; j++) { ovl_snap *o = &g_snap[j]; if (o->state >= 2 && o->state <= 4 && o->lo < hi && o->hi > lo) clash = 1; }
            void *mb; uint64_t mo;
            if (clash || hi - lo > OVL_SNAP_MAX || !mtl_addr(lo, hi - lo, &mb, &mo)) { s->state = 0; continue; }
            for (int j = 0; j < OVL_SNAPS; j++) { ovl_snap *o = &g_snap[j]; if (o != s && o->state == 1 && o->lo < hi && o->hi > lo) *o = (ovl_snap){0}; }
            size_t len = (size_t)(hi - lo), nw = len / 8;
            s->lo = lo; s->hi = hi;
            s->H = [g_dev newBufferWithLength:len options:MTLResourceStorageModeShared]; res_add(s->H);
            for (int k = 0; k < OVL_RING; k++) { s->S[k] = [g_dev newBufferWithLength:len options:MTLResourceStorageModeShared]; res_add(s->S[k]); s->s_fence[k] = 0; }
            s->base = calloc(nw, 8); s->mval = calloc(nw, 8); s->mmask = calloc(nw, 1);
            if (!s->H || !s->S[OVL_RING - 1] || !s->base || !s->mval || !s->mmask) MDIE("overlap snapshot buffers");
            id<MTL4ComputeCommandEncoder> b = blit(); enc_b_copy_buffer(b, (__bridge id<MTLBuffer>)mb, mo, s->H, 0, len); enc_b_end(g_blit); g_blit = nil;   // after every earlier GPU write of it
            s->base_valid = 0; s->base_fence = g_fence; s->pub_fence = g_fence; s->used_rw = 0;   // bindings up to g_fence wrote memory directly
            os_unfair_lock_lock(&g_snap_lk); s->state = 2; os_unfair_lock_unlock(&g_snap_lk);
            atomic_fetch_add(&g_snap_live, 1); g_ovs_stat.snap_activated++;
            MLOG("[overlap] snapshot %#llx+%zu active: GPU bindings use a host buffer, frame waits hand the game the newest completed copy", (unsigned long long)lo, len);
        }
    }
}
/* mtl_sub_commit (replay thread, before the tail is ended): H of every snapshot this submission wrote goes to a ring slot
   that is neither in flight nor the newest completed one (that one may be being published). */
static void ovl_snap_commit(uint64_t fence) {
    if (!atomic_load_explicit(&g_snap_live, memory_order_relaxed)) return;
    for (int i = 0; i < OVL_SNAPS; i++) {
        ovl_snap *s = &g_snap[i];
        if (s->state != 2 || !s->used_rw) continue;
        s->used_rw = 0;
        os_unfair_lock_lock(&g_snap_lk);
        uint64_t done = mtl_done_now(); int newest = -1, pick = -1;
        for (int k = 0; k < OVL_RING; k++) if (s->s_fence[k] && s->s_fence[k] <= done && (newest < 0 || s->s_fence[k] > s->s_fence[newest])) newest = k;
        for (int k = 0; k < OVL_RING; k++) if (k != newest && s->s_fence[k] <= done && (pick < 0 || s->s_fence[k] < s->s_fence[pick])) pick = k;
        if (pick >= 0) s->s_fence[pick] = fence;
        os_unfair_lock_unlock(&g_snap_lk);
        if (pick < 0) { g_ovs_stat.snap_skip++; continue; }
        id<MTL4ComputeCommandEncoder> b = blit(); enc_b_copy_buffer(b, s->H, 0, s->S[pick], 0, s->hi - s->lo); enc_b_end(g_blit); g_blit = nil;
    }
}
/* Game thread, when a frame wait returns or the game invalidates a pool range: the newest completed copy of each snapshot
   goes into guest memory. Words the CPU wrote since the last sync are kept (and merged into H at the next submission). */
void mtl_overlap_publish(void) {
    if (!atomic_load_explicit(&g_snap_live, memory_order_relaxed)) return;
    uint64_t done = mtl_done_now();
    os_unfair_lock_lock(&g_snap_lk);
    for (int i = 0; i < OVL_SNAPS; i++) {
        ovl_snap *s = &g_snap[i];
        if (s->state != 2) continue;
        int k = -1;
        for (int j = 0; j < OVL_RING; j++) if (s->s_fence[j] > s->pub_fence && s->s_fence[j] <= done && (k < 0 || s->s_fence[j] > s->s_fence[k])) k = j;
        if (k < 0) continue;
        size_t len = (size_t)(s->hi - s->lo), nw = len / 8; volatile uint64_t *g = (volatile uint64_t *)(uintptr_t)s->lo;
        if (!s->base_valid) { if (done < s->base_fence) continue; for (size_t w = 0; w < nw; w++) s->base[w] = g[w]; s->base_valid = 1; }
        void *tok = nvn_dirty_tracked((const void *)(uintptr_t)s->lo, len) ? nvn_dirty_gpu_write_begin((void *)(uintptr_t)s->lo, len) : NULL;
        if (!tok) dm_guest_write_prepare((void *)(uintptr_t)s->lo, len);
        const uint64_t *src = (const uint64_t *)s->S[k].contents;
        for (size_t w = 0; w < nw; w++) {
            uint64_t cur = g[w];
            if (cur != s->base[w]) { s->mval[w] = cur; s->mmask[w] = 1; }   // the CPU wrote it: its word stays (and goes into H)
            else if (cur != src[w]) g[w] = src[w];
            s->base[w] = g[w];
        }
        nvn_dirty_gpu_write_end(tok);
        s->pub_fence = s->s_fence[k]; g_ovs_stat.snap_pub++;
        if (g_snap_log && g_ovs_stat.snap_pub % 60 == 1) {   // TOTK_OVERLAP_SNAP_LOG=1: what the game is handed (depth statistics: words 7300-7307 = bytes 0x7210..)
            size_t at = len >= 0x72a0 ? 0x7210 : 0; const float *f = (const float *)((const uint8_t *)src + at); const uint32_t *u = (const uint32_t *)((const uint8_t *)src + at);
            MLOG("[overlap] snapshot %#llx published fence %llu (completed %llu): +%#zx = %g %g %g %g %g %g %g %g (%08x %08x)%s", (unsigned long long)s->lo, (unsigned long long)s->pub_fence, (unsigned long long)done, at,
                 f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], u[0], u[1], len >= 0x7268 ? [NSString stringWithFormat:@"; flag byte +0x7264 = %#x", ((const uint8_t *)src)[0x7264]].UTF8String : "");
        }
    }
    os_unfair_lock_unlock(&g_snap_lk);
}
static void ovl_snap_pool_gone(uintptr_t lo, uintptr_t hi) {   // mtl_pool_del (the translation queue is drained)
    if (!atomic_load_explicit(&g_snap_live, memory_order_relaxed)) return;
    os_unfair_lock_lock(&g_snap_lk);
    for (int i = 0; i < OVL_SNAPS; i++) { ovl_snap *s = &g_snap[i]; if (s->state == 2 && s->lo < hi && s->hi > lo) { s->state = 3; s->gone = 1; } }
    os_unfair_lock_unlock(&g_snap_lk);
}

// ---------- submission hooks ----------
static void ovl_sub_begin(void) {   // end of mtl_sub_begin
    ovl_init();
    if (g_ovl <= 0) return;
    ovl_seg_reset(); g_ot.tail_started = 0; g_ot.draws0 = g_stat_draw; g_ot.key = 0;
    if (g_ovl_snap) ovl_snap_begin();
}
static void ovl_stats_line(void) {
    if (g_ovs_stat.subs % 600) return;
    uint64_t hv[6]; nvn_hold_stats(hv);
    MLOG("[overlap] %llu submissions: %llu released early (eligible %llu; not: unsplit %llu, query in the tail %llu, too many hold ranges %llu, control mode %llu; hold slots full %llu); tail %.1f draws, released %.2f ms (max %.2f) before completion; gate waits %llu, %.3f ms avg",
         (unsigned long long)g_ovs_stat.subs, (unsigned long long)g_ovs_stat.released, (unsigned long long)g_ovs_stat.eligible, (unsigned long long)g_ovs_stat.why[1], (unsigned long long)g_ovs_stat.why[2],
         (unsigned long long)g_ovs_stat.why[3], (unsigned long long)g_ovs_stat.why[4], (unsigned long long)g_ovs_stat.hold_fail, g_ovs_stat.tail_n ? (double)g_ovs_stat.tail_draws / (double)g_ovs_stat.tail_n : 0.0,
         g_ovs_stat.lead_n ? (double)g_ovs_stat.lead_ns / (double)g_ovs_stat.lead_n / 1e6 : 0.0, g_ovs_stat.lead_max / 1e6,
         (unsigned long long)g_ovs_stat.gate_n, g_ovs_stat.gate_n ? (double)g_ovs_stat.gate_ns / (double)g_ovs_stat.gate_n / 1e6 : 0.0);
    MLOG("[overlap] holds %llu (%.1f ranges per release, %llu pages), CPU accesses that waited %llu (timed out %llu), host writes that waited %llu; snapshots active %d (activated %llu), published %llu, CPU merges %llu, conflicts %llu, ring full %llu, GPU bindings redirected %llu",
         (unsigned long long)hv[0], g_ovs_stat.eligible ? (double)g_ovs_stat.hold_ranges / (double)g_ovs_stat.eligible : 0.0, (unsigned long long)hv[1], (unsigned long long)hv[2], (unsigned long long)hv[3], (unsigned long long)hv[4],
         atomic_load(&g_snap_live), (unsigned long long)g_ovs_stat.snap_activated, (unsigned long long)g_ovs_stat.snap_pub, (unsigned long long)g_ovs_stat.snap_merge, (unsigned long long)g_ovs_stat.snap_conflict,
         (unsigned long long)g_ovs_stat.snap_skip, (unsigned long long)g_ovs_stat.snap_redirects);
    g_ovs_stat.lead_max = 0;
}
