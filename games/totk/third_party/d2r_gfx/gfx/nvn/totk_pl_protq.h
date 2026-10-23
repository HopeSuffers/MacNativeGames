/* Included by totk_pipeline.h (and tests/pipeline_hold/probe.c). TOTK_PIPELINE: write protection of spans the replay
   decided on (static spans, probation runs, non-static in-place ranges) runs on a helper thread, not on the translate
   thread (the mprotect calls of new data made the translations of streaming frames slow). A release first makes sure
   every span queued up to its submission is done (pl_protq_sync).
   One producer (the translate thread), consumers: the helper and a release. Two buffers: the producer fills one, the
   helper swaps it for its own (capacities travel with their buffers) and protects outside the lock; a release takes what
   is still queued by copy. Needs: typedef pl_rng { uintptr_t lo, hi; }, PL_PROTQ_RUN(lo, hi), PL_PROTQ_DIE(msg). */
static pthread_mutex_t g_plpq_mu = PTHREAD_MUTEX_INITIALIZER; static pthread_cond_t g_plpq_cv = PTHREAD_COND_INITIALIZER;
static pl_rng *g_plpq; static int g_plpq_n, g_plpq_cap; static uint64_t g_plpq_seq, g_plpq_done, g_plpq_busy_to; static int g_plpq_busy, g_plpq_started;
static _Atomic uint64_t g_plpq_inline, g_plpq_spans;
static void pl_protq_run(const pl_rng *v, int n) { for (int i = 0; i < n; i++) PL_PROTQ_RUN(v[i].lo, v[i].hi); atomic_fetch_add(&g_plpq_spans, (uint64_t)n); }
static void *pl_protq_main(void *arg) {
    (void)arg; pthread_setname_np("totk-pl-protect"); pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
    pl_rng *own = NULL; int own_cap = 0;
    for (;;) {
        pthread_mutex_lock(&g_plpq_mu);
        while (!g_plpq_n) pthread_cond_wait(&g_plpq_cv, &g_plpq_mu);
        pl_rng *v = g_plpq; int n = g_plpq_n, v_cap = g_plpq_cap;
        g_plpq = own; g_plpq_cap = own_cap; g_plpq_n = 0;   // the producer goes on in the other buffer
        own = v; own_cap = v_cap;
        uint64_t to = g_plpq_seq; g_plpq_busy = 1; g_plpq_busy_to = to;
        pthread_mutex_unlock(&g_plpq_mu);
        pl_protq_run(v, n);
        pthread_mutex_lock(&g_plpq_mu); g_plpq_busy = 0; if (to > g_plpq_done) g_plpq_done = to; pthread_cond_broadcast(&g_plpq_cv); pthread_mutex_unlock(&g_plpq_mu);
    }
    return NULL;
}
static void pl_protq_push(uintptr_t lo, uintptr_t hi) {   // the producer thread only
    if (!g_plpq_started) { g_plpq_started = 1; pthread_t t; if (pthread_create(&t, NULL, pl_protq_main, NULL)) PL_PROTQ_DIE("pipeline protection thread"); pthread_detach(t); }
    pthread_mutex_lock(&g_plpq_mu);
    if (g_plpq_n == g_plpq_cap) {
        int cap = g_plpq_cap ? g_plpq_cap * 2 : 1024; pl_rng *b = realloc(g_plpq, sizeof *b * (size_t)cap);
        if (!b) PL_PROTQ_DIE("pipeline protection queue");
        g_plpq = b; g_plpq_cap = cap;
    }
    g_plpq[g_plpq_n++] = (pl_rng){lo, hi}; g_plpq_seq++;
    if (g_plpq_n == 1) pthread_cond_broadcast(&g_plpq_cv);
    pthread_mutex_unlock(&g_plpq_mu);
}
static void pl_protq_cb(uintptr_t lo, uintptr_t hi) { pl_protq_push(lo, hi); }
static uint64_t pl_protq_seq(void) { pthread_mutex_lock(&g_plpq_mu); uint64_t q = g_plpq_seq; pthread_mutex_unlock(&g_plpq_mu); return q; }
/* the spans queued up to seq are protected: what is still queued is done here, a batch the helper works on is waited for */
static void pl_protq_sync(uint64_t seq) {
    pthread_mutex_lock(&g_plpq_mu);
    while (g_plpq_done < seq) {
        if (g_plpq_n) {
            int n = g_plpq_n; uint64_t to = g_plpq_seq; pl_rng *v = malloc(sizeof *v * (size_t)n);
            if (!v) PL_PROTQ_DIE("pipeline protection queue");
            memcpy(v, g_plpq, sizeof *v * (size_t)n); g_plpq_n = 0;
            pthread_mutex_unlock(&g_plpq_mu);
            pl_protq_run(v, n); free(v); atomic_fetch_add(&g_plpq_inline, (uint64_t)n);
            pthread_mutex_lock(&g_plpq_mu);
            while (g_plpq_busy && g_plpq_busy_to < to) pthread_cond_wait(&g_plpq_cv, &g_plpq_mu);   // an older batch still in the helper
            if (to > g_plpq_done) g_plpq_done = to;
            pthread_cond_broadcast(&g_plpq_cv);
        } else pthread_cond_wait(&g_plpq_cv, &g_plpq_mu);   // the helper is protecting the last batch
    }
    pthread_mutex_unlock(&g_plpq_mu);
}
