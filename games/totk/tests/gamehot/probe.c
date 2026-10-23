/* CPU only: TOTK_GAMEHOT's host answer (runtime/gamehot_patch.h) against the game's own RSDB query executor.
   The 1.2.1 main ELF (argument, default extracted/v121/elf/main.elf) is mapped executable at its own layout (text and
   read-only data, so the executor's jump tables resolve) and main+0x7f3688 is called directly on mock tables whose
   virtuals behave like the actor-info table's (main+0x27fdd0c count, main+0x193efd8 index, main+0x193efec value: tag 8
   and the row's first word). Every query of the "row whose column 0 is <key>" shape (as main+0xb95b98 builds it) must
   give the same row from the host as from the game:
     sorted unique names (n = 0 .. 4000, keys present, absent, before the first, after the last, prefixes, extensions,
     the same pointer as the row's), sorted names with duplicates (the first one wins), each twice (search, then hint);
   an unsorted table and a table with a non-string value must be left to the game; hints overwritten with garbage must
   not change any answer; four threads at once; gh_query in modes 1 and 2. Ends with value reads and time per lookup
   for the game and for the host (hint hits) on a 12,000-row table of actor-like names. */
#include "../../runtime/runtime.h"
#include <libkern/OSCacheControl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <pthread.h>

_Noreturn void runtime_fail(const char *message) { fprintf(stderr, "FAIL: %s\n", message); exit(1); }
uint64_t runtime_now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec; }
void *runtime_hle_wrap_named(void *fn, const char *name) { (void)name; return fn; }

#include "../../runtime/gamehot_patch.h"

/* value(row, column): {tag = row+8, value = row+0} through x8, 16 bytes like main+0x193efec; counts reads */
uint64_t mock_reads;
__asm__(".text\n.p2align 2\n.globl _mock_value\n_mock_value:\n"
        " adrp x10, _mock_reads@PAGE\n ldr x11, [x10, _mock_reads@PAGEOFF]\n add x11, x11, #1\n str x11, [x10, _mock_reads@PAGEOFF]\n"
        " ldr w9, [x1, #8]\n str w9, [x8]\n ldr x9, [x1]\n str x9, [x8, #8]\n ret\n");
void mock_value(void);

typedef struct { int32_t n, pad; uint64_t *rows; void *table; uint32_t column, pad2; } MockIndex;
typedef struct { void **vt; uint32_t count, pad; MockIndex index; } MockTable;
typedef struct { const char *name; uint32_t tag, pad; } MockRow;

static uint32_t mock_count(MockTable *t) { return t->count; }
static void *mock_index(MockTable *t, uint64_t column) { if ((uint32_t)column) runtime_fail("index for another column"); return &t->index; }
static void mock_unexpected(void) { runtime_fail("unexpected virtual call"); }
static void *mock_vt[0x100 / 8];

static gh_query_fn game;   /* main+0x7f3688 in the mapped ELF */

static void *map_main(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s (argument: the 1.2.1 main.elf)\n", path); exit(2); }
    fseek(f, 0, SEEK_END); long size = ftell(f); fseek(f, 0, SEEK_SET);
    uint8_t *elf = malloc((size_t)size);
    if (!elf || fread(elf, 1, (size_t)size, f) != (size_t)size) runtime_fail("read ELF");
    fclose(f);
    uint64_t phoff; uint16_t phnum; memcpy(&phoff, elf + 0x20, 8); memcpy(&phnum, elf + 0x38, 2);
    /* text and the read-only data with the executor's jump tables (main+0x3877xxx), then zeros standing in for the patch
       area's tail (section 8 installs the veneer there) */
    uint64_t copied = 0x3880000, span = 0x3888000;
    uint8_t *base = mmap(NULL, span, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON | MAP_JIT, -1, 0);
    if (base == MAP_FAILED) runtime_fail("mmap MAP_JIT");
    pthread_jit_write_protect_np(0);
    for (unsigned i = 0; i < phnum; i++) {
        uint32_t type; uint64_t off, vaddr, filesz; const uint8_t *ph = elf + phoff + 56 * i;
        memcpy(&type, ph, 4); memcpy(&off, ph + 8, 8); memcpy(&vaddr, ph + 16, 8); memcpy(&filesz, ph + 32, 8);
        if (type != 1 || vaddr >= copied) continue;
        memcpy(base + vaddr, elf + off, vaddr + filesz > copied ? copied - vaddr : filesz);
    }
    pthread_jit_write_protect_np(1);
    sys_icache_invalidate(base, span);
    uint32_t w; memcpy(&w, base + 0x7f3688, 4);
    if (w != 0xd10203ff) runtime_fail("main+0x7f3688 is not the 1.2.1 executor (sub sp, sp, #0x80)");
    free(elf);
    return base;
}

/* The query main+0xb95b98 builds: column 0 equal to the string key, no forced index, not reversed, no predicates. */
typedef struct { _Alignas(16) uint8_t b[0x100]; } Query;
static void make_query(Query *q, MockTable *t, const char *key) {
    memset(q, 0, sizeof *q);
    void *tp = t; memcpy(q->b + GH_Q_TABLE, &tp, 8);
    int32_t m1 = -1; uint32_t tag = 8; memcpy(q->b + GH_Q_INDEX_COLUMN, &m1, 4); memcpy(q->b + GH_Q_TAG, &tag, 4);
    memcpy(q->b + GH_Q_KEY, &key, 8);
}

static uint64_t g_rng = 0x2545f4914f6cdd1dull;
static uint64_t rnd(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return g_rng; }
static char *random_name(void) {
    static const char *pre[] = {"Obj_", "Npc_", "Enemy_", "Item_", "Weapon_Sword_", "Obj_Tree_", "Obj_TreeApple_", "DgnObj_"};
    char b[64]; int k = snprintf(b, sizeof b, "%s", pre[rnd() % 8]);
    int len = 1 + (int)(rnd() % 14);
    for (int i = 0; i < len; i++) b[k++] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"[rnd() % 63];
    b[k] = 0;
    return strdup(b);
}
static int by_name(const void *a, const void *b) { return strcmp(*(char *const *)a, *(char *const *)b); }

static MockTable *make_table(char **names, int n, int sort, int dedupe) {
    if (sort) qsort(names, (size_t)n, sizeof *names, by_name);
    if (dedupe) { int m = 0; for (int i = 0; i < n; i++) if (!m || strcmp(names[m - 1], names[i])) names[m++] = names[i]; n = m; }
    MockTable *t = calloc(1, sizeof *t);
    MockRow *rows = calloc((size_t)n + 1, sizeof *rows);
    uint64_t *ptrs = calloc((size_t)n + 1, 8);
    for (int i = 0; i < n; i++) { rows[i].name = names[i]; rows[i].tag = 8; ptrs[i] = (uint64_t)(uintptr_t)&rows[i]; }
    t->vt = mock_vt; t->count = (uint32_t)n;
    t->index = (MockIndex){n, 0, ptrs, t, 0, 0};
    return t;
}

static unsigned long g_queries, g_answered, g_found;
static void check(MockTable *t, const char *key, int expect_answer) {
    Query q; make_query(&q, t, key);
    uint64_t want = game(q.b);
    for (int pass = 0; pass < 2; pass++) {
        uint64_t got = 0xdead;
        int ok = gh_answer(q.b, &got);
        g_queries++;
        if (ok) {
            g_answered++;
            if (got != want) { fprintf(stderr, "MISMATCH key \"%s\" n=%d pass %d: host %#llx game %#llx\n", key, t->index.n, pass, (unsigned long long)got, (unsigned long long)want); exit(1); }
        }
        if (expect_answer >= 0 && ok != expect_answer) { fprintf(stderr, "key \"%s\" n=%d: answered=%d, expected %d\n", key, t->index.n, ok, expect_answer); exit(1); }
    }
    if (want) g_found++;
}

static void keys_for(MockTable *t, int expect) {
    int n = t->index.n;
    MockRow *rows = n ? (MockRow *)(uintptr_t)t->index.rows[0] : NULL;
    check(t, "", expect); check(t, "\x01", expect); check(t, "\xff\xff", expect); check(t, "A", expect); check(t, "Obj_", expect);
    for (int i = 0; i < n; i++) {
        const char *name = rows[i].name;
        char *copy = strdup(name); check(t, copy, expect); free(copy);
        check(t, name, expect);   /* the row's own pointer: equal without comparing */
        char longer[80]; snprintf(longer, sizeof longer, "%sx", name); check(t, longer, expect);
        size_t len = strlen(name); char shorter[80]; snprintf(shorter, sizeof shorter, "%.*s", (int)(len > 1 ? len - 1 : 0), name); check(t, shorter, expect);
        if (rnd() % 4 == 0) { char *r = random_name(); check(t, r, expect); free(r); }
    }
}

typedef struct { MockTable *t; char **keys; uint64_t *want; int nkeys; } Work;
static void *worker(void *arg) {
    Work *w = arg;
    for (int round = 0; round < 20; round++)
        for (int i = 0; i < w->nkeys; i++) {
            Query q; make_query(&q, w->t, w->keys[i]);
            uint64_t row = 0;
            if (gh_answer(q.b, &row) && row != w->want[i]) { fprintf(stderr, "THREAD MISMATCH \"%s\"\n", w->keys[i]); exit(1); }
        }
    return NULL;
}

int main(int argc, char **argv) {
    uint8_t *base = map_main(argc > 1 ? argv[1] : "extracted/v121/elf/main.elf");
    game = (gh_query_fn)(void *)(base + 0x7f3688);
    gh_original = game;
    for (unsigned i = 0; i < sizeof mock_vt / sizeof *mock_vt; i++) mock_vt[i] = (void *)mock_unexpected;
    mock_vt[0x60 / 8] = (void *)mock_count; mock_vt[0x70 / 8] = (void *)mock_index; mock_vt[0x78 / 8] = (void *)mock_value;

    /* 1. sorted unique names */
    static const int sizes[] = {0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 31, 100, 257, 1000, 4000};
    for (unsigned s = 0; s < sizeof sizes / sizeof *sizes; s++) {
        int n = sizes[s]; char **names = malloc(sizeof(char *) * (size_t)(n + 1));
        for (int i = 0; i < n; i++) names[i] = random_name();
        MockTable *t = make_table(names, n, 1, 1);
        keys_for(t, 1);
    }
    printf("sorted unique: %lu host answers checked against the game (%lu queries found a row), all equal\n", g_answered, g_found);

    /* 2. sorted with duplicates: the first of equal rows */
    unsigned long before = g_answered;
    for (int rep = 0; rep < 30; rep++) {
        int n = 2 + (int)(rnd() % 300); char **names = malloc(sizeof(char *) * (size_t)n);
        for (int i = 0; i < n; i++) names[i] = i && rnd() % 3 == 0 ? strdup(names[rnd() % (unsigned)i]) : random_name();
        MockTable *t = make_table(names, n, 1, 0);
        keys_for(t, 1);
    }
    printf("sorted with duplicates: %lu answers, all equal (the first equal row)\n", g_answered - before);

    /* 3. unsorted, and a non-string value: left to the game */
    {
        int n = 200; char **names = malloc(sizeof(char *) * (size_t)n);
        for (int i = 0; i < n; i++) names[i] = random_name();
        MockTable *t = make_table(names, n, 1, 1);
        MockRow *r0 = (MockRow *)(uintptr_t)t->index.rows[0];
        int m = t->index.n; const char *tmp = r0[3].name; r0[3].name = r0[m - 3].name; r0[m - 3].name = tmp;   /* two rows swapped */
        keys_for(t, 0);
        char **names2 = malloc(sizeof(char *) * (size_t)n);
        for (int i = 0; i < n; i++) names2[i] = random_name();
        MockTable *u = make_table(names2, n, 1, 1);
        ((MockRow *)(uintptr_t)u->index.rows[0])[u->index.n / 2].tag = 3;
        keys_for(u, 0);
        printf("unsorted / non-string value: every query left to the game (%u indexes known, %u not sorted)\n",
               atomic_load(&gh_index_count), gh_unsorted);
    }

    /* 4. garbage hints */
    before = g_answered;
    {
        int n = 1500; char **names = malloc(sizeof(char *) * (size_t)n);
        for (int i = 0; i < n; i++) names[i] = random_name();
        MockTable *t = make_table(names, n, 1, 1);
        for (int round = 0; round < 3; round++) {
            for (unsigned i = 0; i < (1u << GH_HINT_BITS); i++) atomic_store(&gh_hints[i], rnd());
            keys_for(t, 1);
            for (unsigned i = 0; i < (1u << GH_HINT_BITS); i++) {   /* right tags, wrong positions */
                uint64_t e = atomic_load(&gh_hints[i]);
                if ((uint32_t)e) atomic_store(&gh_hints[i], (e & ~0xffffffffull) | (1 + rnd() % (uint64_t)(n + 2)));
            }
            keys_for(t, 1);
        }
    }
    printf("garbage hints: %lu answers, all equal\n", g_answered - before);

    /* 4b. the proof at every position directly, with duplicates; negative control: without the left neighbour's check */
    {
        unsigned long proven = 0, naive_wrong = 0;
        for (int rep = 0; rep < 40; rep++) {
            int n = 1 + (int)(rnd() % 60); char **names = malloc(sizeof(char *) * (size_t)n);
            for (int i = 0; i < n; i++) names[i] = i && rnd() % 3 == 0 ? strdup(names[rnd() % (unsigned)i]) : random_name();
            MockTable *t = make_table(names, n, 1, 0);
            gh_value_fn get = (gh_value_fn)(void *)mock_value;
            for (int k = 0; k < 3 * n; k++) {
                const MockRow *r = (const MockRow *)(uintptr_t)t->index.rows[rnd() % (uint64_t)n];
                char *key = k % 3 ? strdup(r->name) : random_name();
                Query q; make_query(&q, t, key);
                uint64_t want = game(q.b);
                for (uint32_t i = 0; i <= (uint32_t)n + 1; i++) {
                    uint64_t row = 0xdead; unsigned reads = 0;
                    if (gh_prove(t, get, t->index.rows, n, 0, (const uint8_t *)key, i, &row, &reads)) {
                        proven++;
                        if (row != want) { fprintf(stderr, "PROOF WRONG at %u for \"%s\"\n", i, key); return 1; }
                    }
                    if (i < (uint32_t)n) {   /* naive: only the row at i */
                        int c = strcmp(((const MockRow *)(uintptr_t)t->index.rows[i])->name, key);
                        if (c >= 0 && (c == 0 ? t->index.rows[i] : 0) != want) naive_wrong++;
                    }
                }
                free(key);
            }
        }
        if (!naive_wrong) runtime_fail("negative control: the naive proof was never wrong (the test would not see a bad proof)");
        printf("proof at every position: %lu proven, all equal; negative control (no left-neighbour check) wrong %lu times\n", proven, naive_wrong);
    }

    /* 5. four threads */
    {
        int n = 3000; char **names = malloc(sizeof(char *) * (size_t)n);
        for (int i = 0; i < n; i++) names[i] = random_name();
        MockTable *t = make_table(names, n, 1, 1);
        Work w = {t, malloc(sizeof(char *) * 4000), malloc(8 * 4000), 4000};
        for (int i = 0; i < w.nkeys; i++) {
            w.keys[i] = i % 2 ? random_name() : strdup(((MockRow *)(uintptr_t)t->index.rows[rnd() % (uint64_t)t->index.n])->name);
            Query q; make_query(&q, t, w.keys[i]); w.want[i] = game(q.b);
        }
        for (unsigned i = 0; i < (1u << GH_HINT_BITS); i++) atomic_store(&gh_hints[i], 0);
        pthread_t th[4];
        for (int i = 0; i < 4; i++) pthread_create(&th[i], NULL, worker, &w);
        for (int i = 0; i < 4; i++) pthread_join(th[i], NULL);
        printf("4 threads x 20 rounds x %d keys: all equal\n", w.nkeys);
    }

    /* 6. gh_query (the veneer's host entry) in both modes, and other query shapes left to the game */
    {
        int n = 800; char **names = malloc(sizeof(char *) * (size_t)n);
        for (int i = 0; i < n; i++) names[i] = random_name();
        MockTable *t = make_table(names, n, 1, 1);
        for (int mode = 1; mode <= 2; mode++) {
            gh_mode = mode;
            for (int i = 0; i < 2000; i++) {
                char *k = i % 2 ? random_name() : strdup(((MockRow *)(uintptr_t)t->index.rows[rnd() % (uint64_t)t->index.n])->name);
                Query q; make_query(&q, t, k);
                if (gh_query(q.b) != game(q.b)) { fprintf(stderr, "gh_query mode %d differs for \"%s\"\n", mode, k); return 1; }
                free(k);
            }
        }
        Query q; make_query(&q, t, "Obj_x"); int32_t one = 1; memcpy(q.b + GH_Q_OPERATOR, &one, 4);
        uint64_t row; if (gh_answer(q.b, &row)) runtime_fail("another operator was answered");
        make_query(&q, t, "Obj_x"); q.b[GH_Q_REVERSE] = 1; if (gh_answer(q.b, &row)) runtime_fail("reverse order was answered");
        printf("gh_query modes 1 and 2: %llu calls, %llu answered on the host, %llu left to the game, %llu self-checked; other shapes left to the game\n",
               (unsigned long long)gh_calls, (unsigned long long)gh_answered, (unsigned long long)gh_fallbacks, (unsigned long long)gh_checked);
    }

    /* 7. cost: an actor-info-sized table, names that share long prefixes */
    {
        int n = 12000; char **names = malloc(sizeof(char *) * (size_t)n);
        for (int i = 0; i < n; i++) names[i] = random_name();
        MockTable *t = make_table(names, n, 1, 1);
        enum { K = 4096, ROUNDS = 100 };
        static Query qs[K];
        for (int i = 0; i < K; i++) make_query(&qs[i], t, ((MockRow *)(uintptr_t)t->index.rows[rnd() % (uint64_t)t->index.n])->name);
        for (int i = 0; i < K; i++) { uint64_t r; gh_answer(qs[i].b, &r); }   /* hints in */
        volatile uint64_t sink = 0;
        uint64_t r0 = mock_reads, t0 = runtime_now_ns();
        for (int r = 0; r < ROUNDS; r++) for (int i = 0; i < K; i++) sink += game(qs[i].b);
        uint64_t t1 = runtime_now_ns(), r1 = mock_reads;
        for (int r = 0; r < ROUNDS; r++) for (int i = 0; i < K; i++) { uint64_t row = 0; gh_answer(qs[i].b, &row); sink += row; }
        uint64_t t2 = runtime_now_ns(), r2 = mock_reads;
        double calls = (double)K * ROUNDS;
        printf("cost, %d rows (%d live): game %.0f ns and %.1f value reads per lookup; host with hints %.0f ns and %.1f reads (%.1fx faster)\n",
               t->index.n, n, (t1 - t0) / calls, (r1 - r0) / calls, (t2 - t1) / calls, (r2 - r1) / calls, (double)(t1 - t0) / (double)(t2 - t1));
        (void)sink;
    }
    /* 8. the veneer: main+0x7f3688 patched as the runtime does; queries of the host's shape reach gh_query, others run the
       game's code from the relocated first instruction */
    {
        int n = 2500; char **names = malloc(sizeof(char *) * (size_t)n);
        for (int i = 0; i < n; i++) names[i] = random_name();
        MockTable *t = make_table(names, n, 1, 1);
        Module m = {0};
        m.base = base; m.file_size = 0x3880010; m.patch_off = 0x3880000; m.size = 0x3888000;
        setenv("TOTK_GAMEHOT", "1", 1);
        uint64_t calls0 = gh_calls;
        pthread_jit_write_protect_np(0);
        gamehot_install(&m, 0x7f3688, 0xd10203ff);
        pthread_jit_write_protect_np(1);
        sys_icache_invalidate(base, 0x3888000);
        gh_query_fn entry = (gh_query_fn)(void *)(base + 0x7f3688);
        unsigned long shaped = 0, other = 0;
        for (int i = 0; i < 6000; i++) {
            char *k = i % 2 ? random_name() : strdup(((MockRow *)(uintptr_t)t->index.rows[rnd() % (uint64_t)t->index.n])->name);
            Query q; make_query(&q, t, k);
            if (i % 3 == 2) { q.b[GH_Q_REVERSE] = 1; other++; } else shaped++;
            uint64_t got = entry(q.b), want = gh_original(q.b);
            if (got != want) { fprintf(stderr, "VENEER MISMATCH \"%s\" reverse=%d\n", k, q.b[GH_Q_REVERSE]); return 1; }
            free(k);
        }
        if (gh_calls - calls0 != shaped) { fprintf(stderr, "veneer sent %llu queries to the host, expected %lu\n", (unsigned long long)(gh_calls - calls0), shaped); return 1; }
        printf("veneer: %lu queries of the host's shape went to the host, %lu others ran the game's code; all equal to the game's\n", shaped, other);
    }
    printf("PASS\n");
    return 0;
}
