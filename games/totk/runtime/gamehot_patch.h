/* TOTK_GAMEHOT (1.2.1, Mac; off by default): the RSDB row lookup answered on the host from a validated hint.

   What it is. main+0x7f3688 is the query executor of the game's RSDB tables (engine::rsdb / pp::rsdb, RSDB/<name>.rstbl.byml):
   Row *find_first(Query *q). The 332 callers of main+0xb95b98 ("row whose __RowId is <name>") build the query on the stack
   and call it; so do 17 other sites with other query shapes. In the Korok Forest flight of the UltraCam benchmark it was the
   hottest guest function (macOS sample, 1535 of 4241 guest top-of-stack samples): 1281 on a streaming worker that looks up
   an actor-info table by actor name while actors spawn (main+0x26aee58 -> 0x7fb57c -> 0x7fc668 -> 0x1988740 -> 0x19888c0 ->
   0xb95b98), about 200 inside the per-frame job batch the game thread waits for (main+0x7edb8c -> 0x86a98c -> 0xcd1a20 ->
   ... -> 0x1988ea0 -> 0x19888c0), of which about 100 on the game thread itself.

   Query fields read here (all as the executor reads them): +0x00 table, +0x08 number of extra predicates, +0xc8 column
   (-1 none), +0xcc operator (0 equal), +0xd0 / +0xd8 the tagged value compared with (tag 8 = C string), +0xe0 forced
   index column (-1 none), +0xe4 reverse order. Table virtuals: +0x60 row count (called, result unused on this path),
   +0x70 index for a column, +0x78 value of (row, column) -> {u32 tag, u64 value} through x8. Index: +0x00 int count,
   +0x08 row pointers sorted by the column, +0x10 the table it reads, +0x18 its column. The executor does, for an equality
   on a string with an index and no predicates: a lower-bound binary search (one virtual value call and one byte-by-byte
   strcmp, capped at 0x400000 bytes, per probe), an upper-bound binary search (the same again), then a scan of that range
   for the first row whose value equals the key; it returns that row or null. Two binary searches over ~log2(rows) probes
   each, every probe a virtual call plus a strcmp of names that share long prefixes: that is the cost, and the same names
   are looked up again and again while actors spawn.

   What the host does, for exactly that query shape (the veneer at the entry checks it; any other shape runs the game's
   code unchanged): it keeps, per (index, key), the position the lower bound had last time (a hint in a 2-way hashed
   table) and proves the answer at that position with at most two value reads (strcmp = the game's comparison):
     i > 0  : value(i-1) is a string and strcmp(value(i-1), key) < 0
     i < n  : value(i) is a string; strcmp == 0 -> row i is the answer; > 0 -> no row has the key; < 0 -> not proven
     i == n : no row has the key
   On an index whose values are all strings in non-decreasing strcmp order, that is precisely what the game computes:
   its lower bound lands on i, its scan starts there, and row i matches (or the range holds no equal row). That order is
   checked once per index (all n values read and compared pairwise, the index identified by its address, row array,
   count, table and column); an index that fails stays on the game's code. Without a hint, or when it fails, the host
   runs its own lower-bound search (log2(n) probes, half of the game's) and proves the result the same way; whatever
   cannot be proven (a non-string value, a null key, an index whose table or column differs from the query's) runs the
   game's code. A hint is only a guess: a stale, torn or colliding entry fails the proof and costs a
   search, never a wrong row, so no generation or invalidation is needed and no lock is taken on the lookup path.
   The value calls are the game's own virtual functions, called in the same order of kinds (+0x70, +0x60, then +0x78).

   TOTK_GAMEHOT=1 enables it; =2 also runs the game's code for every query of that shape and stops the game on any
   difference (self-check; slower than the game alone). A statistics line every 10 s while queries happen. */
#pragma once
#include "runtime.h"
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { GH_Q_TABLE = 0x00, GH_Q_PREDICATES = 0x08, GH_Q_COLUMN = 0xc8, GH_Q_OPERATOR = 0xcc, GH_Q_TAG = 0xd0, GH_Q_KEY = 0xd8,
       GH_Q_INDEX_COLUMN = 0xe0, GH_Q_REVERSE = 0xe4 };
enum { GH_TAG_STRING = 8, GH_STRCMP_CAP = 0x400000, GH_MAX_ROWS = 1 << 22 };

/* The game writes 16 bytes {tag, value} through x8; a result type over 16 bytes makes the compiler pass x8 the same way. */
typedef struct { uint32_t tag, pad; uint64_t value; uint64_t spare[2]; } GhValue;
typedef void *(*gh_index_fn)(void *table, uint64_t column);
typedef uint32_t (*gh_count_fn)(void *table);
typedef GhValue (*gh_value_fn)(void *table, void *row, uint64_t column);
typedef uint64_t (*gh_query_fn)(void *query);

static int gh_mode;                 /* 0 off, 1 on, 2 self-check */
static gh_query_fn gh_original;     /* the game's executor (its first instruction, then main+0x7f368c) */

static _Atomic uint64_t gh_calls, gh_answered, gh_hits, gh_searches, gh_fallbacks, gh_reads, gh_hit_reads, gh_checked;
static _Atomic uint64_t gh_last_print_ns, gh_last_print_calls;

/* The game's comparison (main+0x7f37a4..0x7f37dc): the same pointer is equal; otherwise unsigned bytes until a difference
   or a NUL in both, at most 0x400000 bytes (then equal). strncmp with that limit is the same function (unsigned bytes,
   0 when the limit is reached). Sign of row value versus key. */
static inline int gh_cmp(const uint8_t *value, const uint8_t *key) {
    return value == key ? 0 : strncmp((const char *)value, (const char *)key, GH_STRCMP_CAP);
}

/* Indexes checked for string values in non-decreasing order. Appended under the lock, read without it (count published
   with release after the entry is written); each thread remembers the last entry it matched. */
typedef struct { const void *index, *rows, *table; int32_t n; uint32_t column; int sorted; } GhIndex;
#define GH_MAX_INDEXES 256
static GhIndex gh_indexes[GH_MAX_INDEXES];
static _Atomic unsigned gh_index_count;
static unsigned gh_unsorted;
static pthread_mutex_t gh_index_lock = PTHREAD_MUTEX_INITIALIZER;
static _Thread_local const GhIndex *gh_last_index;

static inline int gh_index_is(const GhIndex *x, const void *index, const void *rows, const void *table, int32_t n, uint32_t column) {
    return x->index == index && x->rows == rows && x->table == table && x->n == n && x->column == column;
}
static int gh_index_find(const void *index, const void *rows, const void *table, int32_t n, uint32_t column) {
    const GhIndex *last = gh_last_index;
    if (last && gh_index_is(last, index, rows, table, n, column)) return last->sorted;
    unsigned count = atomic_load_explicit(&gh_index_count, memory_order_acquire);
    for (unsigned k = 0; k < count; k++)
        if (gh_index_is(&gh_indexes[k], index, rows, table, n, column)) { gh_last_index = &gh_indexes[k]; return gh_indexes[k].sorted; }
    return -1;
}

/* 1 when every value is a non-null string and each is <= the next by the game's comparison; 0 otherwise. n value reads. */
static int gh_index_verify(void *table, gh_value_fn get, const uint64_t *rows, int32_t n, uint32_t column) {
    const uint8_t *previous = NULL;
    for (int32_t k = 0; k < n; k++) {
        GhValue v = get(table, (void *)(uintptr_t)rows[k], column);
        if (v.tag != GH_TAG_STRING || !v.value) return 0;
        const uint8_t *s = (const uint8_t *)(uintptr_t)v.value;
        if (previous && gh_cmp(previous, s) > 0) return 0;
        previous = s;
    }
    return 1;
}

static int gh_index_state(void *index, const uint64_t *rows, void *table, gh_value_fn get, int32_t n, uint32_t column) {
    int state = gh_index_find(index, rows, table, n, column);
    if (state >= 0) return state;
    uint64_t t0 = runtime_now_ns();
    int sorted = gh_index_verify(table, get, rows, n, column);
    double ms = (runtime_now_ns() - t0) / 1e6;
    pthread_mutex_lock(&gh_index_lock);
    state = gh_index_find(index, rows, table, n, column);
    unsigned count = atomic_load_explicit(&gh_index_count, memory_order_relaxed);
    if (state < 0 && count < GH_MAX_INDEXES) {
        gh_indexes[count] = (GhIndex){index, rows, table, n, column, sorted};
        atomic_store_explicit(&gh_index_count, count + 1, memory_order_release);
        if (!sorted) gh_unsorted++;
        fprintf(stderr, "[totk] gamehot: index %p (table %p, column %u, %d rows) %s (checked in %.2f ms)\n", index, table, column, n,
                sorted ? "in string order: answered on the host" : "NOT in string order: left to the game", ms);
        if (count + 1 == GH_MAX_INDEXES) fprintf(stderr, "[totk] gamehot: %d indexes known, later ones are left to the game\n", GH_MAX_INDEXES);
        state = sorted;
    } else if (state < 0) state = 0;
    pthread_mutex_unlock(&gh_index_lock);
    return state;
}

/* Hints: the lower-bound position for (index, key) seen last time, in 2-way sets. Entry = hash high 32 bits << 32 |
   (position + 1); 0 = empty. */
#define GH_HINT_BITS 15
static _Atomic uint64_t gh_hints[1u << GH_HINT_BITS];
static uint64_t gh_hash(const uint8_t *key, const void *index) {
    size_t len = strnlen((const char *)key, 256);
    uint64_t h = (uint64_t)(uintptr_t)index * 0x9e3779b97f4a7c15ull ^ len;
    size_t k = 0;
    for (; k + 8 <= len; k += 8) { uint64_t w; memcpy(&w, key + k, 8); h = (h ^ w) * 0xff51afd7ed558ccdull; h ^= h >> 32; }
    uint64_t tail = 0;
    memcpy(&tail, key + k, len - k);
    h = (h ^ tail) * 0xc4ceb9fe1a85ec53ull;
    return h ^ (h >> 29);
}

/* Proves the answer at lower-bound position i (see the top). 1 with *row set (0 = none), 0 when not proven. */
static int gh_prove(void *table, gh_value_fn get, const uint64_t *rows, int32_t n, uint32_t column, const uint8_t *key,
                    uint32_t i, uint64_t *row, unsigned *reads) {
    if (i > (uint32_t)n) return 0;
    if (i > 0) {
        GhValue v = get(table, (void *)(uintptr_t)rows[i - 1], column);
        ++*reads;
        if (v.tag != GH_TAG_STRING || gh_cmp((const uint8_t *)(uintptr_t)v.value, key) >= 0) return 0;
    }
    if (i == (uint32_t)n) { *row = 0; return 1; }
    GhValue v = get(table, (void *)(uintptr_t)rows[i], column);
    ++*reads;
    if (v.tag != GH_TAG_STRING) return 0;
    int c = gh_cmp((const uint8_t *)(uintptr_t)v.value, key);
    if (c < 0) return 0;
    *row = c == 0 ? rows[i] : 0;
    return 1;
}

static int gh_shape(const uint8_t *q) {
    return *(const uint32_t *)(q + GH_Q_TAG) == GH_TAG_STRING && *(const uint32_t *)(q + GH_Q_OPERATOR) == 0 &&
           *(const int32_t *)(q + GH_Q_INDEX_COLUMN) == -1 && q[GH_Q_REVERSE] == 0 && *(const int32_t *)(q + GH_Q_PREDICATES) == 0 &&
           *(const int32_t *)(q + GH_Q_COLUMN) != -1;
}

/* 1 with *row = the game's answer, 0 when the game's code has to run. */
static int gh_answer(uint8_t *q, uint64_t *row) {
    if (!gh_shape(q)) return 0;
    void *table = *(void **)(q + GH_Q_TABLE);
    uint32_t column = *(const uint32_t *)(q + GH_Q_COLUMN);
    const uint8_t *key = *(const uint8_t **)(q + GH_Q_KEY);
    if (!key) return 0;
    void **vt = *(void ***)table;
    uint8_t *index = ((gh_index_fn)vt[0x70 / 8])(table, column);
    ((gh_count_fn)vt[0x60 / 8])(table);
    int32_t n = *(const int32_t *)index;
    if (n == 0) { *row = 0; return 1; }   /* main+0x7f3728: an empty index finds nothing */
    if (n < 0 || n > GH_MAX_ROWS) return 0;
    const uint64_t *rows = *(uint64_t **)(index + 0x08);
    void *reader = *(void **)(index + 0x10);
    if (reader != table || *(const uint32_t *)(index + 0x18) != column) return 0;
    gh_value_fn get = (gh_value_fn)vt[0x78 / 8];
    if (gh_index_state(index, rows, table, get, n, column) != 1) return 0;

    uint64_t h = gh_hash(key, index), tag = h >> 32;
    _Atomic uint64_t *set = &gh_hints[h & ((1u << GH_HINT_BITS) - 2)];
    unsigned reads = 0;
    for (int way = 0; way < 2; way++) {
        uint64_t e = atomic_load_explicit(&set[way], memory_order_relaxed);
        if ((uint32_t)e && (e >> 32) == tag && gh_prove(table, get, rows, n, column, key, (uint32_t)e - 1, row, &reads)) {
            atomic_fetch_add_explicit(&gh_hits, 1, memory_order_relaxed);
            atomic_fetch_add_explicit(&gh_hit_reads, reads, memory_order_relaxed);
            atomic_fetch_add_explicit(&gh_reads, reads, memory_order_relaxed);
            return 1;
        }
    }
    uint32_t lo = 0, hi = (uint32_t)n;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        GhValue v = get(table, (void *)(uintptr_t)rows[mid], column);
        reads++;
        if (v.tag != GH_TAG_STRING) { atomic_fetch_add_explicit(&gh_reads, reads, memory_order_relaxed); return 0; }
        if (gh_cmp((const uint8_t *)(uintptr_t)v.value, key) < 0) lo = mid + 1; else hi = mid;
    }
    int ok = gh_prove(table, get, rows, n, column, key, lo, row, &reads);
    atomic_fetch_add_explicit(&gh_reads, reads, memory_order_relaxed);
    if (!ok) return 0;
    atomic_fetch_add_explicit(&gh_searches, 1, memory_order_relaxed);
    /* the newest entry goes to the first way; another key there moves to the second (the two most recent stay) */
    uint64_t e0 = atomic_load_explicit(&set[0], memory_order_relaxed);
    if ((uint32_t)e0 && (e0 >> 32) != tag) atomic_store_explicit(&set[1], e0, memory_order_relaxed);
    atomic_store_explicit(&set[0], tag << 32 | (uint64_t)(lo + 1), memory_order_relaxed);
    return 1;
}

static void gh_stats(void) {
    uint64_t calls = atomic_load_explicit(&gh_calls, memory_order_relaxed);
    if (calls & 4095) return;
    uint64_t now = runtime_now_ns(), last = atomic_load_explicit(&gh_last_print_ns, memory_order_relaxed);
    if (now - last < 10000000000ull || !atomic_compare_exchange_strong(&gh_last_print_ns, &last, now)) return;
    uint64_t answered = atomic_load_explicit(&gh_answered, memory_order_relaxed), hits = atomic_load_explicit(&gh_hits, memory_order_relaxed);
    uint64_t searches = atomic_load_explicit(&gh_searches, memory_order_relaxed), reads = atomic_load_explicit(&gh_reads, memory_order_relaxed);
    uint64_t hit_reads = atomic_load_explicit(&gh_hit_reads, memory_order_relaxed), checked = atomic_load_explicit(&gh_checked, memory_order_relaxed);
    uint64_t before = atomic_exchange_explicit(&gh_last_print_calls, calls, memory_order_relaxed);
    fprintf(stderr, "[totk] gamehot: %llu queries (+%llu), %llu answered on the host (%llu hint hits, %llu searches), "
            "%llu left to the game; value reads per answer %.2f (hit %.2f, search %.2f); %u indexes (%u unsorted)\n",
            (unsigned long long)calls, (unsigned long long)(calls - before), (unsigned long long)answered, (unsigned long long)hits,
            (unsigned long long)searches, (unsigned long long)atomic_load_explicit(&gh_fallbacks, memory_order_relaxed),
            hits + searches ? (double)reads / (double)(hits + searches) : 0.0, hits ? (double)hit_reads / hits : 0.0,
            searches ? (double)(reads - hit_reads) / searches : 0.0, atomic_load_explicit(&gh_index_count, memory_order_relaxed),
            gh_unsorted);
    if (gh_mode == 2) fprintf(stderr, "[totk] gamehot self-check: %llu answers compared with the game's, all equal\n", (unsigned long long)checked);
}

/* The host entry (through runtime_hle_wrap: on the guest thread's host stack). x0 = the query, returns the row. */
static uint64_t gh_query(void *query) {
    uint8_t *q = query;
    uint64_t row = 0;
    atomic_fetch_add_explicit(&gh_calls, 1, memory_order_relaxed);
    if (gh_mode == 2) {
        uint64_t want = gh_original(query);
        if (gh_answer(q, &row)) {
            atomic_fetch_add_explicit(&gh_answered, 1, memory_order_relaxed);
            if (row != want) {
                const char *key = *(const char **)(q + GH_Q_KEY);
                fprintf(stderr, "[totk] gamehot self-check MISMATCH: table %p column %u key \"%.200s\": host row %#llx, game row %#llx\n",
                        *(void **)(q + GH_Q_TABLE), *(const uint32_t *)(q + GH_Q_COLUMN), key ? key : "(null)",
                        (unsigned long long)row, (unsigned long long)want);
                runtime_fail("TOTK_GAMEHOT=2: the host's RSDB row differs from the game's");
            }
            atomic_fetch_add_explicit(&gh_checked, 1, memory_order_relaxed);
        } else atomic_fetch_add_explicit(&gh_fallbacks, 1, memory_order_relaxed);
        gh_stats();
        return want;
    }
    if (gh_answer(q, &row)) atomic_fetch_add_explicit(&gh_answered, 1, memory_order_relaxed);
    else { atomic_fetch_add_explicit(&gh_fallbacks, 1, memory_order_relaxed); row = gh_original(query); }
    gh_stats();
    return row;
}

static inline uint32_t gh_branch(uint64_t from, uint64_t to) {   /* b <to>, both offsets in main */
    int64_t delta = (int64_t)to - (int64_t)from;
    if ((delta & 3) || delta < -(1 << 27) || delta >= (1 << 27)) runtime_fail("gamehot branch outside range");
    return 0x14000000u | ((uint32_t)(delta >> 2) & 0x3ffffffu);
}

static inline int gamehot_requested(void) {
    const char *e = getenv("TOTK_GAMEHOT");
    if (!e || !*e || !strcmp(e, "0")) return 0;
    if (!strcmp(e, "1")) return 1;
    if (!strcmp(e, "2")) return 2;
    runtime_fail("TOTK_GAMEHOT must be 0, 1 or 2 (self-check)");
}

/* The veneer at main+0x7f3688 (the executor's first instruction, sub sp, sp, #0x80): queries of the host's shape go to
   the host (x0 = query, x30 = the caller's return address, x16 free at a call); the rest run the first instruction and
   continue at +4. It sits in the patch area's tail at +0x400 (the 1.2.1 benchmark camera stubs use the first 192 bytes),
   executable and never written by the rewriter. The host address is a literal in the veneer, so the code is Mac only
   (iOS runs code patched at build time). */
static inline void gamehot_install(Module *m, uint64_t site, uint32_t first_word) {
    gh_mode = gamehot_requested();
    if (!gh_mode) return;
    if (getenv("TOTK_EXPORT_IOS")) { fprintf(stderr, "[totk] gamehot: not with TOTK_EXPORT_IOS (host address in the code)\n"); gh_mode = 0; return; }
    const uint64_t at = ((m->file_size + 15) & ~15ULL) + 0x400, data = (m->file_size + 0x3fff) & ~0x3fffULL;
    enum { WORDS = 22 };   /* 19 instructions, a brk as padding, the 8-byte literal at +80 */
    if (at < m->patch_off || at + 4 * WORDS > data) runtime_fail("gamehot veneer layout");
    for (unsigned i = 0; i < 4 * WORDS; i++) if (m->base[at + i]) runtime_fail("gamehot veneer area not empty");
    uint32_t word;
    memcpy(&word, m->base + site, 4);
    if (word != first_word) {
        fprintf(stderr, "[totk] gamehot main+%#llx: found %08x, expected %08x\n", (unsigned long long)site, word, first_word);
        runtime_fail("gamehot: the RSDB query executor does not match this game version");
    }
    const uint32_t orig = 17 * 4;   /* the original's first instruction */
    uint32_t v[WORDS] = {
        0xb940d010,                                   /* ldr  w16, [x0, #0xd0]   value tag */
        0x7100221f,                                   /* cmp  w16, #8            string */
        0x54000000u | ((orig - 8) / 4) << 5 | 0x1,     /* b.ne orig */
        0xb940cc10,                                   /* ldr  w16, [x0, #0xcc]   operator */
        0x35000000u | ((orig - 16) / 4) << 5 | 16,     /* cbnz w16, orig          not "equal" */
        0xb940e010,                                   /* ldr  w16, [x0, #0xe0]   forced index column */
        0x3100061f,                                   /* cmn  w16, #1 */
        0x54000000u | ((orig - 28) / 4) << 5 | 0x1,    /* b.ne orig */
        0x39439010,                                   /* ldrb w16, [x0, #0xe4]   reverse */
        0x35000000u | ((orig - 36) / 4) << 5 | 16,     /* cbnz w16, orig */
        0xb9400810,                                   /* ldr  w16, [x0, #0x8]    extra predicates */
        0x35000000u | ((orig - 44) / 4) << 5 | 16,     /* cbnz w16, orig */
        0xb940c810,                                   /* ldr  w16, [x0, #0xc8]   column */
        0x3100061f,                                   /* cmn  w16, #1 */
        0x54000000u | ((orig - 56) / 4) << 5 | 0x0,    /* b.eq orig               no column */
        0x58000000u | ((80 - 60) / 4) << 5 | 16,       /* ldr  x16, host          (literal at +80) */
        0xd61f0200,                                   /* br   x16 */
        first_word,                                   /* orig: the replaced instruction */
        gh_branch(at + orig + 4, site + 4),           /* b    main+0x7f368c */
        0xd4200000,                                   /* brk  #0 (padding) */
        0, 0};                                        /* the host thunk's address */
    uint64_t host = (uint64_t)(uintptr_t)runtime_hle_wrap_named((void *)gh_query, "gamehot: RSDB query (main+0x7f3688)");
    memcpy(&v[20], &host, 8);
    memcpy(m->base + at, v, sizeof v);
    gh_original = (gh_query_fn)(void *)(m->base + at + orig);
    uint32_t jump = gh_branch(site, at);
    memcpy(m->base + site, &jump, 4);
    fprintf(stderr, "[totk] gamehot (TOTK_GAMEHOT=%d): RSDB row lookups by string key answered on the host from validated hints%s; "
            "veneer main+%#llx at main+%#llx\n", gh_mode, gh_mode == 2 ? ", every answer compared with the game's (self-check)" : "",
            (unsigned long long)at, (unsigned long long)site);
}
