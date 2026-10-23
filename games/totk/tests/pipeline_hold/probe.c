/* CPU only: TOTK_PIPELINE page state (nvn_dirty.c "one frame deeper"): classification of GPU reads (snapshot or in
   place), probation, the release-time protection and the holds of CPU writes while a released frame still runs. A fake
   "completed fence" stands in for the GPU and a fake guest-thread handle for the runtime's.
   - a page seen the first time is put on probation (protected, snapshot for PROBE frames), then static;
   - a CPU write on a protected page makes it volatile (snapshot) for KEEP frames;
   - a page with a GPU write in flight is read in place (-1);
   - a write to a page read in place by a released, unfinished submission waits until the GPU passes it; a host write
     announced through the write-prepare hook waits the same way; a host thread (handle 0) is not held;
   - a write to a page read only by a submission not released does not wait; neither one read by a completed one;
   - a page read by both the released submission and the next one (pl_rf2) waits for the released one only;
   - nvn_pl_protect protects again a page unprotected before the release;
   - TOTK_PIPELINE=2 (dry run) counts the write and does not wait.
   TOTK_PROBE_NEGATIVE=1: the fence never completes; the held write gives up after PL_HOLD_MS (a timeout) instead of
   hanging, which is the deadlock guard. */
#define NVN_DIRTY_NO_HANDLE_DEFAULT 1
#include "../../third_party/d2r_gfx/gfx/nvn/nvn_dirty.c"
#include <mach/mach.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ucontext.h>

static int (*g_fault)(void *, int, void *);
static void (*g_prep)(void *, size_t);
void dm_fault_hook_register(int (*fn)(void *, int, void *)) { g_fault = fn; }
void dm_guest_write_prep_register(void (*fn)(void *, size_t)) { g_prep = fn; }
static _Thread_local uint32_t t_handle;
uint32_t runtime_thread_current_handle(void) { return t_handle; }
static void on_signal(int sig, siginfo_t *si, void *uc) {
    int is_write = (int)((((ucontext_t *)uc)->uc_mcontext->__es.__esr >> 6) & 1);
    if (g_fault && g_fault(si->si_addr, is_write, uc)) return;
    signal(sig, SIG_DFL);
}
static _Atomic uint64_t g_done_fence;
static uint64_t done_fn(void) { return atomic_load(&g_done_fence); }
static uint64_t now_ms(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1000000ull; }
static int g_fail, g_checks;
#define EXPECT(name, want, got) do { long long w_ = (long long)(want), g_ = (long long)(got); g_checks++; \
    printf("%-70s expected %-6lld got %-6lld %s\n", name, w_, g_, w_ == g_ ? "ok" : "FAIL"); if (w_ != g_) g_fail++; } while (0)

static volatile uint8_t *g_mem; static size_t g_pg_sz;
static _Atomic int g_t_done; static uint32_t g_t_handle;
static void *writer(void *arg) { t_handle = g_t_handle; size_t off = (size_t)(uintptr_t)arg; g_mem[off] = 99; atomic_store(&g_t_done, 1); return NULL; }
static void *host_writer(void *arg) { t_handle = g_t_handle; size_t off = (size_t)(uintptr_t)arg; g_prep((void *)(g_mem + off), 16); g_mem[off] = 77; atomic_store(&g_t_done, 1); return NULL; }
/* 1 = the thread finished within wait_ms (before the fence was completed); then the fence completes and the thread is joined */
static int run_thread(void *(*fn)(void *), size_t off, uint32_t handle, int wait_ms, uint64_t complete_fence) {
    pthread_t t; atomic_store(&g_t_done, 0); g_t_handle = handle;
    pthread_create(&t, NULL, fn, (void *)(uintptr_t)off);
    uint64_t t0 = now_ms(); while (!atomic_load(&g_t_done) && now_ms() - t0 < (uint64_t)wait_ms) usleep(1000);
    int early = atomic_load(&g_t_done);
    if (complete_fence) atomic_store(&g_done_fence, complete_fence);
    pthread_join(t, NULL);
    return early;
}
static int page_writable(size_t off) {
    vm_address_t a = (vm_address_t)(g_mem + off); vm_size_t sz = 0; vm_region_basic_info_data_64_t info; mach_msg_type_number_t cnt = VM_REGION_BASIC_INFO_COUNT_64; mach_port_t obj;
    if (vm_region_64(mach_task_self(), &a, &sz, VM_REGION_BASIC_INFO_64, (vm_region_info_t)&info, &cnt, &obj) != KERN_SUCCESS) return -1;
    return (info.protection & VM_PROT_WRITE) ? 2 : (info.protection & VM_PROT_READ) ? 1 : 0;
}
static uintptr_t g_pages[64]; static int g_npages;
static void add_page(uintptr_t pg, void *ctx) { (void)ctx; if (g_npages < 64) g_pages[g_npages++] = pg; }

int main(int argc, char **argv) {
    int dry = argc > 1 && !strcmp(argv[1], "dry"), negative = getenv("TOTK_PROBE_NEGATIVE") && atoi(getenv("TOTK_PROBE_NEGATIVE"));
    setenv("TOTK_PIPELINE", dry ? "2" : "1", 1);
    struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_sigaction = on_signal; sa.sa_flags = SA_SIGINFO; sigaction(SIGSEGV, &sa, NULL); sigaction(SIGBUS, &sa, NULL);
    g_pg_sz = (size_t)getpagesize();
    g_mem = mmap(NULL, 16 * g_pg_sz, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    nvn_dirty_add_range((const void *)g_mem, 16 * g_pg_sz);
    nvn_pl_set_done(done_fn);
    EXPECT("mode", dry ? 2 : 1, nvn_pl_mode());
    const uint32_t KEEP = 8, PROBE = 3; uint32_t fr = 100; nvn_pl_set_frame(fr);
    atomic_store(&g_done_fence, 10);
    if (negative) {   // released fence 11 never completes: the held write must give up (timeout), not hang
        nvn_pl_live((const void *)(g_mem + 9 * g_pg_sz), 16, 11, 0, add_page, NULL);
        nvn_dirty_protect((const void *)(g_mem + 9 * g_pg_sz), g_pg_sz);
        nvn_pl_set_released(11, 10);
        uint64_t t0 = now_ms(); g_t_handle = 7; t_handle = 7; g_mem[9 * g_pg_sz] = 5; uint64_t dt = now_ms() - t0;
        uint64_t v[NVN_PLS_N]; nvn_pl_stats(v);
        EXPECT("negative: the held write timed out (1 timeout)", dry ? 0 : 1, v[NVN_PLS_TIMEOUT]);
        EXPECT("negative: it waited about PL_HOLD_MS (>= 400 ms)", dry ? 0 : 1, dt >= 400);
        printf("%d checks, %d failed\n", g_checks, g_fail); return g_fail != 0;
    }
    // page 0: first sight -> probation (protected, snapshot), then static after PROBE frames without writes
    EXPECT("first sight: snapshot (probation)", 1, nvn_pl_classify((const void *)g_mem, 64, 10, KEEP, PROBE, 1));
    EXPECT("first sight: page write-protected", 1, page_writable(0));
    nvn_pl_set_frame(fr + PROBE - 1);
    EXPECT("still on probation", 1, nvn_pl_classify((const void *)g_mem, 64, 10, KEEP, PROBE, 1));
    nvn_pl_set_frame(fr + PROBE);
    EXPECT("probation over without writes: static", 0, nvn_pl_classify((const void *)g_mem, 64, 10, KEEP, PROBE, 1));
    EXPECT("first=0: an unseen page is static", 0, nvn_pl_classify((const void *)(g_mem + 5 * g_pg_sz), 64, 10, KEEP, PROBE, 0));
    // a CPU write on the protected page (no GPU read in flight): not held, page volatile for KEEP frames
    fr += PROBE; nvn_pl_set_frame(fr);
    EXPECT("write with nothing in flight finishes at once", 1, run_thread(writer, 8, 7, 200, 0));
    EXPECT("written page: snapshot (volatile)", 1, nvn_pl_classify((const void *)g_mem, 64, 10, KEEP, PROBE, 1));
    nvn_pl_set_frame(fr + KEEP - 1);
    EXPECT("volatile until KEEP frames", 1, nvn_pl_classify((const void *)g_mem, 64, 10, KEEP, PROBE, 1));
    nvn_pl_set_frame(fr + KEEP);
    EXPECT("KEEP expired: probation again (snapshot)", 1, nvn_pl_classify((const void *)g_mem, 64, 10, KEEP, PROBE, 1));
    EXPECT("probation again: page protected", 1, page_writable(0));
    fr += KEEP; nvn_pl_set_frame(fr);
    // GPU write in flight (fence 11 > done 10): read in place
    nvn_pl_live((const void *)(g_mem + 2 * g_pg_sz), 32, 11, 1, NULL, NULL);
    EXPECT("GPU write in flight: read in place (-1)", -1, nvn_pl_classify((const void *)(g_mem + 2 * g_pg_sz), 64, 10, KEEP, PROBE, 1));
    EXPECT("after it completed: not -1", 1, nvn_pl_classify((const void *)(g_mem + 2 * g_pg_sz), 64, 11, KEEP, PROBE, 1) >= 0);
    // page 3 read in place by fence 11, released at 11 (done 10): a guest write waits until 11 completes
    g_npages = 0;
    nvn_pl_live((const void *)(g_mem + 3 * g_pg_sz), 64, 11, 0, add_page, NULL);
    EXPECT("pages noted for the release", 1, g_npages);
    EXPECT("release protects the noted page", 1, nvn_pl_protect(g_pages, g_npages, 10));
    EXPECT("protected page read-only", 1, page_writable(3 * g_pg_sz));
    nvn_pl_set_released(11, 10);
    EXPECT(dry ? "dry: guest write is not held" : "guest write held until fence 11", dry ? 1 : 0, run_thread(writer, 3 * g_pg_sz, 7, 150, 11));
    { uint64_t v[NVN_PLS_N]; nvn_pl_stats(v); EXPECT(dry ? "dry: counted as would-hold" : "counted as a hold", 1, dry ? v[NVN_PLS_WOULD] : v[NVN_PLS_HOLD]); }
    EXPECT("written through: byte stored", 99, g_mem[3 * g_pg_sz]);
    EXPECT("held page is volatile now", 1, nvn_pl_classify((const void *)(g_mem + 3 * g_pg_sz), 64, 11, KEEP, PROBE, 1));
    // page 4: read by 12 (released 12) and 13 (not released); done 11: a write waits until 12 only (pl_rf2)
    atomic_store(&g_done_fence, 11);
    nvn_pl_live((const void *)(g_mem + 4 * g_pg_sz), 16, 12, 0, NULL, NULL);
    nvn_pl_live((const void *)(g_mem + 4 * g_pg_sz), 16, 13, 0, NULL, NULL);
    nvn_dirty_protect((const void *)(g_mem + 4 * g_pg_sz), g_pg_sz);
    nvn_pl_set_released(12, 11);
    EXPECT(dry ? "dry: rf2 write not held" : "rf2: write held until 12", dry ? 1 : 0, run_thread(writer, 4 * g_pg_sz, 7, 150, 12));
    EXPECT("rf2: it waited for 12, not 13 (done = 12 now)", 12, (long long)atomic_load(&g_done_fence));
    // page 6: read only by 14 (not released; released 12, done 12): not held
    nvn_pl_live((const void *)(g_mem + 6 * g_pg_sz), 16, 14, 0, NULL, NULL);
    nvn_dirty_protect((const void *)(g_mem + 6 * g_pg_sz), g_pg_sz);
    EXPECT("page of an unreleased submission: write not held", 1, run_thread(writer, 6 * g_pg_sz, 7, 150, 0));
    // page 7: read by 13 released at 13, but a host thread (handle 0) writes: not held
    nvn_pl_live((const void *)(g_mem + 7 * g_pg_sz), 16, 13, 0, NULL, NULL);
    nvn_dirty_protect((const void *)(g_mem + 7 * g_pg_sz), g_pg_sz);
    nvn_pl_set_released(13, 12);
    EXPECT("host thread (handle 0): write not held", 1, run_thread(writer, 7 * g_pg_sz, 0, 150, 0));
    // page 8: host write announced through prep_write by a guest thread: held until 13
    nvn_pl_live((const void *)(g_mem + 8 * g_pg_sz), 16, 13, 0, NULL, NULL);
    nvn_dirty_protect((const void *)(g_mem + 8 * g_pg_sz), g_pg_sz);
    EXPECT(dry ? "dry: announced host write not held" : "announced host write held until 13", dry ? 1 : 0, run_thread(host_writer, 8 * g_pg_sz, 7, 150, 13));
    EXPECT("announced host write stored", 77, g_mem[8 * g_pg_sz]);
    // completed fence: page 10 read by 13, done 13: not held
    nvn_pl_live((const void *)(g_mem + 10 * g_pg_sz), 16, 13, 0, NULL, NULL);
    nvn_dirty_protect((const void *)(g_mem + 10 * g_pg_sz), g_pg_sz);
    EXPECT("completed submission: write not held", 1, run_thread(writer, 10 * g_pg_sz, 7, 150, 0));
    // re-protection: page 11 noted for 14, unprotected by a write before the release, protected again by nvn_pl_protect
    g_npages = 0;
    nvn_pl_live((const void *)(g_mem + 11 * g_pg_sz), 16, 14, 0, add_page, NULL);
    g_mem[11 * g_pg_sz] = 1;   // never protected: no fault
    EXPECT("page writable before the release", 2, page_writable(11 * g_pg_sz));
    EXPECT("release protects it again", 1, nvn_pl_protect(g_pages, g_npages, 13));
    EXPECT("page read-only after the release", 1, page_writable(11 * g_pg_sz));
    EXPECT("a page whose fence completed is not protected", 0, nvn_pl_protect(g_pages, g_npages, 14));
    EXPECT("nvn_pl_gpu_writes: GPU write in flight over page 2 (done 10)", 1, nvn_pl_gpu_writes((const void *)(g_mem + 2 * g_pg_sz), 16, 10));
    EXPECT("nvn_pl_gpu_writes: none once done 11", 0, nvn_pl_gpu_writes((const void *)(g_mem + 2 * g_pg_sz), 16, 11));
    { nvn_pl_hold_rec h[8]; int n = nvn_pl_holds(h, 8); EXPECT("hold records kept", 1, n >= 2); }
    if (dry) {   // TOTK_PIPELINE_RWATCH: a watched GPU-written page is inaccessible; the first read is recorded and the page restored
        uintptr_t pg = (uintptr_t)(g_mem + 12 * g_pg_sz), pg2 = (uintptr_t)(g_mem + 13 * g_pg_sz); uintptr_t two[2] = {pg2, pg};
        nvn_dirty_protect((const void *)pg2, g_pg_sz);   // the tracker wants page 13 read-only
        EXPECT("watch: two pages", 2, nvn_pl_watch(two, 2));
        EXPECT("watched page inaccessible", 0, page_writable(12 * g_pg_sz));
        volatile uint8_t x = g_mem[12 * g_pg_sz]; (void)x;
        uint64_t v[NVN_PLS_N]; nvn_pl_stats(v);
        EXPECT("watched read recorded", 1, v[NVN_PLS_WATCH_READ]);
        EXPECT("page accessible again (tracker: writable)", 2, page_writable(12 * g_pg_sz));
        EXPECT("watch over: the other page back to the tracker's read-only", 1, (nvn_pl_watch(NULL, 0), page_writable(13 * g_pg_sz)));
        g_prep((void *)(g_mem + 13 * g_pg_sz), 8); g_mem[13 * g_pg_sz] = 3;
        EXPECT("then a host write goes through", 3, g_mem[13 * g_pg_sz]);
    }
    printf("%d checks, %d failed\n", g_checks, g_fail);
    return g_fail != 0;
}
