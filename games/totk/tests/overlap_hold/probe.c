/* CPU only: TOTK_OVERLAP page holds (nvn_dirty.c nvn_hold_*), the part of the early frame release that keeps the CPU off
   guest memory the GPU's last command buffer still uses. A fake "completed fence" stands in for the GPU.
   - PROT_NONE hold: a read from another thread waits until the fence completes, then sees the bytes; a host write
     announced through the write-prepare hook waits the same way.
   - PROT_READ hold: reads go through at once, a write waits for the fence.
   - A hold over a page the dirty tracker keeps read-only: when the hold ends the page is read-only again (not writable),
     and the next write is recorded by the tracker as dirty.
   - Two holds on one page (read-only + no access): ending the stricter one leaves the page read-only.
   - nvn_hold_end ends only the holds of completed fences.
   TOTK_PROBE_NEGATIVE=1: the fence never completes; a read of a PROT_NONE page gives up after HOLD_WAIT_MS (counted as
   a timeout) instead of hanging, which is the deadlock guard. */
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
static void on_signal(int sig, siginfo_t *si, void *uc) {   // as the runtime: the hook decides "run again", else crash
    int is_write = (int)((((ucontext_t *)uc)->uc_mcontext->__es.__esr >> 6) & 1);
    if (g_fault && g_fault(si->si_addr, is_write, uc)) return;
    signal(sig, SIG_DFL);
}
static _Atomic uint64_t g_done_fence;
static uint64_t done_fn(void) { return atomic_load(&g_done_fence); }
static uint64_t now_ms(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1000000ull; }

static int g_fail, g_checks;
#define EXPECT(name, want, got) do { long long w_ = (long long)(want), g_ = (long long)(got); g_checks++; \
    printf("%-64s expected %-6lld got %-6lld %s\n", name, w_, g_, w_ == g_ ? "ok" : "FAIL"); if (w_ != g_) g_fail++; } while (0)

static volatile uint8_t *g_mem; static size_t g_pg_sz;
static _Atomic int g_t_done; static _Atomic long long g_t_val;
static void *reader(void *arg) { size_t off = (size_t)(uintptr_t)arg; atomic_store(&g_t_val, g_mem[off]); atomic_store(&g_t_done, 1); return NULL; }
static void *writer(void *arg) { size_t off = (size_t)(uintptr_t)arg; g_mem[off] = 99; atomic_store(&g_t_done, 1); return NULL; }
static void *host_writer(void *arg) { size_t off = (size_t)(uintptr_t)arg; g_prep((void *)(g_mem + off), 16); g_mem[off] = 77; atomic_store(&g_t_done, 1); return NULL; }
static int run_thread(void *(*fn)(void *), size_t off, int wait_ms, uint64_t complete_fence) {   // 1 = it finished before the fence completed
    pthread_t t; atomic_store(&g_t_done, 0);
    pthread_create(&t, NULL, fn, (void *)(uintptr_t)off);
    uint64_t t0 = now_ms(); while (!atomic_load(&g_t_done) && now_ms() - t0 < (uint64_t)wait_ms) usleep(1000);
    int early = atomic_load(&g_t_done);
    if (complete_fence) atomic_store(&g_done_fence, complete_fence);
    pthread_join(t, NULL);
    return early;
}
static int page_writable(size_t off) {   // probe the protection from the kernel's view (mincore cannot; use vm_region)
    vm_address_t a = (vm_address_t)(g_mem + off); vm_size_t sz = 0; vm_region_basic_info_data_64_t info; mach_msg_type_number_t cnt = VM_REGION_BASIC_INFO_COUNT_64; mach_port_t obj;
    if (vm_region_64(mach_task_self(), &a, &sz, VM_REGION_BASIC_INFO_64, (vm_region_info_t)&info, &cnt, &obj) != KERN_SUCCESS) return -1;
    return (info.protection & VM_PROT_WRITE) ? 2 : (info.protection & VM_PROT_READ) ? 1 : 0;
}

int main(void) {
    int negative = getenv("TOTK_PROBE_NEGATIVE") && atoi(getenv("TOTK_PROBE_NEGATIVE"));
    struct sigaction sa; memset(&sa, 0, sizeof sa); sa.sa_sigaction = on_signal; sa.sa_flags = SA_SIGINFO; sigaction(SIGSEGV, &sa, NULL); sigaction(SIGBUS, &sa, NULL);
    g_pg_sz = (size_t)getpagesize();
    g_mem = mmap(NULL, 6 * g_pg_sz, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    for (int i = 0; i < 6; i++) g_mem[i * g_pg_sz] = (uint8_t)(10 + i);
    nvn_dirty_add_range((const void *)g_mem, 6 * g_pg_sz);   // the tracker (registers the hooks)
    nvn_hold_set_done(done_fn);
    if (negative) {
        nvn_hold_begin((const void *)g_mem, 16, 1, PROT_NONE);
        uint64_t t0 = now_ms(); int early = run_thread(reader, 0, 2000, 0); uint64_t ms = now_ms() - t0;
        uint64_t v[6]; nvn_hold_stats(v);
        EXPECT("negative: read of a held page whose fence never completes ends", 1, early);
        EXPECT("negative: ... after the wait limit (timeouts)", 1, (long long)v[3]);
        EXPECT("negative: ... and sees the bytes", 10, atomic_load(&g_t_val));
        printf("negative: waited %llu ms\n", (unsigned long long)ms);
        printf("{\"status\":\"%s\",\"checks\":%d,\"failed\":%d}\n", g_fail ? "fail" : "pass", g_checks, g_fail);
        return g_fail != 0;
    }
    // 1. PROT_NONE: a read waits for the fence
    EXPECT("hold page 0 (no access) until fence 5", 1, nvn_hold_begin((const void *)g_mem, 16, 5, PROT_NONE));
    EXPECT("page 0 not readable while held", 0, page_writable(0));
    EXPECT("read of page 0 did not finish before fence 5", 0, run_thread(reader, 0, 60, 5));
    EXPECT("... and read the bytes after it", 10, atomic_load(&g_t_val));
    EXPECT("page 0 writable again after the faulting reader ended the hold", 2, page_writable(0));
    // 2. PROT_READ: reads at once, writes wait
    EXPECT("hold page 1 (read only) until fence 6", 1, nvn_hold_begin((const void *)(g_mem + g_pg_sz), 16, 6, PROT_READ));
    EXPECT("read of page 1 finished at once", 1, run_thread(reader, g_pg_sz, 60, 0));
    EXPECT("write to page 1 did not finish before fence 6", 0, run_thread(writer, g_pg_sz, 60, 6));
    EXPECT("... and landed after it", 99, g_mem[g_pg_sz]);
    // 3. host write through the prepare hook waits
    EXPECT("hold page 2 (no access) until fence 7", 1, nvn_hold_begin((const void *)(g_mem + 2 * g_pg_sz), 16, 7, PROT_NONE));
    EXPECT("host write into page 2 did not finish before fence 7", 0, run_thread(host_writer, 2 * g_pg_sz, 60, 7));
    EXPECT("... and landed after it", 77, g_mem[2 * g_pg_sz]);
    // 4. tracker read-only page under a hold: back to read-only, then a write is recorded as dirty
    nvn_dirty_protect((const void *)(g_mem + 3 * g_pg_sz), g_pg_sz);
    uint64_t stamp = nvn_dirty_now();
    EXPECT("tracker made page 3 read-only", 1, page_writable(3 * g_pg_sz));
    EXPECT("hold page 3 (no access) until fence 8", 1, nvn_hold_begin((const void *)(g_mem + 3 * g_pg_sz), 16, 8, PROT_NONE));
    EXPECT("page 3 not readable while held", 0, page_writable(3 * g_pg_sz));
    atomic_store(&g_done_fence, 8); nvn_hold_end(8);
    EXPECT("page 3 read-only again after the hold (tracker state)", 1, page_writable(3 * g_pg_sz));
    EXPECT("no write to page 3 yet: clean", 0, nvn_dirty_since((const void *)(g_mem + 3 * g_pg_sz), 16, stamp));
    g_mem[3 * g_pg_sz] = 42;
    EXPECT("a write to page 3 is seen by the tracker", 1, nvn_dirty_since((const void *)(g_mem + 3 * g_pg_sz), 16, stamp));
    // 5. two holds on one page: ending the stricter one leaves it read-only
    nvn_hold_begin((const void *)(g_mem + 4 * g_pg_sz), 16, 9, PROT_READ);
    nvn_hold_begin((const void *)(g_mem + 4 * g_pg_sz), 16, 10, PROT_NONE);
    EXPECT("page 4 under read-only + no-access holds: no access", 0, page_writable(4 * g_pg_sz));
    atomic_store(&g_done_fence, 9); nvn_hold_end(9);
    EXPECT("fence 9 done: still no access (fence 10 hold)", 0, page_writable(4 * g_pg_sz));
    atomic_store(&g_done_fence, 10); nvn_hold_end(10);
    EXPECT("fence 10 done: writable", 2, page_writable(4 * g_pg_sz));
    nvn_hold_begin((const void *)(g_mem + 4 * g_pg_sz), 16, 12, PROT_READ);
    nvn_hold_begin((const void *)(g_mem + 4 * g_pg_sz), 16, 11, PROT_NONE);
    atomic_store(&g_done_fence, 11); nvn_hold_end(11);
    EXPECT("no-access hold of fence 11 ended, read-only of 12 stays", 1, page_writable(4 * g_pg_sz));
    atomic_store(&g_done_fence, 12); nvn_hold_end(12);
    EXPECT("fence 12 done: writable", 2, page_writable(4 * g_pg_sz));
    // 6. a hold whose fence completed before it was set up does not stay
    nvn_hold_begin((const void *)(g_mem + 5 * g_pg_sz), 16, 3, PROT_NONE);
    EXPECT("hold of an already completed fence ends at once", 2, page_writable(5 * g_pg_sz));
    uint64_t v[6]; nvn_hold_stats(v);
    EXPECT("no wait timed out", 0, (long long)v[3]);
    printf("holds %llu, pages %llu, CPU accesses that waited %llu, host writes that waited %llu\n", (unsigned long long)v[0], (unsigned long long)v[1], (unsigned long long)v[2], (unsigned long long)v[4]);
    printf("{\"status\":\"%s\",\"checks\":%d,\"failed\":%d}\n", g_fail ? "fail" : "pass", g_checks, g_fail);
    return g_fail != 0;
}
