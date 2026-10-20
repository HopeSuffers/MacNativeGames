// TOTK runtime shader translation (totk_shader_rt.m).
#pragma once
#ifdef __OBJC__
#import <Foundation/Foundation.h>
#include <stdint.h>
NSString *totk_rt_dir(void);
// Returns the variant name whose <dir>/<variant>_b.{metal,json} match the answers, translating when needed.
// stage: NVN shader stage (0 vertex, 1 fragment, 5 compute). answers: {"tex","img","attr","fout"} dictionaries.
NSString *totk_rt_variant(NSString *shader, int stage, const uint8_t *control, const uint8_t *code, NSDictionary *answers, NSArray *local);   // nil on iOS when the variant is not translated yet (request recorded); on the Mac while it is translated in the background
// Called on the background queue with "<variant>_b" and the NVN stage once a variant is translated: compile its library there.
void totk_rt_set_ready_hook(void (*hook)(NSString *name, int stage));
#endif
#include <stdint.h>
/* Variants new to the translation cache this run, the variants in the cache (-1 until counted in the background) and
   the CLOCK_UPTIME_RAW time of the last new one (0: none yet). Any pointer may be NULL. */
void totk_rt_counts(uint64_t *fresh, int64_t *total, uint64_t *last_new_ns);
/* Moves on each time a background translation finishes: callers that got nil retry when it changed. */
uint64_t totk_rt_ready_gen(void);
/* Lookups answered "not ready yet" (draws skipped instead of a stall) this run. */
uint64_t totk_rt_async_skips(void);
