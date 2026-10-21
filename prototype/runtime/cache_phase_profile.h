#ifndef UQ_CACHE_PHASE_PROFILE_H
#define UQ_CACHE_PHASE_PROFILE_H

#include <stdint.h>

/* Private diagnostic ABI for the projected attention source only. Durations
 * are host steady-clock nanoseconds, including the existing synchronization.
 * They are not device event or kernel-only times. A successful call has
 * complete=1 and host_checks_ns+validation_status_ns+reader_sync_ns=total_ns.
 * Every attention call replaces this thread's record, including refusals.
 * Sequence starts at one per thread and never wraps; exhaustion is refused.
 */
typedef struct uq_cache_phase_profile_v1 {
    uint64_t schema;
    uint64_t sequence;
    uint64_t complete;
    int64_t api_result;
    uint64_t host_checks_ns;
    uint64_t validation_status_ns;
    uint64_t reader_sync_ns;
    uint64_t total_ns;
} uq_cache_phase_profile_v1;

#ifdef __cplusplus
extern "C" {
#endif
/* Return 0 for a copied record, 1 for null/misaligned output, wrong exact size,
 * no attention call on this thread, or sequence exhaustion. Invalid calls do
 * not modify output. This getter performs no HIP calls or allocations.
 */
int uq_cache_phase_profile_last_v1(uq_cache_phase_profile_v1* out, uint64_t bytes);
#ifdef __cplusplus
}
#endif

#endif
