"""W5 time and durable-byte arithmetic, independent of execution authority."""
from .canon import need

TIME_KEYS = ('client_seconds', 'worker_startup_seconds', 'owner_cleanup_seconds',
             'arm_release_seconds', 'admission_seconds', 'source_preparation_seconds',
             'locked_admission_seconds', 'collection_reserve_seconds', 'campaign_margin_seconds',
             'seal_reserve_seconds', 'caller_reserve_seconds')
BYTE_KEYS = ('result_json_bytes', 'client_bundle_bytes', 'native_bundle_bytes', 'http_stream_bytes',
             'client_tree_bytes', 'native_tree_bytes', 'arm_metadata_bytes', 'source_copies_bytes',
             'observer_copies_bytes', 'outer_stdio_bytes', 'summary_closeout_bytes',
             'small_metadata_bytes', 'session_bytes', 'settings_bytes', 'http_member_read_bytes')
COUNT_KEYS = ('client_tree_files', 'native_tree_files')
HTTP_ARTIFACT_KEYS = tuple(key for key in BYTE_KEYS if key != 'settings_bytes') + COUNT_KEYS


def derive(primaries):
    need(type(primaries) is dict and set(primaries) == set(TIME_KEYS + BYTE_KEYS + COUNT_KEYS),
         'exact primary limits required; derived values cannot be supplied')
    need(all(type(value) is int and 0 < value < 2**63 for value in primaries.values()),
         'positive bounded integer limits required')
    p = primaries
    worker = p['client_seconds'] + p['worker_startup_seconds']
    work = worker + p['owner_cleanup_seconds']
    envelope = work + p['arm_release_seconds']
    active = (2 * envelope + p['admission_seconds'] + p['source_preparation_seconds']
              + p['locked_admission_seconds'] + p['collection_reserve_seconds'] + p['campaign_margin_seconds'])
    total = active + p['seal_reserve_seconds']
    payload = p['result_json_bytes'] + p['client_bundle_bytes'] + p['native_bundle_bytes']
    per_arm = (2 * p['http_stream_bytes'] + payload + p['client_tree_bytes']
               + p['native_tree_bytes'] + p['arm_metadata_bytes'])
    session = 2 * per_arm + sum(p[key] for key in ('source_copies_bytes', 'observer_copies_bytes',
                                                 'outer_stdio_bytes', 'summary_closeout_bytes', 'small_metadata_bytes'))
    need(session < p['session_bytes'], 'durable partitions exceed session ceiling')
    return dict(worker_seconds=worker, arm_work_seconds=work, arm_envelope_seconds=envelope,
                active_seconds=active, total_seconds=total, caller_seconds=total + p['caller_reserve_seconds'],
                http_payload_bytes=payload, http_session_max_bytes=session)
