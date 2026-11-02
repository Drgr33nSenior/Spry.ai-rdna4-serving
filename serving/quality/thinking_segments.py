"""Separate authenticated thinking/final token vectors without decoding or execution."""
def split_thinking_ids(ids):
    if not (type(ids) is list and ids and all(type(x) is int and x>=0 for x in ids)):
        raise ValueError('nonempty integer output vector required')
    if ids.count(248069)!=1:
        raise ValueError('exactly one closing thinking marker required')
    index=ids.index(248069)
    tail=ids[index+1:]
    if any(x in (248068,248069) for x in tail):
        raise ValueError('thinking marker after final boundary')
    final_ids=[x for x in tail if x not in (248046,248044)]
    if not final_ids:
        raise ValueError('final answer vector is empty')
    return dict(reasoning_token_count=index,closing_marker_index=index,final_ids=final_ids,
                final_segment_ids=tail,final_answer_token_count=len(final_ids),output_tokens=len(ids))
