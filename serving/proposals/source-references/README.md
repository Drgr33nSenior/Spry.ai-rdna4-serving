# Protected source references

This proposal produces the same 78-member package as the
[post-load proposal](../post-load/README.md). It references three unchanged
protected files in that proposal instead of adding another review copy.
The protected set, runtime bytes, bindings, requests and workload arguments
remain identical.

The proposer accepts references only for the existing nine protected roles.
Each reference must be relative, name the expected file and resolve to a
canonical regular file within the local proposal tree. The existing source
hash, size limit and protected-set check still apply. Changed, missing,
linked, external or ambiguous copy/reference sources refuse before output.
Historical proposals retain their existing copy and owner-reference behavior.

`SOURCE.patch` records the proposer change. `D2.patch` and `RUNTIME.patch` are
empty because this increment changes no protected or runtime source. The CPU
checks compare every assembled package member with the post-load control and
exercise archive admission and worker hook loading. The complete transport
proof for those identical bytes is retained from the post-load verification.

`VERIFICATION.json` records the observed checks. This source proposal grants no
execution authority. The parent runtime still requires native qualification;
metadata preparation, startup scan consolidation and memory recovery remain.
