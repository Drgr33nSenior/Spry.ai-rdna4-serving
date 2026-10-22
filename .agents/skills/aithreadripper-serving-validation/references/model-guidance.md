# Model guidance and reported limitations

Checked on 21 October 2026. Recheck the linked sources before relying on a later
revision; keep active experiments bound to their frozen requests and assets.

The [official Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices)
supports both modes. Its thinking sampling recipe uses `temperature=1.0`, `top_p=0.95`
and `top_k=20`; nonthinking uses `temperature=0.7`, `top_p=0.80`, `top_k=20` and
`presence_penalty=1.5`. These differ from the selected greedy validation profile. Record that
difference; do not silently change sampling during a matched experiment.

[Qwen issue #236](https://github.com/QwenLM/Qwen3.8/issues/236) reports degraded
long-prompt tool calling with thinking disabled in BF16 and FP8. The reporter
observed improvement with thinking enabled. An inactivity bot closed the issue;
that is neither a demonstrated fix nor maintainer confirmation of a general
nonthinking defect. Tool routing differs from the local ledger task, so this
report does not establish the cause of the ledger errors.

Keep external reports separate from authenticated local evidence. The local
matched J01 tests support thinking for that task. The [combined thinking quality record](../../../../provenance/SPRINT7-RINGFOLD-THINKING-QUALITY-20261021.json)
and [lifecycle record](../../../../provenance/SPRINT7-LIFECYCLE-20261021.json)
provide their own bounded coverage. Neither the external report nor these
local screens prove every nonthinking request broken or every thinking request
correct.
