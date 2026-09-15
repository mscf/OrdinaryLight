# Release follow-up: approved quality baseline

The user explicitly approved updating the quality gates to the current baseline
and completing commit, push, and release. This supersedes the initial held status.

- The six-scene noise baseline now uses the captured candidate metrics and GPU medians, with the existing tolerance policies unchanged. Provenance points to the candidate and published-release control reports.
- The small-emitter, 320×180, eight-bounce, roulette-start-four capture uses its accepted low-frequency noise ratio (1.2439681283), with 15% relative headroom (limit 1.4305633475). A checked-in baseline applies only when every recorded capture setting matches. Other configurations retain 1.15; explicit CLI overrides still work. Error, temporal, and absolute-bias limits are unchanged.
- Fresh GPU reruns of both previously failing gates passed. Logs: `noise-accepted.log`, `termination-accepted.log`, `accepted-gates.json`.
- The complete hardware-independent suite passed: 908 tests and 85 subtests, with 233 opt-in GPU skips. The scope/bias regression protects against applying the accepted exception to other scenes, resolutions, strategies, or bounce configurations.
- The earlier full GPU run, supplemental compiler/browser checks, 8/8 corrected 4K presentation gates, shader validation, and clean consumer checks remain applicable. Only the baseline policy and its test changed after that run; no renderer/shader behavior was changed to pass these gates.

OrdinaryShade 0.1.0a8 and OrdinaryLight 0.7.0 release notes document the features
and explicit baseline acceptance. The original candidate distribution hashes
are retained as pre-acceptance evidence; release distributions are rebuilt from
the committed source trees.
