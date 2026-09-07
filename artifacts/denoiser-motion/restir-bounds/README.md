# ReSTIR bounds-guard experiment

Two reservoir streams reproduce the specialized temporal failure after four
frames, so it is not limited to four streams.

Source inspection found the host allocation and pixel*stream_count+stream
index expression consistent. Area-light evaluation checks light indices.
Direct reservoir buffer helpers did not check reservoir_index against the
runtime SSBO length.

The candidate patch adds:
- previous-buffer load guard: index >= previous_words.length()/3 returns empty;
- current-buffer store guard: index >= current_words.length()/3 returns.

Production variants were compiled into /tmp/restir-bounds-spv and the test
harness redirected matching shader resource reads there. Packaged binaries
were not replaced. The shared GLSL source was restored after the experiment;
candidate.patch preserves the exact edit for further instrumentation.

## Result

Specialized temporal ReSTIR with four streams completed 124 normal-paced,
animated Qt frames at 1280x720 and exited cleanly with the guards. Unguarded
two-stream testing failed after four frames. The existing unguarded
four-stream failure was established in the preceding investigation.

This is a candidate fix, not proof of an invalid access. Adding runtime length
checks changes generated SPIR-V and driver compilation; the successful run
does not tell us whether either guard fired. No general performance or
stability guarantee is claimed.

Next add explicit guard-hit instrumentation, then verify descriptor ranges
and actual indices if hits occur. If neither guard fires, investigate code
generation rather than asserting an allocation bug. Keep the scene's ReSTIR
workaround until there is a verified fix and regression coverage.

No production shader or showcase setting changed in this experiment.
