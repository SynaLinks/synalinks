# NOTES

Scratchpad for the autosolve loop — the *qualitative* companion to
`results.tsv` (which holds the numbers). Keep it short; prune as you go.

## Current best

- **dev_reward:** _(none yet — run the baseline)_
- **commit:** _(short hash)_
- **what it is:** _(one line: model, program shape, key levers)_

## Ideas to try (backlog)

One line each — the lever and the bet. Move to the log once run.

- [ ] _e.g. INSTRUCTIONS: spell out "show arithmetic, then state final number"_
- [ ] _e.g. add a verify→repair module (re-derive the answer, retry if it disagrees)_
- [ ] _e.g. reward ExactMatch → numeric tolerance (off-by-formatting misses?)_

## Experiment log

Newest first. One block per experiment — the why and the verdict, not the score.

### YYYY-MM-DD — <hypothesis> — keep | revert | crash
- **Hypothesis:** _one sentence._
- **Falsifier:** _what dev_reward would prove me wrong._
- **Result:** _dev/test reward; did the falsifier fire?_
- **Verdict & why:** _kept / reverted / crashed, and what it taught me._

## Observations (from actual failing predictions)

Patterns in what the program gets *wrong* — re-read 3–5 real failures in
`predictions_dev.jsonl`, not scores.

- _e.g. answers are right but returned as "42.0" vs expected 42 → formatting._

## Dead ends

What didn't work, so you don't retry it. Include the *why*.

- _e.g. longer instructions: no gain over 3 rewordings — hypothesis was wrong, not the wording._
