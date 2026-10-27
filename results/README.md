# Experiment evidence

- [Measured report](REPORT.md) and [complete metrics](metrics.json): 1,024 held-out examples, both 8,000-update schedules, hashes and paired controls.
- [Training verification](training-verification.json): applied optimizer counts and paired RNG/data progress.
- [CPU tests](test-output.txt): 34 passed.
- [API smoke checks](smoke-test.json): trained inference, reproducibility and compact-weight fallback.
- [Environment](environment.json): actual package and CUDA versions.
- [Browser QA](browser-qa.json) and [screenshots](screenshots/): desktop/mobile interaction evidence.
- [Visual assessment and limitations](QUALITY.md).

Sample grids, GIFs, generation/forward trajectories and microscope inspections are genuine exports from the trained EMA models. Run the evaluator again after changing checkpoints.
