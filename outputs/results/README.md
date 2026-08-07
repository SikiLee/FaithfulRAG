# Experiment outputs

`repro.run_experiment` writes each run to:

`outputs/results/<dataset>/<method>/<run_id>/`

Each completed run contains the resolved config, environment manifest, dataset
manifest, raw predictions, parsed predictions, metrics, a one-sample execution
trace, intermediate checkpoints when applicable, and `run.log`. Failed runs also
contain `error.json`.

Run `python -m repro.collect_results` to rebuild `summary.csv` and `summary.md`.
