# Changelog

## 2.0.0 (2026-11-22)

Restructured release for the revised manuscript. All reported results were regenerated with this version.

### Changed
- All experiment code moved into one shared module, `code/rareguard_lib.py`; the notebooks contain only the experiment loops.
- All settings moved into `config.json`.
- Notebooks renumbered 00-09, one per experiment group (E1-E9), plus statistics (08) and manuscript figures and tables (09).
- Train/validation/test splits are created once (notebook 00), saved, and reused. They select the same rows as version 1.0.0.
- E2-E5 now reuse the component scores saved by E1 instead of retraining, so fusion rules and thresholds are compared on identical scores.
- In E1 the single-detector configurations come from the same fit as the full pipeline.
- Statistics: paired t-tests with Holm correction for every experiment family, in one file (`results/stats/paired_tests.csv`).

### Added
- `run_info.json` for every experiment: date, duration, device, package versions, full configuration, and MD5 hashes of all output files.
- `logs/run_log.csv` (one line per notebook run).
- `manuscript/provenance.csv`, mapping each figure and table to its source file and run.
- `tools/verify_results.py`, which checks the archived results without the raw data.
- Automatic archiving of replaced result files.
- E2 (fusion baselines, including rank fusion) and E6 (training-reference contamination) are fully specified in code. The procedures are documented at the top of notebooks 02 and 04.

### Results
- Numbers differ slightly from version 1.0.0 because of the rerun (different Colab software versions and GPU non-determinism in the autoencoder) and because E2 and E6 were re-implemented. The conclusions are unchanged. The values reported in the revised manuscript are those in `results/`.

## 1.0.0 (2026-08-08)

First public release: one reproducibility notebook, result files and figures for the first version of the study (Zenodo DOI 10.5281/zenodo.21852872).
