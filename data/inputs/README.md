# data/inputs — frozen artefacts used by the platform during the runs

Delivered **as used**, not regenerated (decision 2026-10-12). Read-only.

| File | md5 | Origin | Used by |
|---|---|---|---|
| `pca_model_norm6.npz` | `b904ea7041242d8affcd64653aaf408b` | `2FAST_ML/PCA/pca_model_norm6.npz` (2026-06-22); logged in `raw/S24-M01/S24-M01_parameters.txt` as "PCA decomposition file …/pca_model_norm6.npz" | PCA-cost BO runs (S24-M01 = Fig 4 / S13; S19-M04 = S18). 6 components, max-normalised, 310–1000 nm (1218 points). `toofast_analysis.bo.pca_cosine_cost` with k = 4 reproduces the 9 costs of S24-M01 exactly (`tests/test_fig04.py`). |
| `pca_model_5.npz` | `155ed0167e17146e1aae5c7d8b75bf8e` | `2FAST_ML/PCA/pca_model_5.npz`; logged as "PCA decomposition file …/pca_model_5.npz" in the `_parameters.txt` of S22-M03, S25-M03, S28-M02 | PCA-cosine BO runs of S19, S21, S22 (5 components, `normalize = none`, cost on the first 4 scores); the original figure notebook reproduces their costs.csv exactly (ratio 1.0000 ± 0.0000). |
| `prior_kernel_pca_model_norm6_cosine_k4.pkl` | `b467841779642866d847d6af2e7bdd24` | `2FAST_ML/PCA/prior_kernel_pca_model_norm6_cosine_k4.pkl` (2026-07-03, `kernel_benchmark_pca.ipynb`) | Initial GP kernel of S24-M01 ("PCA-pretrained prior"): `1.04**2 * Matern(length_scale=1.1, nu=2.5) + WhiteKernel(noise_level=0.00556)`, identical to the initial kernel logged in `S24-M01_parameters.txt`. sklearn pickle. |
