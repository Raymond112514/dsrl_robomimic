import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def fit_action_basis(data, n_components):
	"""Fit PCA on (N, D) actions. Returns V (D, k), per-PC explained variance."""
	centered = data - data.mean(axis=0)
	k = min(n_components, centered.shape[0], centered.shape[1])
	pca = PCA(n_components=k)
	pca.fit(centered)
	return pca.components_.T.astype(np.float32), pca.explained_variance_ratio_.astype(
		np.float32
	)


def cumulative_explained_variance(data, max_components=None):
	"""Return (n_components, cumulative_variance) for plotting."""
	centered = data - data.mean(axis=0)
	k = centered.shape[1]
	if max_components is not None:
		k = min(k, max_components, centered.shape[0])
	pca = PCA(n_components=k)
	pca.fit(centered)
	return np.arange(1, k + 1), np.cumsum(pca.explained_variance_ratio_)


def save_action_basis(path, V, explained_variance_ratio, **metadata):
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	V = np.asarray(V, dtype=np.float32)
	explained_variance_ratio = np.asarray(explained_variance_ratio, dtype=np.float32)
	np.savez_compressed(
		path,
		V=V,
		explained_variance_ratio=explained_variance_ratio,
		n_components=np.array([V.shape[1]], dtype=np.int32),
		feature_dim=np.array([V.shape[0]], dtype=np.int32),
	)
	meta = {
		"n_components": int(V.shape[1]),
		"feature_dim": int(V.shape[0]),
		"total_explained_variance": float(explained_variance_ratio.sum()),
		"explained_variance_ratio": explained_variance_ratio.tolist(),
		**metadata,
	}
	path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load_action_basis(path, n_basis=None):
	path = Path(path)
	if not path.is_file():
		raise FileNotFoundError(f"Basis not found: {path}")
	with np.load(path) as data:
		V = np.asarray(data["V"], dtype=np.float32)
		explained = np.asarray(data["explained_variance_ratio"], dtype=np.float32)
	if n_basis is not None:
		k = min(n_basis, V.shape[1])
		V = V[:, :k]
		explained = explained[:k]
	return V, explained
