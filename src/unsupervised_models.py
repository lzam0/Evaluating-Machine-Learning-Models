from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_2d(points_2d, colors, title, out_path):
    plt.figure(figsize=(7, 6))
    scatter = plt.scatter(points_2d[:, 0], points_2d[:, 1],
                          c=colors, s=22, cmap="tab10")

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")

    # ADD COLORBAR HERE
    plt.colorbar(scatter, label="Cluster ID")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def labels_to_codes(labels: pd.Series) -> np.ndarray:
    # Converts A-J (or any labels) -> 0..n-1 codes for colouring
    return labels.astype("category").cat.codes.to_numpy()


def main():
    ROOT = Path(__file__).resolve().parents[1]
    CSV_PATH = ROOT / "data" / "cleaned_features" / "hand_landmarks_cleaned.csv"
    OUT_DIR = ROOT / "outputs"
    ensure_dir(OUT_DIR)

    df = pd.read_csv(CSV_PATH)

    # Keep labels for post-hoc evaluation / colouring (but DO NOT use in clustering)
    y_true = df["label"] if "label" in df.columns else None

    # Build feature matrix (drop id/label if present)
    drop_cols = [c for c in ["instance_id", "label"] if c in df.columns]
    X = df.drop(columns=drop_cols, errors="ignore")

    # Sanity check
    if X.shape[1] == 0:
        raise ValueError("No feature columns found after dropping instance_id/label.")

    # Scale features (critical for clustering + t-SNE)
    X_scaled = StandardScaler().fit_transform(X.to_numpy())

    # Choose k
    k = int(y_true.nunique()) if y_true is not None else 10

    # ---------------------------
    # 1) Clustering (full 63D)
    # ---------------------------
    kmeans_labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X_scaled)
    hier_labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_scaled)

    # ---------------------------
    # 2) Metrics
    # ---------------------------
    metrics = {
        "dataset": str(CSV_PATH),
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "k": k,
        "kmeans": {},
        "hierarchical": {},
    }

    # Silhouette (unsupervised effectiveness)
    metrics["kmeans"]["silhouette"] = float(silhouette_score(X_scaled, kmeans_labels))
    metrics["hierarchical"]["silhouette"] = float(silhouette_score(X_scaled, hier_labels))

    # Post-hoc (compare clusters to true labels)
    if y_true is not None:
        metrics["kmeans"]["ARI"] = float(adjusted_rand_score(y_true, kmeans_labels))
        metrics["kmeans"]["NMI"] = float(normalized_mutual_info_score(y_true, kmeans_labels))
        metrics["hierarchical"]["ARI"] = float(adjusted_rand_score(y_true, hier_labels))
        metrics["hierarchical"]["NMI"] = float(normalized_mutual_info_score(y_true, hier_labels))
    else:
        metrics["kmeans"]["ARI"] = None
        metrics["kmeans"]["NMI"] = None
        metrics["hierarchical"]["ARI"] = None
        metrics["hierarchical"]["NMI"] = None

    # Save metrics JSON
    (OUT_DIR / "unsupervised_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Save cluster assignments
    out_df = pd.DataFrame()
    if "instance_id" in df.columns:
        out_df["instance_id"] = df["instance_id"]
    if y_true is not None:
        out_df["true_label"] = y_true

    out_df["kmeans_cluster"] = kmeans_labels
    out_df["hier_cluster"] = hier_labels
    out_df.to_csv(OUT_DIR / "cluster_assignments.csv", index=False)

    # ---------------------------
    # 3) PCA 2D (VISUALISATION ONLY)
    # ---------------------------
    pca = PCA(n_components=2, random_state=42)
    X_pca_2d = pca.fit_transform(X_scaled)
    (OUT_DIR / "pca_explained_variance.txt").write_text(
        f"PCA 2D explained variance sum: {pca.explained_variance_ratio_.sum():.4f}\n"
        f"PC1: {pca.explained_variance_ratio_[0]:.4f}\n"
        f"PC2: {pca.explained_variance_ratio_[1]:.4f}\n",
        encoding="utf-8"
    )

    # Plot PCA coloured by clusters
    plot_2d(X_pca_2d, kmeans_labels, "K-Means Clusters (PCA 2D)", OUT_DIR / "kmeans_pca2d.png")
    plot_2d(X_pca_2d, hier_labels, "Hierarchical Clusters (PCA 2D)", OUT_DIR / "hier_pca2d.png")

    # Plot PCA coloured by true labels (if available)
    if y_true is not None:
        plot_2d(X_pca_2d, labels_to_codes(y_true), "True Labels (PCA 2D)", OUT_DIR / "true_pca2d.png")

    # ---------------------------
    # 4) t-SNE 2D (OPTIONAL / NON-LINEAR VIS)
    # ---------------------------
    # t-SNE is expensive. Using init="pca" improves stability.
    # Perplexity must be < n_samples; we adapt automatically.
    n = X_scaled.shape[0]
    perplexity = min(30, max(5, (n - 1) // 3))  # safe-ish default

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=42
    )
    X_tsne_2d = tsne.fit_transform(X_scaled)

    plot_2d(X_tsne_2d, kmeans_labels, f"K-Means Clusters (t-SNE, perplexity={perplexity})", OUT_DIR / "kmeans_tsne2d.png")
    plot_2d(X_tsne_2d, hier_labels, f"Hierarchical Clusters (t-SNE, perplexity={perplexity})", OUT_DIR / "hier_tsne2d.png")
    if y_true is not None:
        plot_2d(X_tsne_2d, labels_to_codes(y_true), f"True Labels (t-SNE, perplexity={perplexity})", OUT_DIR / "true_tsne2d.png")

    print("✅ Part 2d done.")
    print(f"- Metrics: {OUT_DIR / 'unsupervised_metrics.json'}")
    print(f"- Assignments: {OUT_DIR / 'cluster_assignments.csv'}")
    print(f"- Plots: {OUT_DIR}/*.png")


if __name__ == "__main__":
    main()
