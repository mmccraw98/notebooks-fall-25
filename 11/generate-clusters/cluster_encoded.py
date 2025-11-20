import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
)
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def load_data():
    encoded = np.load("encoded.npy")
    major_clade_labels = np.load("major_clade_labels.npy")
    order_labels = np.load("order_labels.npy")
    return encoded, major_clade_labels, order_labels


def evaluate_kmeans(encoded, major_labels, order_labels, k_values):
    results = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(encoded)

        sil = silhouette_score(encoded, cluster_labels)

        nmi_major = normalized_mutual_info_score(major_labels, cluster_labels)
        nmi_order = normalized_mutual_info_score(order_labels, cluster_labels)

        ari_major = adjusted_rand_score(major_labels, cluster_labels)
        ari_order = adjusted_rand_score(order_labels, cluster_labels)

        # Combined score: encourage good shape clustering + alignment with both label granularities
        combined = sil + 0.5 * nmi_major + 0.5 * nmi_order

        results.append(
            {
                "k": k,
                "silhouette": sil,
                "nmi_major": nmi_major,
                "nmi_order": nmi_order,
                "ari_major": ari_major,
                "ari_order": ari_order,
                "combined": combined,
                "labels": cluster_labels,
            }
        )

        print(
            f"k={k:2d} | sil={sil:.3f} "
            f"| NMI_major={nmi_major:.3f}, NMI_order={nmi_order:.3f} "
            f"| ARI_major={ari_major:.3f}, ARI_order={ari_order:.3f} "
            f"| combined={combined:.3f}"
        )

    return results


def pick_best_result(results):
    # Select k that maximizes the combined score
    best = max(results, key=lambda r: r["combined"])
    print(
        f"\nBest k={best['k']} "
        f"with combined={best['combined']:.3f}, "
        f"sil={best['silhouette']:.3f}, "
        f"NMI_major={best['nmi_major']:.3f}, NMI_order={best['nmi_order']:.3f}"
    )
    return best


def plot_confusion(cm, x_labels, y_labels, title, filename):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="viridis")
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Cluster")
    plt.ylabel("True label")

    if x_labels is not None:
        plt.xticks(
            np.arange(len(x_labels)),
            x_labels,
            rotation=90,
        )
    if y_labels is not None:
        plt.yticks(np.arange(len(y_labels)), y_labels)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    encoded, major_labels, order_labels = load_data()

    # Reasonable search range; adjust if you know roughly how many clusters you expect
    k_values = range(5, 21)

    results = evaluate_kmeans(encoded, major_labels, order_labels, k_values)
    best = pick_best_result(results)

    best_k = best["k"]
    best_labels = best["labels"]

    # Save cluster labels for downstream analysis
    np.save("cluster_labels.npy", best_labels)
    print(f"Saved best cluster labels (k={best_k}) to 'cluster_labels.npy'.")

    # Confusion matrices vs major clade and order
    cm_major = confusion_matrix(major_labels, best_labels)
    cm_order = confusion_matrix(order_labels, best_labels)

    # Use sorted unique labels as tick labels; you can map these back to names if you have a lookup
    major_unique = np.unique(major_labels)
    order_unique = np.unique(order_labels)

    plot_confusion(
        cm_major,
        x_labels=[f"C{c}" for c in range(best_k)],
        y_labels=[str(m) for m in major_unique],
        title=f"Clusters vs Major Clade (k={best_k})",
        filename="confusion_matrix_major_clade_clusters.png",
    )

    plot_confusion(
        cm_order,
        x_labels=[f"C{c}" for c in range(best_k)],
        y_labels=[str(o) for o in order_unique],
        title=f"Clusters vs Order (k={best_k})",
        filename="confusion_matrix_order_clusters.png",
    )

    print(
        "Saved 'confusion_matrix_major_clade_clusters.png' and "
        "'confusion_matrix_order_clusters.png'."
    )


if __name__ == "__main__":
    main()


