import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import math
from sklearn.decomposition import PCA



plt.style.use("seaborn-v0_8-whitegrid")

def drawBoxplot(df, columns, ncols=3, whis=1.5):
    nrows= math.ceil(len(columns) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 3 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]

        ax.boxplot(
            df[col].dropna(),
            orientation="vertical",
            patch_artist=True,  
            boxprops=dict(facecolor="#2b5c8f", color="#1a365d", alpha=0.85),
            medianprops=dict(color="#ff7f0e", linewidth=2),  
            flierprops=dict(
                marker="o", color="#d62728", alpha=0.5, markersize=4
            ),
            whis=whis
        )

        ax.set_title(col, fontsize=12, fontweight="bold", pad=8)
        ax.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )
        ax.tick_params(labelsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

    # Hide any leftover empty subplots
    for j in range(len(columns), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=2.0)
    plt.show()



def drawHsitKDE(df, columns, ncols=3):
    nrows = math.ceil(len(columns) / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]

        sns.histplot(
            df[col].dropna(),
            bins=20,
            color="#3182ce", 
            edgecolor="#ffffff",
            linewidth=0.8,
            alpha=0.7,  
            kde=True,
            ax=ax,
        )

        if ax.lines:
            ax.lines[0].set_color("#2c5282") 
            ax.lines[0].set_linewidth(2)

        ax.set_title(col, fontsize=12, fontweight="bold", pad=8)
        ax.tick_params(labelsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlabel("") 

    for j in range(len(columns), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=2.0)
    plt.show()


def drawHeatmap(df):
    sns.heatmap(
    df,
    cmap='coolwarm',
    annot=True,
    fmt='.2f'
    )

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.show()


def drawPLot(scores, x, y):
    #figure 
    plt.figure(figsize=(8,5))
    #plot
    plt.plot(scores[x], scores[y], marker='o')
    #style
    plt.title(y)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.xticks(scores[x])
    plt.grid(True, alpha= 0.3)

    plt.show()


def plotClustersPCA(processedData, labels):

    pca = PCA(n_components=2)

    components = pca.fit_transform(processedData)

    pca_df = pd.DataFrame(
        components,
        columns=["PC1", "PC2"]
    )

    pca_df["cluster"] = labels

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette="Set1",
        s=60
    )

    plt.title("K-Means Clusters Visualized with PCA")
    plt.xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)"
    )
    plt.ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)"
    )
    plt.legend(title="Cluster")
    plt.show()


def drawBarPlot(data, col):
    plt.figure(figsize=(10, 6))

    plt.bar(
        data["model"],
        data[f"{col}_mean"],
        yerr=data[f"{col}_std"],
        capsize=5
    )

    plt.ylabel(col)
    plt.xlabel("Model")
    plt.title(f"5-Fold Cross-Validation — {col}")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.show()
