import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

import os 
import math
import scipy as sp


def _save_visualization_bar(annot, k=6):
    # Load your CSV
    df = pd.read_csv(f"logs/{annot}.csv")
    df = df[df['n_tasks'] == k]

    # Plot: Grouped bar plot by leave_out_task (season), each bar is a method
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="leave_out_task", y="loss", hue="method")

    plt.title(f"Loss by Method for Each Leave-Out Task - k{k}")
    plt.xlabel("Leave-Out Task (Season)")
    plt.ylabel("Loss")
    plt.legend(title="Method", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"viz/{annot}.svg", format="svg")

    # # Show the plot
    # plt.show()

def _save_visualization_line(annot):
    # Load the data
    df = pd.read_csv(f"logs/{annot}.csv")

    # Filter out missing or inconsistent data if needed
    df = df.dropna(subset=["leave_out_task", "loss", "method", "n_tasks"])

    # Plot: One plot per domain (leave_out_task), x = n_tasks, y = loss, hue = method
    g = sns.FacetGrid(
        df,
        col="leave_out_task",
        col_wrap=3,  # Adjust based on number of domains and how wide you want the output
        sharey=False,  # Let each plot have its own y-axis scale (can set to True for comparison)
        height=4,
        aspect=1.5
    )
    g.map_dataframe(sns.lineplot, x="n_tasks", y="loss", hue="method", marker="o")

    g.set_axis_labels("Number of Tasks Trained On", "Loss")
    g.set_titles(col_template="Leave-Out Task: {col_name}")
    g.add_legend(title="Method")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"viz/{annot}.svg", format="svg")

def plot_per_sample_results(csv_path, annot, output_dir="per-sample-plots", focus_task=None, focus_method=None):
    """
    Load per-sample result CSV and generate summary plots.
    
    Parameters:
    - csv_path: str, path to the saved CSV file (e.g., 'per_sample_results/per_sample_run1.csv')
    - output_dir: str, folder to save plots (default: 'plots')
    - focus_task: str, if specified, plots predicted vs true for this task only
    - focus_method: str, if specified, filters predicted vs true plot by this method
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(csv_path)
    
    # --- Boxplot: Residuals by Task and Method ---
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df, x="task", y="residual", hue="method")
    plt.title("Residuals by Task and Method")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"box_plot_{annot}.png"))
    plt.close()
    
    # --- Barplot: Mean Residuals by Task and Method ---
    mean_res = df.groupby(['task', 'method'])['residual'].mean().reset_index()
    plt.figure(figsize=(14, 6))
    sns.barplot(data=mean_res, x="task", y="residual", hue="method")
    plt.title("Mean Residuals by Task and Method")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"bar_plot_{annot}.png"))
    plt.close()
    
    # --- Scatterplot: Prediction vs True (Optional) ---
    if focus_task and focus_method:
        subset = df[(df['task'] == focus_task) & (df['method'] == focus_method)]
        if not subset.empty:
            plt.figure(figsize=(6, 6))
            sns.scatterplot(data=subset, x="true_label", y="prediction", alpha=0.7)
            min_val = min(subset["true_label"].min(), subset["prediction"].min())
            max_val = max(subset["true_label"].max(), subset["prediction"].max())
            plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label=f"{focus_method}_{focus_task}")
            plt.xlabel("True Label")
            plt.ylabel("Prediction")
            plt.title(f"{focus_method} on {focus_task}")
            plt.legend()
            plt.tight_layout()
            
            os.makedirs(f"{output_dir}/{focus_method}", exist_ok=True)

            fname = f"{focus_method}/{annot}_{focus_method}_{focus_task}.png"
            plt.savefig(os.path.join(output_dir, fname))
            plt.close()
        else:
            print(f"[!] No data found for task={focus_task}, method={focus_method}")

    print(f"✅ Plots saved to '{output_dir}/'")

# Rewrite: they have the same parameter  
def levene(residual, n_samples_per_task):
    num_tasks = len(n_samples_per_task)
    task_boundary = np.concatenate(([0], np.cumsum(n_samples_per_task)))
    residual_boundary = [residual[task_boundary[i]:task_boundary[i+1], :] for i in range(num_tasks)]
    stat, pval = sp.stats.levene(*residual_boundary)
    # print("Stat, pval", stat, pval)
    return pval[0]

def hsic(residual, n_samples_per_task):
    def get_kernel_matrix(X, sX):
        kernel = (X[:, :, np.newaxis] - X.T).T
        kernel = np.exp(-1.0 / (2 * sX) * np.linalg.norm(kernel, axis=1))
        return kernel

    def get_sX(X, y):
        k = X[:, :, np.newaxis] - y.T 
        ls = np.linalg.norm(k, axis=1)
        sX = 0.5 * np.median(ls.flatten())
        return sX
    
    # Optimized 
    def get_task_boundray_matrix(n_samples_per_task):
        return block_diag(*[np.ones((n, n)) for n in n_samples_per_task])

    valid_dom = get_task_boundray_matrix(n_samples_per_task)
    sX = get_sX(residual, residual)
    #------- 
    X = residual
    y = valid_dom

    n = X.T.shape[1]
    kernel_X = get_kernel_matrix(X, sX)
    kernel_y = valid_dom
    coef = 1.0 / n

     # The formula can be founded there https://proceedings.neurips.cc/paper_files/paper/2007/file/d5cfead94f5350c12c322b5b664544c1-Paper.pdf
    HSIC = (
        (coef**2) * np.sum(kernel_X * kernel_y)
        + coef**4 * np.sum(kernel_X) * np.sum(kernel_y)
        - 2 * coef**3 * np.sum(np.sum(kernel_X, axis=1) * np.sum(kernel_y, axis=1))
    )

    # Get sums of Kernels
    KXsum = np.sum(kernel_X)
    KYsum = np.sum(kernel_y)

    # Get stats for gamma approx
    xMu = 1.0 / (n * (n - 1)) * (KXsum - n)
    yMu = 1.0 / (n * (n - 1)) * (KYsum - n)
    V1 = (
        coef**2 * np.sum(kernel_X * kernel_X)
        + coef**4 * KXsum**2
        - 2 * coef**3 * np.sum(np.sum(kernel_X, axis=1) ** 2)
    )
    V2 = (
        coef**2 * np.sum(kernel_y * kernel_y)
        + coef**4 * KYsum**2
        - 2 * coef**3 * np.sum(np.sum(kernel_y, axis=1) ** 2)
    )

    meanH0 = (1.0 + xMu * yMu - xMu - yMu) / n
    varH0 = 2.0 * (n - 4) * (n - 5) / (n * (n - 1.0) * (n - 2.0) * (n - 3.0)) * V1 * V2

    # Parameters of the Gamma
    a = meanH0**2 / varH0
    b = n * varH0 / meanH0

    pval = 1.0 - sp.stats.gamma.cdf(n * HSIC, a, scale=b)
    return pval

def plot_scatter_on_all_tasks(csv_path, annot, set, output_dir="scatter-on-all-task", focus_method=None, point_size=10):
    print(f"FOCUS METHOD: {focus_method}____________")
    df = pd.read_csv(csv_path)
    df = df[df['split'] == set]

    if focus_method is not None:
        df = df[df['method'] == focus_method]

    # Create output directory if it doesn't exist
    output_dir = f"{output_dir}/{focus_method}"
    os.makedirs(output_dir, exist_ok=True)

    # Calculate statistics
    mean_residual = np.mean(np.abs(df['prediction'] - df['true_label']))
    mean_rmse = np.sqrt(np.mean((df['prediction'] - df['true_label'])**2))

    print(f'-----MEAN RESIDUAL: {mean_residual} --------------')

    n_samples_per_tasks = df['task'].value_counts().values


    print(df['residual'].head(5))
    print(n_samples_per_tasks)
    test = 'levene'
    if test == 'levene':
        grouped = df.groupby('task')['residual'].apply(list)
        residual_boundary = [np.array(r) for r in grouped]

        # Gọi kiểm định Levene đúng cách:
        stat, pval = sp.stats.levene(*residual_boundary)
                # pval = levene(df['residual'].to_numpy()[:, np.newaxis], n_samples_per_tasks)
    else: 
        pval = hsic(df['residual'].to_numpy()[:, np.newaxis], n_samples_per_tasks)
        
    # print(df['residual'].head(10))
    # print(n_samples_per_tasks)
    # pval = pval[0]

    print(f"Pvalues {pval}")

    # --- Scatterplot: Prediction vs True for All Tasks ---
    plt.figure(figsize=(10, 8))

    sns.scatterplot(
        data=df,
        x="true_label",
        y="prediction",
        hue="task",
        alpha=0.7,
        s=point_size
    )

    # Add ideal prediction line
    min_val = min(df["true_label"].min(), df["prediction"].min())
    max_val = max(df["true_label"].max(), df["prediction"].max())
    plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal Prediction")

    # Annotate mean residual and RMSE on the plot
    textstr = f"Mean Residual: {mean_residual:.4f}\nMean RMSE: {mean_rmse:.4f}\{test} pvals: {pval}"
    plt.text(
        0.05, 0.95, textstr,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray')
    )

    # Axis labels and title
    plt.xlabel("True Label")
    plt.ylabel("Prediction")
    plt.title(f"{focus_method} — Prediction vs True Label ({set} Set)")

    # Legend and layout
    plt.legend(title="Task", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save plot
    fname = f"{set}_{focus_method}_{annot}_all_tasks.png"
    plt.savefig(os.path.join(output_dir, fname))
    plt.close()

    print(f"✅ Plots saved to '{output_dir}/'")

     # --- KDE Plot: Residual Distribution per Task ---
    plt.figure(figsize=(10, 6))

    sns.kdeplot(data=df, x="residual", hue="task", common_norm=False, fill=True, alpha=0.3)
    # Annotate mean residual and RMSE on the plot
    textstr = f"{test} pvals: {pval}"
    plt.text(
        0.05, 0.95, textstr,
        transform=plt.gca().transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray')
    )
    plt.title(f"{set}_{focus_method} — Residual Distribution per Task ({set} Set)")
    plt.xlabel("Residual")
    plt.ylabel("Density")
    plt.legend(title="Task", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # os.makedirs(f"{output_dir}/kde", exist_ok=True)
    # output_dir = f"{output_dir}/kde"

    kde_fname = f"{set}_{focus_method}_{annot}_residual_kde.png"
    plt.savefig(os.path.join(output_dir, kde_fname))
    plt.close()

    print(f"✅ Plots saved to '{output_dir}/'")

def plot_scatter_per_task(csv_path, annot, output_dir="per-sample-plots", focus_method=None, point_size=60):
    df = pd.read_csv(csv_path)

    if focus_method is not None:
        df = df[df['method'] == focus_method]

    os.makedirs(output_dir, exist_ok=True)

    tasks = sorted(df['task'].unique())
    n_tasks = len(tasks)
    ncols = 3
    nrows = math.ceil(n_tasks / ncols)

    # Shared x/y limits
    min_val = min(df["true_label"].min(), df["prediction"].min())
    max_val = max(df["true_label"].max(), df["prediction"].max())


    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True)

    # Flatten axes for easy iteration
    axes = axes.flatten()

    for i, task in enumerate(tasks):
        ax = axes[i]
        df_task = df[df["task"] == task]
        sns.scatterplot(
            data=df_task,
            x="true_label",
            y="prediction",
            ax=ax,
            alpha=0.7,
            s=point_size
        )
        ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", label="Ideal")
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_title(f"Task {task}")
        ax.set_xlabel("True Label")
        ax.set_ylabel("Prediction")
        ax.legend()

    # Turn off any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Save to SVG
    fname = f"{focus_method}_{annot}_grid.svg"
    fig.savefig(os.path.join(output_dir, fname), format="svg")
    plt.close(fig)

    print(f"✅ Grid plot saved to '{output_dir}/{fname}'")
