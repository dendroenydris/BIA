import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, f_oneway
from matplotlib.patches import Patch

# === Configuration ===
strategy_files = {
    "neutral": "evaluations/results_strategy_0.json",
    "bias-aware": "evaluations/results_strategy_1.json",
    "CoT": "evaluations/results_strategy_2.json"
}
output_dir = "evaluations"
os.makedirs(output_dir, exist_ok=True)

# === Load All Data ===
def load_data():
    results = {}
    for strategy, file in strategy_files.items():
        with open(file, 'r') as f:
            results[strategy] = json.load(f)
    return results

results = load_data()

# === Preprocess Flat Dict ===
def flatten_results(results):
    data = []
    for strategy, biases in results.items():
        for bias_cat, metrics in biases.items():
            for metric_name, value in metrics.items():
                if isinstance(value, dict):
                    # Parity: variance / MAD as imbalance score
                    values = list(value.values())
                    mad = np.mean([abs(v - 1/len(values)) for v in values])
                    data.append({
                        "Strategy": strategy,
                        "Bias Category": bias_cat,
                        "Metric": "Parity",
                        "Value": mad
                    })
                elif isinstance(value, list):
                    # Embedding Distance (mean)
                    data.append({
                        "Strategy": strategy,
                        "Bias Category": bias_cat,
                        "Metric": "Embedding Distance",
                        "Value": np.mean(value)
                    })
                else:
                    # Stereotype Frequency
                    data.append({
                        "Strategy": strategy,
                        "Bias Category": bias_cat,
                        "Metric": "Stereotype Frequency",
                        "Value": value
                    })
    return pd.DataFrame(data)

df = flatten_results(results)

# === Visualization Functions ===
def plot_boxplot(df_cat, bias_cat):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_cat, x="Metric", y="Value", hue="Strategy", palette="Set2")
    plt.title(f"{bias_cat} Bias Metrics Across Strategies", fontsize=14)
    plt.ylabel("Score")
    plt.xlabel("Metric Type")
    plt.legend(title="Prompt Strategy")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{bias_cat}_boxplot.png"))
    plt.close()

def plot_radar(df_cat, bias_cat):
    strategies = df_cat["Strategy"].unique()
    metrics = df_cat["Metric"].unique()
    
    metric_means = {
        strategy: [df_cat[(df_cat["Strategy"] == strategy) & (df_cat["Metric"] == metric)]["Value"].mean()
                   for metric in metrics]
        for strategy in strategies
    }

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    colors = ["tab:blue", "tab:green", "tab:red"]

    for idx, strategy in enumerate(strategies):
        values = metric_means[strategy]
        values += values[:1]
        ax.plot(angles, values, label=strategy, color=colors[idx], linewidth=2)
        ax.fill(angles, values, alpha=0.2, color=colors[idx])

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics)
    ax.set_title(f"{bias_cat} Bias Radar Chart", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{bias_cat}_radar.png"))
    plt.close()

# === Statistical Test ===
def run_significance_tests(df_cat, bias_cat):
    output = []
    for metric in df_cat["Metric"].unique():
        metric_data = df_cat[df_cat["Metric"] == metric]
        groups = [group["Value"].values for name, group in metric_data.groupby("Strategy")]
        if len(groups) > 2:
            stat, p = f_oneway(*groups)
        else:
            stat, p = ttest_ind(*groups)
        sig = "✅" if p < 0.05 else "–"
        output.append({
            "Bias Category": bias_cat,
            "Metric": metric,
            "Test": "ANOVA" if len(groups) > 2 else "t-test",
            "Statistic": stat,
            "p-value": p,
            "Significant": sig
        })
    return output

# === Run All for Each Bias Category ===
all_stats = []
for bias_cat in df["Bias Category"].unique():
    df_cat = df[df["Bias Category"] == bias_cat]
    plot_boxplot(df_cat, bias_cat.replace(" ", "_"))
    plot_radar(df_cat, bias_cat.replace(" ", "_"))
    
    stats = run_significance_tests(df_cat, bias_cat)
    all_stats.extend(stats)

# === Save All Results ===
stats_df = pd.DataFrame(all_stats)
stats_csv_path = os.path.join(output_dir, "significance_tests_summary.csv")
stats_df.to_csv(stats_csv_path, index=False)
