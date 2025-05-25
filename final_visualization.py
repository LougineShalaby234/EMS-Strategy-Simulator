import pandas as pd
import matplotlib.pyplot as plt

def plot_grouped_bar_chart(csv_file):
    # Read the CSV
    df = pd.read_csv(csv_file)

    # Group by scenario and agent to get mean and std
    grouped = df.groupby(["scenario", "agent"]).agg(
        score_mean=('score', 'mean'),
        score_std=('score', 'std')
    ).reset_index()

    # Pivot for plotting
    mean_pivot = grouped.pivot(index='scenario', columns='agent', values='score_mean')
    std_pivot = grouped.pivot(index='scenario', columns='agent', values='score_std')

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))
    mean_pivot.plot(
        kind='bar',
        yerr=std_pivot,
        capsize=4,
        ax=ax,
        rot=0,
        alpha=0.85,
        edgecolor='black'
    )

    # Labeling with APA-style: no bold on ylabel or title
    ax.set_xlabel("Scenario", fontsize=30 , labelpad=15)
    ax.set_ylabel("Call to Admission Time (Min)", fontsize=24, labelpad=15)  # no bold
    ax.set_title("Performance of Policies per Scenario", fontsize=28, pad=20)  # no bold

    ax.tick_params(axis='y', labelsize=22)
    ax.tick_params(axis='x', labelsize=22)

    # Minimalist style: remove grid and top/right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend without frame, moved to the right
    ax.legend( fontsize=24, loc='center left',
              bbox_to_anchor=(1.0, 0.5), frameon=False)

    # Clean layout
    plt.tight_layout()

    # Save the result
    plt.savefig("final_results.jpg", dpi=300, bbox_inches='tight')
    # plt.show()

# Example usage:
plot_grouped_bar_chart("final_results.csv")

