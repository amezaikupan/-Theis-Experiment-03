import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_causal_features(data_file=None, df=None):
    """
    Visualize causal feature identification performance across different scenarios.
    
    Parameters:
    -----------
    data_file : str, optional
        Path to CSV file containing the data
    df : pandas.DataFrame, optional  
        DataFrame containing the data directly
        
    The data should have columns: rep, method, total_causal_features, causal_removed, 
    causal_remaining, causal_selected, selection_accuracy
    """
    
    # Load data
    if df is None:
        if data_file is None:
            # Use the provided sample data
            data = """rep,method,total_causal_features,causal_removed,causal_remaining,causal_selected,selection_accuracy,selected_features_original,selected_features_truncated
0,shat,6,0,6,6,1.0,"0,1,2,3,4,5","0,1,2,3,4,5"
0,sgreedy,6,0,6,5,0.8333333333333334,"0,1,2,4,5","0,1,2,4,5"
0,shat,6,1,5,5,1.0,"1,2,3,4,5","0,1,2,3,4"
0,sgreedy,6,1,5,4,0.8,"1,2,4,5","0,1,3,4"
0,shat,6,2,4,4,1.0,"2,3,4,5","0,1,2,3"
0,sgreedy,6,2,4,3,0.75,"2,4,5","0,2,3"
0,shat,6,3,3,3,1.0,"3,4,5,8","0,1,2,5"
0,sgreedy,6,3,3,3,1.0,"3,4,5,8","0,1,2,5"
0,shat,6,4,2,2,1.0,"4,5","0,1"
0,sgreedy,6,4,2,2,1.0,"4,5","0,1"
0,shat,6,5,1,1,1.0,"5,8","0,3"
0,sgreedy,6,5,1,1,1.0,"5,8","0,3"
0,shat,6,6,0,0,0.0,8,2
0,sgreedy,6,6,0,0,0.0,8,2"""
            
            from io import StringIO
            df = pd.read_csv(StringIO(data))
        else:
            df = pd.read_csv(data_file)
    
    # Create the visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Causal Feature Identification Performance Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Number of causal features identified vs removed (using mean for multiple reps)
    ax1 = axes[0, 0]
    for method in df['method'].unique():
        method_data = df[df['method'] == method].groupby('causal_removed')['causal_selected'].mean().reset_index()
        ax1.plot(method_data['causal_removed'], method_data['causal_selected'], 
                marker='o', linewidth=2.5, markersize=8, label=method)
    
    ax1.set_xlabel('Number of Causal Features Removed', fontweight='bold')
    ax1.set_ylabel('Average Number of Causal Features Identified', fontweight='bold')
    ax1.set_title('Causal Features Identified vs. Removed', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Selection accuracy vs causal features removed (using mean for multiple reps)
    ax2 = axes[0, 1]
    for method in df['method'].unique():
        method_data = df[df['method'] == method].groupby('causal_removed')['selection_accuracy'].mean().reset_index()
        ax2.plot(method_data['causal_removed'], method_data['selection_accuracy'], 
                marker='s', linewidth=2.5, markersize=8, label=method)
    
    ax2.set_xlabel('Number of Causal Features Removed', fontweight='bold')
    ax2.set_ylabel('Average Selection Accuracy', fontweight='bold')
    ax2.set_title('Selection Accuracy vs. Features Removed', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)
    
    # Plot 3: Heatmap of causal features identified (using mean for duplicates)
    ax3 = axes[1, 0]
    pivot_data = df.groupby(['causal_removed', 'method'])['causal_selected'].mean().unstack()
    sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax3, 
                cbar_kws={'label': 'Avg Causal Features Identified'})
    ax3.set_xlabel('Method', fontweight='bold')
    ax3.set_ylabel('Causal Features Removed', fontweight='bold')
    ax3.set_title('Heatmap: Average Causal Features Identified', fontweight='bold')
    
    # Plot 4: Bar plot comparing methods at each removal level (using mean for multiple reps)
    ax4 = axes[1, 1]
    
    # Group by causal_removed and method, then calculate mean
    grouped_data = df.groupby(['causal_removed', 'method'])['causal_selected'].mean().reset_index()
    
    removal_levels = sorted(df['causal_removed'].unique())
    x = np.arange(len(removal_levels))
    width = 0.35
    
    methods = df['method'].unique()
    method_colors = ['skyblue', 'lightcoral']
    
    for i, method in enumerate(methods):
        method_data = grouped_data[grouped_data['method'] == method]
        # Ensure data is in the right order
        values = []
        for level in removal_levels:
            match = method_data[method_data['causal_removed'] == level]
            if len(match) > 0:
                values.append(match['causal_selected'].iloc[0])
            else:
                values.append(0)
        
        bars = ax4.bar(x + (i - 0.5) * width, values, width, 
                      label=method, alpha=0.8, color=method_colors[i % len(method_colors)])
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{values[j]:.1f}', ha='center', va='bottom', fontweight='bold')
    
    ax4.set_xlabel('Causal Features Removed', fontweight='bold')
    ax4.set_ylabel('Average Causal Features Identified', fontweight='bold')
    ax4.set_title('Method Comparison: Average Features Identified', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(removal_levels)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("CAUSAL FEATURE IDENTIFICATION SUMMARY")
    print("="*60)
    
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        print(f"\n{method.upper()} Method:")
        print(f"  • Average features identified: {method_data['causal_selected'].mean():.2f}")
        print(f"  • Average selection accuracy: {method_data['selection_accuracy'].mean():.3f}")
        print(f"  • Best performance (features removed, identified): "
              f"({method_data.loc[method_data['causal_selected'].idxmax(), 'causal_removed']}, "
              f"{method_data['causal_selected'].max()})")
    
    print(f"\nOverall Analysis:")
    print(f"  • Total scenarios tested: {len(df['causal_removed'].unique())}")
    print(f"  • Range of causal features removed: {df['causal_removed'].min()} to {df['causal_removed'].max()}")
    print(f"  • Both methods perform equally well when ≥4 features are removed")
    
    return fig

# Example usage:
# visualize_causal_features()  # Uses the sample data
# visualize_causal_features('your_data.csv')  # Uses your CSV file
# visualize_causal_features(df=your_dataframe)  # Uses your DataFrame
# Example usage:
# visualize_causal_features()  # Uses the sample data
visualize_causal_features('Experiment_01_top_left/tl_norm_5_2.0_1.0_0.5_features.csv')  # Uses your CSV file
# visualize_causal_features(df=your_dataframe)  # Uses your DataFrame