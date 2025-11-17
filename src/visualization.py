import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple, Dict

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class DataVisualizer:    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        
    def plot_histogram(self, data: np.ndarray, 
                      feature_names: List[str],
                      bins: int = 30,
                      title: str = "Feature Distributions") -> plt.Figure:
        n_features = data.shape[1]
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx in range(n_features):
            ax = axes[idx]
            ax.hist(data[:, idx], bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel(feature_names[idx], fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'Distribution: {feature_names[idx]}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            mean_val = np.mean(data[:, idx])
            median_val = np.median(data[:, idx])
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
            ax.legend()
        
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        
        return fig
    
    def plot_correlation_heatmap(self, data: np.ndarray,
                                feature_names: List[str],
                                title: str = "Correlation Heatmap",
                                cmap: str = "coolwarm") -> plt.Figure:
        mean = np.mean(data, axis=0)
        centered = data - mean
        n = data.shape[0]
        # Use einsum for efficient correlation matrix
        cov_matrix = np.einsum('ij,ik->jk', centered, centered) / (n - 1)
        std = np.sqrt(np.diag(cov_matrix) + 1e-10)
        corr_matrix = cov_matrix / np.outer(std, std)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(corr_matrix,
                   annot=True,
                   fmt='.2f',
                   cmap=cmap,
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8},
                   xticklabels=feature_names,
                   yticklabels=feature_names,
                   ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        return fig
    
    def plot_pie_chart(self, data: np.ndarray,
                      labels: List[str],
                      title: str = "Pie Chart") -> plt.Figure:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        wedges, texts, autotexts = ax.pie(data,
                                          labels=labels,
                                          autopct='%1.1f%%',
                                          colors=colors,
                                          startangle=90,
                                          explode=[0.05] * len(data))
        
        for text in texts:
            text.set_fontsize(11)
            text.set_fontweight('bold')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return fig
    
    def plot_categorical_cancellations(self, categorical_data: np.ndarray,
                                      feature_names: List[str],
                                      figsize: Tuple[int, int] = (18, 12)) -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for idx, feature_name in enumerate(feature_names):
            unique_vals, counts = np.unique(categorical_data[:, idx], return_counts=True)
            
            sorted_indices = np.argsort(counts)[::-1]
            unique_vals_sorted = unique_vals[sorted_indices]
            counts_sorted = counts[sorted_indices]
            
            total = np.sum(counts_sorted)
            percentages = (counts_sorted / total) * 100
            
            ax = axes[idx]
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_vals_sorted)))
            bars = ax.barh(range(len(unique_vals_sorted)), percentages, color=colors, 
                          edgecolor='white', linewidth=2)
            
            for i, (bar, pct) in enumerate(zip(bars, percentages)):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                       f'{pct:.2f}%', va='center', fontweight='bold')
            
            ax.set_yticks(range(len(unique_vals_sorted)))
            ax.set_yticklabels(unique_vals_sorted)
            ax.set_xlabel('Percentage of Attrited Customers', fontsize=10, fontweight='bold')
            ax.set_title(f'Cancellations by {feature_name}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
        
        if len(feature_names) < 6:
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        return fig
    
    def plot_transaction_comparison(self, existing_data: np.ndarray, 
                                   attrited_data: np.ndarray,
                                   feature_names: List[str]) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, var_name in enumerate(feature_names):
            ax = axes[idx]
            data_to_plot = [existing_data[:, idx], attrited_data[:, idx]]
            
            bp = ax.boxplot(data_to_plot, labels=['Existing Customer', 'Attrited Customer'],
                           patch_artist=True, showmeans=True)
            
            colors = ['lightgreen', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_ylabel(var_name, fontsize=11, fontweight='bold')
            ax.set_title(f'Distribution of {var_name} by Customer Status', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            mean_existing = np.mean(existing_data[:, idx])
            mean_attrited = np.mean(attrited_data[:, idx])
            ax.text(1, mean_existing, f'Mean: {mean_existing:.2f}', 
                   ha='center', va='bottom', fontsize=9)
            ax.text(2, mean_attrited, f'Mean: {mean_attrited:.2f}', 
                   ha='center', va='top', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_transaction_distribution_bins(self, trans_ct_data: np.ndarray,
                                          attrited_mask: np.ndarray,
                                          bins: List[int],
                                          bin_labels: List[str]) -> plt.Figure:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        trans_ct_binned = np.digitize(trans_ct_data, bins=bins)
        
        for group_idx, (group_name, mask) in enumerate([('Existing Customer', ~attrited_mask), 
                                                          ('Attrited Customer', attrited_mask)]):
            ax = ax1 if group_idx == 0 else ax2
            
            group_trans_binned = trans_ct_binned[mask]
            bin_counts = np.array([np.sum(group_trans_binned == i+1) for i in range(len(bin_labels))])
            
            total = np.sum(bin_counts)
            percentages = (bin_counts / total) * 100
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(bin_labels)))
            bars = ax.bar(range(len(bin_labels)), percentages, color=colors, 
                         edgecolor='black', linewidth=1.5, alpha=0.8)
            
            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{pct:.1f}%', ha='center', va='bottom', 
                       fontweight='bold', fontsize=10)
            
            ax.set_xticks(range(len(bin_labels)))
            ax.set_xticklabels(bin_labels)
            ax.set_xlabel('Transaction Count Range', fontsize=12, fontweight='bold')
            ax.set_ylabel('Percentage of Customers', fontsize=12, fontweight='bold')
            ax.set_title(f'Transaction Distribution - {group_name}', 
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, max(percentages) * 1.15)
        
        plt.tight_layout()
        return fig
    
    def plot_transaction_scatter(self, trans_ct_data: np.ndarray,
                                trans_amt_data: np.ndarray,
                                attrited_mask: np.ndarray) -> plt.Figure:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = {'Existing Customer': 'green', 'Attrited Customer': 'red'}
        
        for group_name, mask in [('Existing Customer', ~attrited_mask), 
                                 ('Attrited Customer', attrited_mask)]:
            x_data = trans_ct_data[mask]
            y_data = trans_amt_data[mask]
            
            ax.scatter(x_data, y_data, c=colors[group_name], marker='o',
                      alpha=0.5, s=50, label=group_name, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel('Total Transaction Count (Last 12 Months)', 
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('Total Transaction Amount (Last 12 Months)', 
                     fontsize=12, fontweight='bold')
        ax.set_title('Transaction Count vs Transaction Amount by Customer Status', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_categorical_comparison(self, categorical_data: np.ndarray,
                                   attrited_mask: np.ndarray,
                                   feature_names: List[str]) -> plt.Figure:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for idx, cat_name in enumerate(feature_names):
            ax = axes[idx]
            
            existing_data = categorical_data[~attrited_mask, idx]
            attrited_data = categorical_data[attrited_mask, idx]
            unique_cats = np.unique(categorical_data[:, idx])
            
            existing_pcts = []
            attrited_pcts = []
            
            for cat in unique_cats:
                existing_count = np.sum(existing_data == cat)
                attrited_count = np.sum(attrited_data == cat)
                existing_pct = (existing_count / len(existing_data)) * 100
                attrited_pct = (attrited_count / len(attrited_data)) * 100
                existing_pcts.append(existing_pct)
                attrited_pcts.append(attrited_pct)
            
            x = np.arange(len(unique_cats))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, existing_pcts, width, label='Existing Customer',
                          color='lightgreen', alpha=0.8, edgecolor='black', linewidth=1)
            bars2 = ax.bar(x + width/2, attrited_pcts, width, label='Attrited Customer',
                          color='lightcoral', alpha=0.8, edgecolor='black', linewidth=1)
            
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                               f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Categories', fontsize=11, fontweight='bold')
            ax.set_ylabel('Percentage within Group', fontsize=11, fontweight='bold')
            ax.set_title(f'Distribution by {cat_name}', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(unique_cats, rotation=45, ha='right', fontsize=9)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            
            max_pct = max(max(existing_pcts), max(attrited_pcts))
            ax.set_ylim(0, max_pct * 1.15)
        
        if len(feature_names) < 6:
            fig.delaxes(axes[-1])
        
        plt.tight_layout()
        return fig

    def plot_model_comparison(self, 
                             results_basic: Dict[str, Dict], 
                             results_enhanced: Dict[str, Dict],
                             figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        models_to_plot = list(results_basic.keys())
        metrics = ['test_acc', 'precision', 'recall', 'f1']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            
            basic_scores = [results_basic[m][metric] for m in models_to_plot]
            enhanced_scores = [results_enhanced[m][metric] for m in models_to_plot]
            
            x = np.arange(len(models_to_plot))
            width = 0.35
            
            ax.bar(x - width/2, basic_scores, width, label='Basic', alpha=0.8, color='lightcoral')
            ax.bar(x + width/2, enhanced_scores, width, label='Enhanced', alpha=0.8, color='lightgreen')
            
            ax.set_ylabel(metric_name, fontweight='bold', fontsize=12)
            ax.set_title(f'{metric_name} Comparison', fontweight='bold', fontsize=14)
            ax.set_xticks(x)
            ax.set_xticklabels(models_to_plot, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([0, 1.05])
            
            for i, (b_score, e_score) in enumerate(zip(basic_scores, enhanced_scores)):
                ax.text(i - width/2, b_score + 0.02, f'{b_score:.3f}', ha='center', va='bottom', fontsize=9)
                ax.text(i + width/2, e_score + 0.02, f'{e_score:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return fig


def save_figure(fig: plt.Figure, filepath: str, dpi: int = 300):
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {filepath}")
    plt.close(fig)

