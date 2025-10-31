"""
Plotting Module

This module contains functions for visualizing geochemical data and regression results.
It includes specialized plotting functions for geochemical analysis including boxplots,
scatter plots, and survival function plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings

# Suppress performance warnings
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def _prepare_boxplot_data(data: Union[Dict[Tuple[str, str], List[float]], Any],
                          parameter: str = 'coefficients',
                          p_value_threshold: Optional[float] = None,
                          n_samples_threshold: Optional[int] = None) -> Dict[Tuple[str, str], List[float]]:
    """
    Prepare data for boxplot visualization.
    
    Handles both dict format and RegressionGroupCollection objects.
    
    Args:
        data: Either a dict mapping (ind_var, dep_var) to lists, or a RegressionGroupCollection
        parameter: Parameter to extract (for RegressionGroupCollection)
        p_value_threshold: Optional p-value threshold (for RegressionGroupCollection)
        n_samples_threshold: Optional sample size threshold (for RegressionGroupCollection)
        
    Returns:
        Dictionary mapping (independent_var, dependent_var) tuples to lists of values
    """
    # Check if it's a RegressionGroupCollection
    if hasattr(data, 'extract_for_boxplots'):
        # It's a RegressionGroupCollection
        return data.extract_for_boxplots(
            parameter=parameter,
            p_threshold=p_value_threshold,
            min_samples=n_samples_threshold
        )
    elif isinstance(data, dict):
        # It's already in the correct format
        return data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Expected dict or RegressionGroupCollection.")


def plot_regression_boxplots(results_dict: Union[Dict[Tuple[str, str], List[float]], Any], 
                           independent_vars: List[str], group_feature: str,
                           p_value_threshold: float = 0.95, n_samples_threshold: int = 5,
                           min_length: int = 5, figsize: Tuple[int, int] = (15, 12),
                           save_path: Optional[str] = None, dpi: int = 300,
                           parameter: str = 'coefficients') -> None:
    """
    Create boxplots for regression slopes grouped by independent variables.
    
    This function creates boxplots showing the distribution of regression slopes
    for different element pairs, filtered by significance and sample size.
    
    Args:
        results_dict: Dictionary containing regression results, or RegressionGroupCollection
        independent_vars: List of independent variable names
        group_feature: Name of the grouping feature
        p_value_threshold: Minimum p-value threshold for significance
        n_samples_threshold: Minimum number of samples per group
        min_length: Minimum number of groups per boxplot
        figsize: Figure size tuple
        save_path: Optional path to save the figure
        dpi: DPI for saved figure
        parameter: Parameter to extract for boxplots (used when results_dict is RegressionGroupCollection)
    """
    # Prepare data - handles both dict and RegressionGroupCollection formats
    boxplot_data = _prepare_boxplot_data(
        results_dict,
        parameter=parameter,
        p_value_threshold=p_value_threshold if p_value_threshold < 1.0 else None,
        n_samples_threshold=n_samples_threshold
    )
    
    fig = plt.figure(figsize=figsize)
    
    # Create title
    fig.suptitle(f'Linear regression boxplots (slope) grouped by: {group_feature}\n\n'
                f'min P-value: {p_value_threshold} | min samples/group: {n_samples_threshold} | '
                f'min groups/boxplot: {min_length}', fontsize=12)
    
    axis_counter = 1
    
    for independent_var in independent_vars:
        ax = fig.add_subplot(len(independent_vars), 1, axis_counter)
        _plot_single_boxplot(ax, boxplot_data, independent_var, min_length)
        axis_counter += 1
    
    fig.subplots_adjust(hspace=0.4)
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def _plot_single_boxplot(axis, main_dict: Dict[Tuple[str, str], List[float]], 
                        independent_var: str, min_length: int = 5) -> None:
    """
    Plot a single boxplot for one independent variable.
    
    Args:
        axis: Matplotlib axis object
        main_dict: Dictionary containing regression results
        independent_var: Independent variable name
        min_length: Minimum number of groups per boxplot
    """
    # Filter data by independent variable
    data = _filter_dictionary_by_independent_var(main_dict, independent_var)
    
    # Filter data by minimum number of groups
    if min_length is not None:
        data = _filter_dictionary_by_length_of_groups(data, min_length)
    
    # Prepare data for plotting
    vals = [data[key] for key in data.keys()]
    labs = [i[1].split('_')[0] for i in list(data.keys())]
    
    # Plot boxplot
    boxplot = axis.boxplot(vals, showfliers=False)
    
    # Add group counts below whiskers
    ybuffer = (axis.get_ylim()[1] - axis.get_ylim()[0]) * 0.01
    for i, group_size in enumerate([len(v) for v in vals], start=1):
        whisker_min = boxplot['whiskers'][2*(i-1)].get_ydata()[1]
        axis.text(i, whisker_min - ybuffer, str(group_size), 
                 ha='center', va='top', fontsize=10, color='k')
    
    # Format plot
    axis.set_xticklabels(labs, rotation=90)
    axis.yaxis.grid(True)
    axis.axhline(0, color='darkred', lw=1)
    axis.set_title(f'Independent variable: {independent_var}', fontsize='medium')
    axis.set_ylabel('OLS linear slope')


def plot_survival_function(values_dict: Dict[str, List[float]], 
                          title: str = 'Survival Function', xlabel: str = 'Value',
                          ylabel: str = 'Fraction Greater', 
                          figsize: Tuple[int, int] = (3, 3),
                          save_path: Optional[str] = None, dpi: int = 300) -> None:
    """
    Plot survival function (complementary cumulative distribution) for multiple datasets.
    
    Args:
        values_dict: Dictionary mapping dataset names to value lists
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size tuple
        save_path: Optional path to save the figure
        dpi: DPI for saved figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, (dataset_name, values) in enumerate(values_dict.items()):
        clean_values = _clean_list(values)
        if len(clean_values) > 0:
            ax.ecdf(clean_values, complementary=True, 
                   color=colors[i % len(colors)], label=dataset_name)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(title='Dataset')
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_regression_scatter(results_dict: Dict[str, Dict], 
                          x_pair: Tuple[str, str], y_pair: Tuple[str, str],
                          color_stat: str = 'p_value', figsize: Tuple[int, int] = (8, 6),
                          save_path: Optional[str] = None, dpi: int = 300) -> None:
    """
    Create scatter plot comparing regression results between two element pairs.
    
    Args:
        results_dict: Dictionary containing regression results
        x_pair: Tuple of (independent_var, dependent_var) for x-axis
        y_pair: Tuple of (independent_var, dependent_var) for y-axis
        color_stat: Statistic to use for coloring points
        figsize: Figure size tuple
        save_path: Optional path to save the figure
        dpi: DPI for saved figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    x_values = []
    y_values = []
    color_values = []
    
    for group_name in results_dict.keys():
        if x_pair in results_dict[group_name] and y_pair in results_dict[group_name]:
            x_values.append(results_dict[group_name][x_pair]['coefficients'])
            y_values.append(results_dict[group_name][y_pair]['coefficients'])
            color_values.append(results_dict[group_name][x_pair][color_stat])
    
    scatter = ax.scatter(x_values, y_values, c=color_values, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label=color_stat)
    
    ax.set_xlabel(f'Slope: {x_pair[0]} / {x_pair[1]}')
    ax.set_ylabel(f'Slope: {y_pair[0]} / {y_pair[1]}')
    ax.set_title('Regression Slope Comparison')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_element_concentrations(df: pd.DataFrame, elements: List[str], 
                               group_by: Optional[str] = None,
                               plot_type: str = 'box', figsize: Tuple[int, int] = (12, 8),
                               save_path: Optional[str] = None, dpi: int = 300) -> None:
    """
    Plot element concentrations using various plot types.
    
    Args:
        df: DataFrame containing geochemical data
        elements: List of elements to plot
        group_by: Optional column name to group by
        plot_type: Type of plot ('box', 'violin', 'hist', 'scatter')
        figsize: Figure size tuple
        save_path: Optional path to save the figure
        dpi: DPI for saved figure
    """
    fig, axes = plt.subplots(figsize=figsize)
    
    if plot_type == 'box':
        if group_by:
            df_melted = df.melt(id_vars=[group_by], value_vars=elements, 
                              var_name='Element', value_name='Concentration')
            sns.boxplot(data=df_melted, x='Element', y='Concentration', hue=group_by, ax=axes)
        else:
            df[elements].boxplot(ax=axes)
        axes.set_ylabel('Concentration (wt%)')
        
    elif plot_type == 'violin':
        if group_by:
            df_melted = df.melt(id_vars=[group_by], value_vars=elements, 
                              var_name='Element', value_name='Concentration')
            sns.violinplot(data=df_melted, x='Element', y='Concentration', hue=group_by, ax=axes)
        else:
            df_melted = df.melt(value_vars=elements, var_name='Element', value_name='Concentration')
            sns.violinplot(data=df_melted, x='Element', y='Concentration', ax=axes)
        axes.set_ylabel('Concentration (wt%)')
        
    elif plot_type == 'hist':
        df[elements].hist(bins=30, alpha=0.7, ax=axes)
        axes.set_xlabel('Concentration (wt%)')
        axes.set_ylabel('Frequency')
        
    elif plot_type == 'scatter':
        if len(elements) >= 2:
            axes.scatter(df[elements[0]], df[elements[1]], alpha=0.6)
            axes.set_xlabel(elements[0])
            axes.set_ylabel(elements[1])
        else:
            raise ValueError("Scatter plot requires at least 2 elements")
    
    axes.set_title(f'Element Concentrations - {plot_type.title()} Plot')
    axes.tick_params(axis='x', rotation=45)
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_geochemical_classification(df: pd.DataFrame, classification_col: str,
                                  figsize: Tuple[int, int] = (10, 6),
                                  save_path: Optional[str] = None, dpi: int = 300) -> None:
    """
    Plot geochemical classification distribution.
    
    Args:
        df: DataFrame containing geochemical data
        classification_col: Name of the classification column
        figsize: Figure size tuple
        save_path: Optional path to save the figure
        dpi: DPI for saved figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    classification_counts = df[classification_col].value_counts()
    
    bars = ax.bar(range(len(classification_counts)), classification_counts.values)
    ax.set_xticks(range(len(classification_counts)))
    ax.set_xticklabels(classification_counts.index, rotation=45)
    ax.set_ylabel('Number of Samples')
    ax.set_title(f'Distribution of {classification_col}')
    
    # Add value labels on bars
    for bar, count in zip(bars, classification_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               str(count), ha='center', va='bottom')
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, elements: List[str],
                          method: str = 'pearson', figsize: Tuple[int, int] = (12, 10),
                          save_path: Optional[str] = None, dpi: int = 300) -> None:
    """
    Plot correlation matrix for elements.
    
    Args:
        df: DataFrame containing geochemical data
        elements: List of elements to include in correlation matrix
        method: Correlation method ('pearson', 'spearman', 'kendall')
        figsize: Figure size tuple
        save_path: Optional path to save the figure
        dpi: DPI for saved figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # Calculate correlation matrix
    corr_matrix = df[elements].corr(method=method)
    
    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(elements)))
    ax.set_yticks(range(len(elements)))
    ax.set_xticklabels(elements, rotation=45)
    ax.set_yticklabels(elements)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{method.title()} Correlation')
    
    # Add correlation values as text
    for i in range(len(elements)):
        for j in range(len(elements)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha="center", va="center", color="black")
    
    ax.set_title(f'{method.title()} Correlation Matrix')
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_regression_summary(results_dict: Dict[str, Dict], 
                          regression_pairs: List[Tuple[str, str]],
                          figsize: Tuple[int, int] = (15, 10),
                          save_path: Optional[str] = None, dpi: int = 300) -> None:
    """
    Create summary plots for regression results.
    
    Args:
        results_dict: Dictionary containing regression results
        regression_pairs: List of regression pairs
        figsize: Figure size tuple
        save_path: Optional path to save the figure
        dpi: DPI for saved figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Extract data for plotting
    r2_values = []
    slope_values = []
    p_values = []
    n_obs_values = []
    
    for group_name in results_dict.keys():
        for pair in regression_pairs:
            if pair in results_dict[group_name]:
                data = results_dict[group_name][pair]
                r2_values.append(data['r_squared'])
                slope_values.append(data['coefficients'])
                p_values.append(data['p_value'])
                n_obs_values.append(data['n_obs'])
    
    # R² distribution
    axes[0, 0].hist(r2_values, bins=30, alpha=0.7, color='blue')
    axes[0, 0].set_xlabel('R²')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of R² Values')
    
    # Slope distribution
    axes[0, 1].hist(slope_values, bins=30, alpha=0.7, color='green')
    axes[0, 1].set_xlabel('Slope')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Slopes')
    
    # P-value distribution
    axes[1, 0].hist(p_values, bins=30, alpha=0.7, color='red')
    axes[1, 0].set_xlabel('P-value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of P-values')
    
    # Sample size distribution
    axes[1, 1].hist(n_obs_values, bins=30, alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Number of Observations')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Distribution of Sample Sizes')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


# Helper functions

def _filter_dictionary_by_independent_var(dictionary: Dict[Tuple[str, str], List[float]], 
                                       value: str) -> Dict[Tuple[str, str], List[float]]:
    """Filter dictionary by independent variable."""
    return {key: dictionary[key] for key in dictionary.keys() if key[0] == value}


def _filter_dictionary_by_length_of_groups(dictionary: Dict[Tuple[str, str], List[float]], 
                                         min_length: int) -> Dict[Tuple[str, str], List[float]]:
    """Filter dictionary by minimum group length."""
    return {k: v for k, v in dictionary.items() if len(v) > min_length}


def _clean_list(values: List[float]) -> np.ndarray:
    """Clean a list by removing NaN values."""
    values_array = np.array(values)
    clean = values_array[~np.isnan(values_array)]
    return clean
