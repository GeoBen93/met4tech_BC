"""
Regression Analysis Module

This module contains functions for performing statistical regression analyses on geochemical data.
It includes parallel processing capabilities for large datasets and comprehensive result extraction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .regression_group import RegressionGroupCollection
from scipy import stats
import statsmodels.api as sm
from joblib import Parallel, delayed
import itertools
import time
import warnings

# Suppress performance warnings
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def linregress_group(group: pd.DataFrame, x_element: str, y_element: str) -> pd.Series:
    """
    Perform linear regression on a group of data.
    
    Args:
        group: DataFrame containing the group data
        x_element: Name of the independent variable column
        y_element: Name of the dependent variable column
        
    Returns:
        Series containing regression statistics
    """
    model = stats.linregress(group[x_element], group[y_element])
    return pd.Series({
        f'{x_element}_{y_element}_slope': model.slope,
        f'{x_element}_{y_element}_intercept': model.intercept,
        f'{x_element}_{y_element}_r2': model.rvalue**2,
        f'{x_element}_{y_element}_pval': model.pvalue,
        f'{x_element}_{y_element}_stderr': model.stderr,
        f'{x_element}_{y_element}_n': len(group)
    })


def linear_regressions(df: pd.DataFrame, group_feature: str, x_element: str, y_element: str) -> pd.DataFrame:
    """
    Perform linear regressions grouped by a feature.
    
    Args:
        df: DataFrame containing geochemical data
        group_feature: Column name to group by
        x_element: Name of the independent variable column
        y_element: Name of the dependent variable column
        
    Returns:
        DataFrame containing regression results for each group
    """
    # Drop samples without required data
    df_temp = df.dropna(subset=[x_element, y_element])
    
    # Filter groups with at least 5 samples
    df_temp = df_temp.groupby(group_feature).filter(lambda x: len(x[x_element]) >= 5)
    
    # Perform linear regression
    result = df_temp.groupby(group_feature).apply(
        lambda x: linregress_group(x, x_element, y_element), 
        include_groups=False
    ).reset_index()
    
    return result


def regress_multiple_elements(df: pd.DataFrame, group_feature: str, 
                            x_elements: List[str], y_elements: List[str],
                            data_suffix: str = '_clr', min_samples: int = 5) -> pd.DataFrame:
    """
    Perform regressions between multiple element pairs.
    
    Args:
        df: DataFrame containing geochemical data
        group_feature: Column name to group by
        x_elements: List of independent variable elements
        y_elements: List of dependent variable elements
        data_suffix: Suffix for data columns (e.g., '_clr', '_norm')
        min_samples: Minimum number of samples required per group
        
    Returns:
        DataFrame containing all regression results
    """
    # Add suffix to element names
    x_elements_suffixed = [f'{element}{data_suffix}' for element in x_elements]
    y_elements_suffixed = [f'{element}{data_suffix}' for element in y_elements]
    
    iteration = 0
    df_group_reg = None
    
    for x_counter, x_element in enumerate(x_elements_suffixed, 1):
        for y_counter, y_element in enumerate(y_elements_suffixed, 1):
            if x_element == y_element:  # Skip same elements
                continue
            
            print(f'X element: {x_counter}/{len(x_elements_suffixed)} {x_element}     '
                  f'Y element: {y_counter}/{len(y_elements_suffixed)} {y_element}')
            
            if iteration == 0:
                df_group_reg = linear_regressions(df, group_feature, x_element, y_element)
                iteration += 1
            else:
                new_reg = linear_regressions(df, group_feature, x_element, y_element)
                df_group_reg = df_group_reg.merge(new_reg, on=group_feature, how='outer')
    
    print(f'X element: {x_counter}/{len(x_elements_suffixed)} {x_element}     '
          f'Y element: {y_counter}/{len(y_elements_suffixed)} {y_element}     DONE')
    
    return df_group_reg


def perform_regression_for_group(group_name: str, group_data: pd.DataFrame, 
                               independent_vars: List[str], dependent_vars: List[str]) -> Optional[Tuple[str, Dict]]:
    """
    Perform regressions for a single group using OLS.
    
    Args:
        group_name: Name of the group
        group_data: DataFrame containing group data
        independent_vars: List of independent variable names
        dependent_vars: List of dependent variable names
        
    Returns:
        Tuple of (group_name, results_dict) or None if no valid regressions
    """
    group_results = {}
    
    for x_var in independent_vars:
        for y_var in dependent_vars:
            if x_var == y_var:
                continue  # Skip if independent and dependent variables are the same
            
            # Select valid rows
            valid_rows = ~group_data[[x_var, y_var]].isna().any(axis=1)
            
            if valid_rows.sum() < 2:  # Need at least two data points
                continue
            
            X = group_data.loc[valid_rows, x_var].values
            y = group_data.loc[valid_rows, y_var].values
            
            if len(np.unique(X)) == 1:  # Skip if X has no variation
                continue
            
            # Add constant (intercept) to the model
            X_with_const = sm.add_constant(X)
            
            # Perform OLS linear regression
            model = sm.OLS(y, X_with_const).fit()
            
            if len(model.params) < 2:  # Check if there's a slope coefficient
                continue
            
            # Store the result
            group_results[(x_var, y_var)] = {
                'coefficients': model.params[1],  # Coefficient for the independent variable
                'intercept': model.params[0],  # Intercept
                'p_value': model.pvalues[1],  # p-value for the independent variable
                'r_squared': model.rsquared,  # R^2 score
                'mse': model.mse_resid,  # Mean squared error of the residuals
                'n_obs': model.nobs  # Number of observations
            }
    
    if group_results:
        return group_name, group_results
    else:
        return None


def regress_parallel(df: pd.DataFrame, group_column: str, independent_vars: List[str], 
                    dependent_vars: Optional[List[str]] = None, n_jobs: int = -1,
                    return_groups: bool = False) -> Union[Dict[str, Dict], 'RegressionGroupCollection']:
    """
    Perform parallel regressions across groups.
    
    Args:
        df: DataFrame containing geochemical data
        group_column: Column name to group by
        independent_vars: List of independent variable names
        dependent_vars: List of dependent variable names (if None, uses all columns except independent vars)
        n_jobs: Number of parallel jobs (-1 for all cores)
        return_groups: If True, return RegressionGroupCollection instead of dict (default: False)
        
    Returns:
        Dictionary containing regression results for each group, or RegressionGroupCollection if return_groups=True
    """
    # Import here to avoid circular imports
    from .regression_group import RegressionGroupCollection
    
    grouped = df.groupby(group_column)
    
    if dependent_vars is None:
        # Use all columns except independent vars and group column
        dependent_vars = [col for col in df.columns if col != group_column]
    
    # Parallel processing across groups
    results = Parallel(n_jobs=n_jobs)(
        delayed(perform_regression_for_group)(
            group_name, 
            group_data.drop(columns=[group_column]), 
            independent_vars, 
            dependent_vars
        )
        for group_name, group_data in grouped
    )
    
    # Filter out any None values from the results
    valid_results = [result for result in results if result is not None]
    
    # Convert list of tuples to dictionary
    results_dict = {group_name: result for group_name, result in valid_results}
    
    if return_groups:
        return RegressionGroupCollection(results_dict)
    else:
        return results_dict


def extract_result(results: Dict[str, Dict], regression_pair: Tuple[str, str], 
                  parameter: str, threshold_parameter: Optional[str] = None, 
                  threshold: Optional[float] = None, threshold_parameter2: Optional[str] = None, 
                  threshold2: Optional[float] = None) -> List[float]:
    """
    Extract specific regression parameters with optional filtering.
    
    Args:
        results: Dictionary containing regression results
        regression_pair: Tuple of (independent_var, dependent_var)
        parameter: Parameter to extract ('coefficients', 'intercept', 'p_value', 'r_squared', 'mse', 'n_obs')
        threshold_parameter: First parameter for filtering
        threshold: First threshold value
        threshold_parameter2: Second parameter for filtering
        threshold2: Second threshold value
        
    Returns:
        List of parameter values
    """
    parameter_list = []
    
    for group_item in results.keys():
        if regression_pair in results[group_item]:
            if threshold_parameter is None and threshold_parameter2 is None:
                parameter_list.append(results[group_item][regression_pair][parameter])
            elif threshold_parameter is not None and threshold_parameter2 is None:
                if results[group_item][regression_pair][threshold_parameter] > threshold:
                    parameter_list.append(results[group_item][regression_pair][parameter])
            else:
                if (results[group_item][regression_pair][threshold_parameter] > threshold and
                    results[group_item][regression_pair][threshold_parameter2] > threshold2):
                    parameter_list.append(results[group_item][regression_pair][parameter])
    
    return parameter_list


def transpose_to_dict(results: Dict[str, Dict], regression_pairs: List[Tuple[str, str]], 
                     parameter: str, threshold_parameter: Optional[str] = None, 
                     threshold: Optional[float] = None, threshold_parameter2: Optional[str] = None, 
                     threshold2: Optional[float] = None) -> Dict[Tuple[str, str], List[float]]:
    """
    Transpose results to dictionary format for analysis.
    
    Args:
        results: Dictionary containing regression results
        regression_pairs: List of regression pairs
        parameter: Parameter to extract
        threshold_parameter: First parameter for filtering
        threshold: First threshold value
        threshold_parameter2: Second parameter for filtering
        threshold2: Second threshold value
        
    Returns:
        Dictionary mapping regression pairs to parameter lists
    """
    result_dict = {}
    for regression_pair in regression_pairs:
        result_dict[regression_pair] = extract_result(
            results, regression_pair, parameter, 
            threshold_parameter, threshold, threshold_parameter2, threshold2
        )
    return result_dict


def filter_dictionary_by_independent_var(dictionary: Dict[Tuple[str, str], List[float]], 
                                       value: str) -> Dict[Tuple[str, str], List[float]]:
    """
    Filter dictionary by independent variable.
    
    Args:
        dictionary: Dictionary to filter
        value: Independent variable value to filter by
        
    Returns:
        Filtered dictionary
    """
    return {key: dictionary[key] for key in dictionary.keys() if key[0] == value}


def filter_dictionary_by_length_of_groups(dictionary: Dict[Tuple[str, str], List[float]], 
                                         min_length: int) -> Dict[Tuple[str, str], List[float]]:
    """
    Filter dictionary by minimum group length.
    
    Args:
        dictionary: Dictionary to filter
        min_length: Minimum length threshold
        
    Returns:
        Filtered dictionary
    """
    return {k: v for k, v in dictionary.items() if len(v) > min_length}


def calculate_regression_statistics(results: Dict[str, Dict], 
                                  regression_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    Calculate summary statistics for regression results.
    
    Args:
        results: Dictionary containing regression results
        regression_pairs: List of regression pairs
        
    Returns:
        Dictionary containing summary statistics
    """
    stats_dict = {}
    
    for pair in regression_pairs:
        pair_stats = {
            'total_groups': 0,
            'significant_groups': 0,
            'mean_r2': 0,
            'mean_slope': 0,
            'mean_p_value': 0,
            'median_n_obs': 0
        }
        
        r2_values = []
        slope_values = []
        p_values = []
        n_obs_values = []
        
        for group_name in results.keys():
            if pair in results[group_name]:
                pair_stats['total_groups'] += 1
                group_data = results[group_name][pair]
                
                r2_values.append(group_data['r_squared'])
                slope_values.append(group_data['coefficients'])
                p_values.append(group_data['p_value'])
                n_obs_values.append(group_data['n_obs'])
                
                if group_data['p_value'] < 0.05:  # Significant at 5% level
                    pair_stats['significant_groups'] += 1
        
        if r2_values:
            pair_stats['mean_r2'] = np.mean(r2_values)
            pair_stats['mean_slope'] = np.mean(slope_values)
            pair_stats['mean_p_value'] = np.mean(p_values)
            pair_stats['median_n_obs'] = np.median(n_obs_values)
        
        stats_dict[pair] = pair_stats
    
    return stats_dict


def compare_grouping_methods(df: pd.DataFrame, group_features: List[str], 
                           independent_vars: List[str], dependent_vars: List[str]) -> Dict[str, Dict]:
    """
    Compare regression results across different grouping methods.
    
    Args:
        df: DataFrame containing geochemical data
        group_features: List of grouping features to compare
        independent_vars: List of independent variable names
        dependent_vars: List of dependent variable names
        
    Returns:
        Dictionary containing results for each grouping method
    """
    comparison_results = {}
    
    for group_feature in group_features:
        print(f"Processing grouping by: {group_feature}")
        start_time = time.time()
        
        results = regress_parallel(df, group_feature, independent_vars, dependent_vars)
        
        end_time = time.time()
        print(f"Completed in {end_time - start_time:.2f} seconds")
        
        comparison_results[group_feature] = results
    
    return comparison_results


def export_regression_results(results: Dict[str, Dict], output_file: str) -> None:
    """
    Export regression results to CSV file.
    
    Args:
        results: Dictionary containing regression results
        output_file: Path to output CSV file
    """
    # Convert results to DataFrame format
    rows = []
    
    for group_name, group_results in results.items():
        for regression_pair, regression_data in group_results.items():
            row = {
                'group': group_name,
                'independent_var': regression_pair[0],
                'dependent_var': regression_pair[1],
                **regression_data
            }
            rows.append(row)
    
    df_results = pd.DataFrame(rows)
    df_results.to_csv(output_file, index=False)
    print(f"Results exported to {output_file}")


def clean_list(values: List[float]) -> np.ndarray:
    """
    Clean a list by removing NaN values.
    
    Args:
        values: List of values
        
    Returns:
        Cleaned numpy array
    """
    values_array = np.array(values)
    clean = values_array[~np.isnan(values_array)]
    return clean
