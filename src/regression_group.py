"""
Regression Group Module

This module provides classes for managing and analyzing regression results organized by groups.
It includes RegressionGroup for individual groups and RegressionGroupCollection for managing
multiple groups, enabling easy inspection, filtering, and visualization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import warnings

# Suppress performance warnings
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class RegressionGroup:
    """
    Represents a single group of regression analyses (e.g., all regressions for one ARC_CODE).
    
    Attributes:
        group_id: Identifier for the group (e.g., ARC_CODE value)
        regressions: Dictionary mapping (independent_var, dependent_var) tuples to regression results
        metadata: Optional metadata about the group
    """
    
    def __init__(self, group_id: Union[str, int], regressions: Dict[Tuple[str, str], Dict], 
                 metadata: Optional[Dict] = None):
        """
        Initialize a RegressionGroup.
        
        Args:
            group_id: Unique identifier for the group
            regressions: Dictionary mapping (ind_var, dep_var) tuples to regression result dicts
            metadata: Optional metadata dictionary
        """
        self.group_id = str(group_id) if group_id is not None else None
        self.regressions = regressions.copy() if regressions else {}
        self.metadata = metadata.copy() if metadata else {}
    
    def get_regression(self, independent_var: str, dependent_var: str) -> Optional[Dict]:
        """
        Get regression results for a specific element pair.
        
        Args:
            independent_var: Name of the independent variable
            dependent_var: Name of the dependent variable
            
        Returns:
            Regression result dictionary or None if not found
        """
        return self.regressions.get((independent_var, dependent_var))
    
    def list_dependent_vars(self, independent_var: str) -> List[str]:
        """
        List all dependent variables for a given independent variable.
        
        Args:
            independent_var: Name of the independent variable
            
        Returns:
            List of dependent variable names
        """
        return [dep for ind, dep in self.regressions.keys() if ind == independent_var]
    
    def list_independent_vars(self) -> List[str]:
        """
        List all independent variables in this group.
        
        Returns:
            List of unique independent variable names
        """
        return sorted(list(set(ind for ind, dep in self.regressions.keys())))
    
    def list_all_pairs(self) -> List[Tuple[str, str]]:
        """
        List all regression pairs (independent, dependent) in this group.
        
        Returns:
            List of (independent_var, dependent_var) tuples
        """
        return list(self.regressions.keys())
    
    def filter_significant(self, p_threshold: float = 0.05) -> 'RegressionGroup':
        """
        Return new group with only significant regressions.
        
        Args:
            p_threshold: Maximum p-value to consider significant
            
        Returns:
            New RegressionGroup with filtered regressions
        """
        filtered = {
            pair: result for pair, result in self.regressions.items()
            if result.get('p_value', 1.0) < p_threshold
        }
        return RegressionGroup(self.group_id, filtered, self.metadata)
    
    def filter_by_min_samples(self, min_samples: int = 5) -> 'RegressionGroup':
        """
        Return new group with only regressions meeting minimum sample size.
        
        Args:
            min_samples: Minimum number of observations required
            
        Returns:
            New RegressionGroup with filtered regressions
        """
        filtered = {
            pair: result for pair, result in self.regressions.items()
            if result.get('n_obs', 0) >= min_samples
        }
        return RegressionGroup(self.group_id, filtered, self.metadata)
    
    def filter_by_r2(self, min_r2: float = 0.0) -> 'RegressionGroup':
        """
        Return new group with only regressions meeting minimum R².
        
        Args:
            min_r2: Minimum R² value required
            
        Returns:
            New RegressionGroup with filtered regressions
        """
        filtered = {
            pair: result for pair, result in self.regressions.items()
            if result.get('r_squared', -1.0) >= min_r2
        }
        return RegressionGroup(self.group_id, filtered, self.metadata)
    
    def get_regression_stat(self, independent_var: str, dependent_var: str, 
                           stat_name: str) -> Optional[float]:
        """
        Get a specific statistic for a regression pair.
        
        Args:
            independent_var: Name of the independent variable
            dependent_var: Name of the dependent variable
            stat_name: Name of the statistic ('coefficients', 'r_squared', 'p_value', etc.)
            
        Returns:
            Statistic value or None if not found
        """
        result = self.get_regression(independent_var, dependent_var)
        if result is None:
            return None
        return result.get(stat_name)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for all regressions in this group.
        
        Returns:
            Dictionary containing summary statistics
        """
        if not self.regressions:
            return {
                'n_regressions': 0,
                'mean_r2': np.nan,
                'median_r2': np.nan,
                'mean_p_value': np.nan,
                'significant_count': 0,
                'mean_slope': np.nan,
                'median_slope': np.nan,
                'mean_n_obs': np.nan
            }
        
        r2_values = [r['r_squared'] for r in self.regressions.values() if 'r_squared' in r]
        p_values = [r['p_value'] for r in self.regressions.values() if 'p_value' in r]
        slopes = [r['coefficients'] for r in self.regressions.values() if 'coefficients' in r]
        n_obs_values = [r['n_obs'] for r in self.regressions.values() if 'n_obs' in r]
        
        return {
            'n_regressions': len(self.regressions),
            'mean_r2': np.mean(r2_values) if r2_values else np.nan,
            'median_r2': np.median(r2_values) if r2_values else np.nan,
            'mean_p_value': np.mean(p_values) if p_values else np.nan,
            'significant_count': sum(1 for p in p_values if p < 0.05) if p_values else 0,
            'mean_slope': np.mean(slopes) if slopes else np.nan,
            'median_slope': np.median(slopes) if slopes else np.nan,
            'mean_n_obs': np.mean(n_obs_values) if n_obs_values else np.nan
        }
    
    def to_dict(self) -> Dict:
        """
        Convert to dictionary format (for backward compatibility).
        
        Returns:
            Dictionary in format {group_id: regressions_dict}
        """
        return {self.group_id: self.regressions}
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all regressions to a DataFrame.
        
        Returns:
            DataFrame with one row per regression pair
        """
        rows = []
        for (ind, dep), result in self.regressions.items():
            row = {
                'group_id': self.group_id,
                'independent_var': ind,
                'dependent_var': dep,
                **result
            }
            rows.append(row)
        
        if not rows:
            return pd.DataFrame(columns=['group_id', 'independent_var', 'dependent_var'])
        
        return pd.DataFrame(rows)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"RegressionGroup(id={self.group_id}, n_regressions={len(self.regressions)})"
    
    def __len__(self) -> int:
        """Return number of regressions in this group."""
        return len(self.regressions)


class RegressionGroupCollection:
    """
    Manages a collection of RegressionGroup objects.
    
    Provides methods for extraction, filtering, and visualization across multiple groups.
    """
    
    def __init__(self, data: Union[Dict, List[RegressionGroup]]):
        """
        Initialize RegressionGroupCollection.
        
        Args:
            data: Either a dictionary from regress_parallel() format {group_id: {pairs: results}},
                  or a list of RegressionGroup objects
        """
        self.groups: Dict[str, RegressionGroup] = {}
        
        if isinstance(data, dict):
            # Convert from regress_parallel() output format
            for group_id, regressions in data.items():
                if isinstance(regressions, dict):
                    self.groups[str(group_id)] = RegressionGroup(group_id, regressions)
        elif isinstance(data, list):
            # Convert from list of RegressionGroup objects
            for group in data:
                if isinstance(group, RegressionGroup):
                    self.groups[group.group_id] = group
    
    def get_group(self, group_id: Union[str, int]) -> Optional[RegressionGroup]:
        """
        Get a specific group by ID.
        
        Args:
            group_id: Identifier for the group
            
        Returns:
            RegressionGroup or None if not found
        """
        return self.groups.get(str(group_id))
    
    def list_groups(self) -> List[str]:
        """
        List all group IDs in the collection.
        
        Returns:
            List of group IDs
        """
        return sorted(self.groups.keys())
    
    def add_group(self, group: RegressionGroup) -> None:
        """
        Add a group to the collection.
        
        Args:
            group: RegressionGroup to add
        """
        if group.group_id is not None:
            self.groups[group.group_id] = group
    
    def remove_group(self, group_id: Union[str, int]) -> Optional[RegressionGroup]:
        """
        Remove a group from the collection.
        
        Args:
            group_id: Identifier for the group to remove
            
        Returns:
            Removed RegressionGroup or None if not found
        """
        return self.groups.pop(str(group_id), None)
    
    def extract_for_boxplots(self, parameter: str = 'coefficients',
                            p_threshold: Optional[float] = None,
                            min_samples: Optional[int] = None,
                            min_r2: Optional[float] = None) -> Dict[Tuple[str, str], List[float]]:
        """
        Extract data for boxplot visualization.
        
        Extracts specified parameter across all groups for each regression pair,
        with optional filtering by p-value, sample size, or R².
        
        Args:
            parameter: Parameter to extract ('coefficients', 'r_squared', 'p_value', etc.)
            p_threshold: Optional maximum p-value threshold (only include significant regressions)
            min_samples: Optional minimum number of observations required
            min_r2: Optional minimum R² value required
            
        Returns:
            Dictionary mapping (independent_var, dependent_var) tuples to lists of parameter values
        """
        result_dict: Dict[Tuple[str, str], List[float]] = {}
        
        # Collect all unique regression pairs across all groups
        all_pairs = set()
        for group in self.groups.values():
            all_pairs.update(group.list_all_pairs())
        
        # Extract parameter for each pair across all groups
        for pair in all_pairs:
            values = []
            for group in self.groups.values():
                regression_result = group.get_regression(pair[0], pair[1])
                
                if regression_result is None:
                    continue
                
                # Apply filters
                if p_threshold is not None:
                    if regression_result.get('p_value', 1.0) >= p_threshold:
                        continue
                
                if min_samples is not None:
                    if regression_result.get('n_obs', 0) < min_samples:
                        continue
                
                if min_r2 is not None:
                    if regression_result.get('r_squared', -1.0) < min_r2:
                        continue
                
                # Extract parameter value
                if parameter in regression_result:
                    value = regression_result[parameter]
                    if not np.isnan(value) if isinstance(value, (float, np.floating)) else True:
                        values.append(value)
            
            if values:  # Only add if we have at least one value
                result_dict[pair] = values
        
        return result_dict
    
    def filter_groups(self, condition_func: Callable[[RegressionGroup], bool]) -> 'RegressionGroupCollection':
        """
        Filter groups based on a condition function.
        
        Args:
            condition_func: Function that takes a RegressionGroup and returns True to keep it
            
        Returns:
            New RegressionGroupCollection with filtered groups
        """
        filtered_groups = [
            group for group in self.groups.values()
            if condition_func(group)
        ]
        return RegressionGroupCollection(filtered_groups)
    
    def filter_all_significant(self, p_threshold: float = 0.05) -> 'RegressionGroupCollection':
        """
        Return collection where all groups have been filtered to only significant regressions.
        
        Args:
            p_threshold: Maximum p-value to consider significant
            
        Returns:
            New RegressionGroupCollection with filtered groups
        """
        filtered_groups = [
            group.filter_significant(p_threshold)
            for group in self.groups.values()
        ]
        return RegressionGroupCollection(filtered_groups)
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics across all groups in the collection.
        
        Returns:
            Dictionary containing collection-wide statistics
        """
        if not self.groups:
            return {
                'n_groups': 0,
                'total_regressions': 0,
                'mean_regressions_per_group': np.nan
            }
        
        total_regressions = sum(len(group) for group in self.groups.values())
        
        # Collect stats from all groups
        all_r2 = []
        all_p_values = []
        all_slopes = []
        
        for group in self.groups.values():
            group_stats = group.get_summary_stats()
            # Get individual regression stats for aggregation
            for pair, result in group.regressions.items():
                if 'r_squared' in result:
                    all_r2.append(result['r_squared'])
                if 'p_value' in result:
                    all_p_values.append(result['p_value'])
                if 'coefficients' in result:
                    all_slopes.append(result['coefficients'])
        
        return {
            'n_groups': len(self.groups),
            'total_regressions': total_regressions,
            'mean_regressions_per_group': total_regressions / len(self.groups) if self.groups else 0,
            'mean_r2': np.mean(all_r2) if all_r2 else np.nan,
            'median_r2': np.median(all_r2) if all_r2 else np.nan,
            'mean_p_value': np.mean(all_p_values) if all_p_values else np.nan,
            'significant_count': sum(1 for p in all_p_values if p < 0.05) if all_p_values else 0,
            'mean_slope': np.mean(all_slopes) if all_slopes else np.nan,
            'median_slope': np.median(all_slopes) if all_slopes else np.nan
        }
    
    def compare_groups(self, group_id1: Union[str, int], group_id2: Union[str, int],
                      independent_var: str, dependent_var: str) -> Optional[Dict[str, Any]]:
        """
        Compare a specific regression pair between two groups.
        
        Args:
            group_id1: ID of first group
            group_id2: ID of second group
            independent_var: Name of independent variable
            dependent_var: Name of dependent variable
            
        Returns:
            Dictionary with comparison statistics or None if either group/pair not found
        """
        group1 = self.get_group(group_id1)
        group2 = self.get_group(group_id2)
        
        if not group1 or not group2:
            return None
        
        result1 = group1.get_regression(independent_var, dependent_var)
        result2 = group2.get_regression(independent_var, dependent_var)
        
        if not result1 or not result2:
            return None
        
        return {
            'group1_id': group1.group_id,
            'group2_id': group2.group_id,
            'slope_diff': result1['coefficients'] - result2['coefficients'],
            'r2_diff': result1['r_squared'] - result2['r_squared'],
            'p_value_group1': result1['p_value'],
            'p_value_group2': result2['p_value'],
            'n_obs_group1': result1['n_obs'],
            'n_obs_group2': result2['n_obs']
        }
    
    def plot_boxplots(self, independent_vars: List[str], group_feature: str = 'Group',
                     parameter: str = 'coefficients',
                     p_value_threshold: Optional[float] = None,
                     n_samples_threshold: Optional[int] = None,
                     min_length: int = 5, figsize: Tuple[int, int] = (15, 12),
                     save_path: Optional[str] = None, dpi: int = 300) -> None:
        """
        Convenience method to create boxplots from this collection.
        
        Args:
            independent_vars: List of independent variable names to plot
            group_feature: Name of the grouping feature (for plot title)
            parameter: Parameter to extract for boxplots
            p_value_threshold: Optional p-value threshold for filtering
            n_samples_threshold: Optional minimum samples threshold
            min_length: Minimum number of groups per boxplot
            figsize: Figure size tuple
            save_path: Optional path to save the figure
            dpi: DPI for saved figure
        """
        from .plotting import plot_regression_boxplots
        
        boxplot_data = self.extract_for_boxplots(
            parameter=parameter,
            p_threshold=p_value_threshold,
            min_samples=n_samples_threshold
        )
        
        plot_regression_boxplots(
            boxplot_data,
            independent_vars=independent_vars,
            group_feature=group_feature,
            p_value_threshold=p_value_threshold if p_value_threshold else 0.95,
            n_samples_threshold=n_samples_threshold if n_samples_threshold else 5,
            min_length=min_length,
            figsize=figsize,
            save_path=save_path,
            dpi=dpi
        )
    
    def to_dict(self) -> Dict:
        """
        Convert to dictionary format (backward compatible with regress_parallel output).
        
        Returns:
            Dictionary in format {group_id: {pairs: results}}
        """
        result = {}
        for group_id, group in self.groups.items():
            result[group_id] = group.regressions
        return result
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all groups to a single DataFrame.
        
        Returns:
            DataFrame with one row per regression pair across all groups
        """
        dfs = []
        for group in self.groups.values():
            df = group.to_dataframe()
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame(columns=['group_id', 'independent_var', 'dependent_var'])
        
        return pd.concat(dfs, ignore_index=True)
    
    def export_to_csv(self, output_file: str) -> None:
        """
        Export all regression results to a CSV file.
        
        Args:
            output_file: Path to output CSV file
        """
        df = self.to_dataframe()
        df.to_csv(output_file, index=False)
        print(f"Results exported to {output_file}")
    
    def __repr__(self) -> str:
        """String representation."""
        return f"RegressionGroupCollection(n_groups={len(self.groups)})"
    
    def __len__(self) -> int:
        """Return number of groups in collection."""
        return len(self.groups)
    
    def __iter__(self):
        """Iterate over groups."""
        return iter(self.groups.values())
    
    def __getitem__(self, group_id: Union[str, int]) -> RegressionGroup:
        """Get group by ID using bracket notation."""
        return self.groups[str(group_id)]

