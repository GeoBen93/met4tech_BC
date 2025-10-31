"""
Geochemical Analysis Package

This package provides tools for cleaning, transforming, and analyzing geochemical data
from the GEOROC database. It includes modules for data cleaning, transformations,
regression analysis, and visualization.

Modules:
- sample: GeochemicalSample class for individual sample handling
- data_cleaning: Functions for cleaning and organizing geochemical data
- data_transformations: Functions for compositional data transformations
- regression_analysis: Statistical regression analysis functions
- regression_group: RegressionGroup and RegressionGroupCollection classes for managing regression results
- plotting: Visualization functions for geochemical data and results
"""

from .sample import GeochemicalSample
from .regression_group import RegressionGroup, RegressionGroupCollection
from .data_cleaning import (
    clean_data, ci_norm_ree, convert_ppm2wtpc, add_geochemical_classifications,
    add_volcano_names, validate_sample_completeness, remove_outliers_iqr
)
from .data_transformations import (
    normalise_totals, molar_fraction, clr_transform, ppm_to_wt,
    convert_ppm2wtpc, apply_compositional_transformation, batch_transform_samples,
    validate_compositional_data, handle_missing_data, calculate_element_ratios,
    standardize_compositional_data
)
from .regression_analysis import (
    linregress_group, linear_regressions, regress_multiple_elements,
    perform_regression_for_group, regress_parallel, extract_result,
    transpose_to_dict, filter_dictionary_by_independent_var,
    filter_dictionary_by_length_of_groups, calculate_regression_statistics,
    compare_grouping_methods, export_regression_results, clean_list
)
from .plotting import (
    plot_regression_boxplots, plot_survival_function, plot_regression_scatter,
    plot_element_concentrations, plot_geochemical_classification,
    plot_correlation_matrix, plot_regression_summary
)

__version__ = "1.0.0"
__author__ = "Ben Clarke"
__email__ = "ben.clarke@example.com"

__all__ = [
    # Sample class
    "GeochemicalSample",
    
    # Regression group classes
    "RegressionGroup", "RegressionGroupCollection",
    
    # Data cleaning functions
    "clean_data", "ci_norm_ree", "convert_ppm2wtpc", "add_geochemical_classifications",
    "add_volcano_names", "validate_sample_completeness", "remove_outliers_iqr",
    
    # Data transformation functions
    "normalise_totals", "molar_fraction", "clr_transform", "ppm_to_wt",
    "convert_ppm2wtpc", "apply_compositional_transformation", "batch_transform_samples",
    "validate_compositional_data", "handle_missing_data", "calculate_element_ratios",
    "standardize_compositional_data",
    
    # Regression analysis functions
    "linregress_group", "linear_regressions", "regress_multiple_elements",
    "perform_regression_for_group", "regress_parallel", "extract_result",
    "transpose_to_dict", "filter_dictionary_by_independent_var",
    "filter_dictionary_by_length_of_groups", "calculate_regression_statistics",
    "compare_grouping_methods", "export_regression_results", "clean_list",
    
    # Plotting functions
    "plot_regression_boxplots", "plot_survival_function", "plot_regression_scatter",
    "plot_element_concentrations", "plot_geochemical_classification",
    "plot_correlation_matrix", "plot_regression_summary"
]
