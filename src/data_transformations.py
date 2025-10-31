"""
Data Transformations Module

This module contains functions for transforming geochemical data into different
compositional spaces, including normalization, molar fractions, and center log-ratio transformations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
import warnings

# Suppress performance warnings
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Import constants from sample module
from .sample import GeochemicalSample


def normalise_totals(df: pd.DataFrame, elements: List[str]) -> pd.DataFrame:
    """
    Normalize all elements to 100% total.
    
    Args:
        df: DataFrame containing geochemical data
        elements: List of element column names
        
    Returns:
        DataFrame with normalized elements and calculated total
    """
    df_norm = df.copy()
    
    # Calculate total for each sample
    df_norm['calc_TOTAL'] = df_norm[elements].sum(axis=1)
    
    # Normalize each element
    for element in elements:
        if element in df_norm.columns:
            df_norm[f'{element}_norm'] = df_norm[element] / df_norm['calc_TOTAL']
    
    return df_norm


def molar_fraction(df: pd.DataFrame, elements: List[str], elements_mass: Dict[str, float]) -> pd.DataFrame:
    """
    Calculate molar fractions for all elements.
    
    Args:
        df: DataFrame containing geochemical data
        elements: List of element column names
        elements_mass: Dictionary mapping elements to their molecular masses
        
    Returns:
        DataFrame with molar fraction columns
    """
    df_mol = df.copy()
    
    # Calculate moles for each element
    mol_columns = {}
    for element in elements:
        if element in df_mol.columns and element in elements_mass:
            mol_columns[f'{element}_mol'] = df_mol[element] / elements_mass[element]
    
    # Add mole columns to dataframe
    for col_name, mol_values in mol_columns.items():
        df_mol[col_name] = mol_values
    
    # Calculate total moles
    mol_cols = [f'{element}_mol' for element in elements if f'{element}_mol' in df_mol.columns]
    df_mol['mol_TOT'] = df_mol[mol_cols].sum(axis=1)
    
    # Calculate molar fractions
    for element in elements:
        mol_col = f'{element}_mol'
        if mol_col in df_mol.columns:
            df_mol[f'{element}_molfrac'] = df_mol[mol_col] / df_mol['mol_TOT']
    
    # Remove intermediate mole columns
    df_mol = df_mol.drop(mol_cols, axis='columns')
    df_mol = df_mol.drop(['mol_TOT'], axis='columns')
    
    return df_mol


def clr_transform(df: pd.DataFrame, elements: List[str]) -> pd.DataFrame:
    """
    Apply center log-ratio transformation.
    
    Args:
        df: DataFrame containing normalized geochemical data
        elements: List of element column names
        
    Returns:
        DataFrame with CLR-transformed elements
    """
    df_clr = df.copy()
    
    # Get normalized element columns
    elements_norm = [f'{element}_norm' for element in elements 
                    if f'{element}_norm' in df_clr.columns]
    
    if len(elements_norm) == 0:
        raise ValueError("No normalized elements found for CLR transformation")
    
    # Calculate geometric mean for each sample
    log_values = np.log(df_clr[elements_norm])
    df_clr['geom_mean'] = np.exp(log_values.mean(axis=1))
    
    # Apply CLR transformation
    for element in elements:
        norm_col = f'{element}_norm'
        if norm_col in df_clr.columns:
            df_clr[f'{element}_clr'] = np.log(df_clr[norm_col] / df_clr['geom_mean'])
    
    return df_clr


def ppm_to_wt(ppm_value: float) -> float:
    """
    Convert ppm to weight percent.
    
    Args:
        ppm_value: Concentration in ppm
        
    Returns:
        Concentration in weight percent
    """
    return ppm_value / 10000


def convert_ppm2wtpc(df: pd.DataFrame, ppm_elements: List[str]) -> pd.DataFrame:
    """
    Convert ppm elements to weight percent.
    
    Args:
        df: DataFrame containing geochemical data
        ppm_elements: List of elements in ppm units
        
    Returns:
        DataFrame with converted values
    """
    df_conv = df.copy()
    df_conv[ppm_elements] = df_conv[ppm_elements].apply(ppm_to_wt)
    return df_conv


def calculate_geometric_mean(values: np.ndarray) -> float:
    """
    Calculate geometric mean of an array.
    
    Args:
        values: Array of values
        
    Returns:
        Geometric mean
    """
    # Remove NaN values
    clean_values = values[~np.isnan(values)]
    
    if len(clean_values) == 0:
        return np.nan
    
    return np.exp(np.mean(np.log(clean_values)))


def apply_compositional_transformation(df: pd.DataFrame, elements: List[str], 
                                     transformation_type: str = 'clr',
                                     elements_mass: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """
    Apply a compositional transformation to the data.
    
    Args:
        df: DataFrame containing geochemical data
        elements: List of element column names
        transformation_type: Type of transformation ('norm', 'molfrac', 'clr')
        elements_mass: Dictionary mapping elements to molecular masses (required for molfrac)
        
    Returns:
        DataFrame with applied transformation
    """
    df_trans = df.copy()
    
    if transformation_type == 'norm':
        df_trans = normalise_totals(df_trans, elements)
    elif transformation_type == 'molfrac':
        if elements_mass is None:
            raise ValueError("elements_mass required for molar fraction transformation")
        df_trans = normalise_totals(df_trans, elements)
        df_trans = molar_fraction(df_trans, elements, elements_mass)
    elif transformation_type == 'clr':
        df_trans = normalise_totals(df_trans, elements)
        df_trans = clr_transform(df_trans, elements)
    else:
        raise ValueError(f"Unknown transformation type: {transformation_type}")
    
    return df_trans


def batch_transform_samples(samples: List[GeochemicalSample], 
                          transformation_type: str = 'clr') -> List[GeochemicalSample]:
    """
    Apply transformations to a batch of samples.
    
    Args:
        samples: List of GeochemicalSample objects
        transformation_type: Type of transformation to apply
        
    Returns:
        List of transformed GeochemicalSample objects
    """
    transformed_samples = []
    
    for sample in samples:
        try:
            if transformation_type == 'clr':
                sample.clean().convert_ppm_to_wt().normalize_totals().clr_transform()
            elif transformation_type == 'molfrac':
                sample.clean().convert_ppm_to_wt().normalize_totals().calculate_molar_fractions()
            elif transformation_type == 'norm':
                sample.clean().convert_ppm_to_wt().normalize_totals()
            else:
                raise ValueError(f"Unknown transformation type: {transformation_type}")
            
            transformed_samples.append(sample)
        except Exception as e:
            print(f"Error transforming sample {sample.sample_id}: {e}")
            continue
    
    return transformed_samples


def validate_compositional_data(df: pd.DataFrame, elements: List[str], 
                              min_total: float = 95.0, max_total: float = 105.0) -> pd.DataFrame:
    """
    Validate compositional data by checking totals.
    
    Args:
        df: DataFrame containing geochemical data
        elements: List of element column names
        min_total: Minimum acceptable total (default: 95.0)
        max_total: Maximum acceptable total (default: 105.0)
        
    Returns:
        DataFrame with valid samples only
    """
    df_valid = df.copy()
    
    # Calculate totals
    totals = df_valid[elements].sum(axis=1)
    
    # Filter samples with acceptable totals
    valid_mask = (totals >= min_total) & (totals <= max_total)
    df_valid = df_valid[valid_mask]
    
    return df_valid


def handle_missing_data(df: pd.DataFrame, elements: List[str], 
                       method: str = 'drop', threshold: float = 0.5) -> pd.DataFrame:
    """
    Handle missing data in geochemical dataset.
    
    Args:
        df: DataFrame containing geochemical data
        elements: List of element column names
        method: Method to handle missing data ('drop', 'threshold', 'interpolate')
        threshold: Threshold for missing data (fraction of elements that must be present)
        
    Returns:
        DataFrame with handled missing data
    """
    df_handled = df.copy()
    
    if method == 'drop':
        # Drop rows with any missing values in elements
        df_handled = df_handled.dropna(subset=elements)
    elif method == 'threshold':
        # Keep rows where at least threshold fraction of elements are present
        completeness = df_handled[elements].notna().sum(axis=1) / len(elements)
        df_handled = df_handled[completeness >= threshold]
    elif method == 'interpolate':
        # Interpolate missing values (not recommended for compositional data)
        df_handled[elements] = df_handled[elements].interpolate(method='linear')
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return df_handled


def calculate_element_ratios(df: pd.DataFrame, numerator_elements: List[str], 
                           denominator_elements: List[str]) -> pd.DataFrame:
    """
    Calculate element ratios.
    
    Args:
        df: DataFrame containing geochemical data
        numerator_elements: List of elements for numerator
        denominator_elements: List of elements for denominator
        
    Returns:
        DataFrame with added ratio columns
    """
    df_ratios = df.copy()
    
    for num_element in numerator_elements:
        for den_element in denominator_elements:
            if num_element in df_ratios.columns and den_element in df_ratios.columns:
                ratio_name = f'{num_element}_{den_element}_ratio'
                df_ratios[ratio_name] = df_ratios[num_element] / df_ratios[den_element]
    
    return df_ratios


def standardize_compositional_data(df: pd.DataFrame, elements: List[str], 
                                 method: str = 'zscore') -> pd.DataFrame:
    """
    Standardize compositional data (use with caution).
    
    Args:
        df: DataFrame containing geochemical data
        elements: List of element column names
        method: Standardization method ('zscore', 'minmax')
        
    Returns:
        DataFrame with standardized data
    """
    df_std = df.copy()
    
    for element in elements:
        if element in df_std.columns:
            if method == 'zscore':
                mean_val = df_std[element].mean()
                std_val = df_std[element].std()
                df_std[f'{element}_std'] = (df_std[element] - mean_val) / std_val
            elif method == 'minmax':
                min_val = df_std[element].min()
                max_val = df_std[element].max()
                df_std[f'{element}_std'] = (df_std[element] - min_val) / (max_val - min_val)
            else:
                raise ValueError(f"Unknown standardization method: {method}")
    
    return df_std
