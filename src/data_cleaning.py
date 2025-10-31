"""
Data Cleaning Module

This module contains functions for cleaning and organizing geochemical data from the GEOROC database.
It includes functions for data validation, volcano name matching, and adding derived columns.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from difflib import SequenceMatcher
from tqdm.auto import tqdm
import warnings

# Suppress performance warnings
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Import constants from sample module
from .sample import GeochemicalSample


def clean_data(df: pd.DataFrame, elements: List[str], loi_threshold: float = 3.5) -> pd.DataFrame:
    """
    Clean geochemical data by removing invalid values and samples.
    
    Args:
        df: DataFrame containing geochemical data
        elements: List of element column names
        loi_threshold: Maximum LOI value to retain (default: 3.5)
        
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Replace zeros with NaN
    df_clean = df_clean.replace(0, np.nan)
    
    # Replace negative concentrations with NaN
    df_clean[elements] = df_clean[elements].mask(df_clean[elements] < 0)
    
    # Drop samples without SiO2
    df_clean = df_clean.dropna(subset=['SIO2'])
    
    # Drop samples with high LOI
    if 'LOI' in df_clean.columns:
        df_clean = df_clean.loc[df_clean["LOI"] <= loi_threshold]
    
    return df_clean


def ci_norm_ree(df: pd.DataFrame, ree_elements: List[str], ci_data: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize REE elements to chondrite values.
    
    Args:
        df: DataFrame containing geochemical data
        ree_elements: List of REE element names
        ci_data: DataFrame containing chondrite values
        
    Returns:
        DataFrame with CI-normalized REE columns
    """
    df_norm = df.copy()
    ci_values = ci_data.set_index('ELEMENT')['CI_ppm']
    
    for element in ree_elements:
        if element in df_norm.columns and element in ci_values.index:
            df_norm[f'{element}_CInorm'] = df_norm[element] / ci_values[element]
    
    return df_norm


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
    df_conv[ppm_elements] = df_conv[ppm_elements] / 10000
    return df_conv


def add_geochemical_classifications(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add geochemical classification columns to the DataFrame.
    
    Args:
        df: DataFrame with normalized geochemical data
        
    Returns:
        DataFrame with added classification columns
    """
    df_class = df.copy()
    
    # SiO2 classification
    df_class['class_SIO2'] = df_class.apply(
        lambda x: _class_sio2(x['SIO2_norm']), axis=1
    )
    
    # Alkalinity classification
    df_class['class_alkalic'] = df_class.apply(
        lambda x: _class_alkalic(x['SIO2_norm'], x['NA2O_norm'], x['K2O_norm']), axis=1
    )
    
    # LeMaitre K classification
    df_class['class_LeMaitre'] = df_class.apply(
        lambda x: _class_lemaitre(x['SIO2_norm'], x['K2O_norm']), axis=1
    )
    
    # Mg number
    df_class['MG_no'] = df_class.apply(
        lambda x: _mg_number(x['MGO_norm'], x['FEO_norm'], x['FE2O3_norm'], x['FEOT_norm']), axis=1
    )
    
    # Fe index
    df_class['Fe_index'] = df_class.apply(
        lambda x: _fe_index(x['SIO2_norm'], x['FEO_norm'], x['FE2O3_norm'], x['FEOT_norm'], x['MGO_norm']), axis=1
    )
    
    # Fe index classification
    df_class['class_Fe_index'] = df_class.apply(
        lambda x: _class_fe_index(x['Fe_index'], x['SIO2_norm']), axis=1
    )
    
    # Peralkalinity index
    df_class['peralkalinity_index'] = df_class.apply(
        lambda x: _peralkalinity_index(x['NA2O_molfrac'], x['K2O_molfrac'], x['AL2O3_molfrac']), axis=1
    )
    
    return df_class


def add_volcano_names(df: pd.DataFrame, volcano_database: pd.DataFrame, 
                     similarity_threshold: float = 0.9) -> pd.DataFrame:
    """
    Add volcano names to the DataFrame using fuzzy matching.
    
    Args:
        df: DataFrame containing location data
        volcano_database: DataFrame containing volcano names
        similarity_threshold: Minimum similarity score for matching
        
    Returns:
        DataFrame with added volcano names
    """
    df_volc = df.copy()
    
    # Prepare volcano list
    volcano_list = volcano_database['Volcano Name'].str.upper().tolist()
    volcano_list = [_switch_suffixes(name) for name in volcano_list]
    
    # Create location to volcano mapping
    locations = df_volc['LOCATION'].unique()
    match_dict = {}
    
    for location in tqdm(locations, desc="Matching volcano names"):
        match_dict[location] = _match_names(location, volcano_list, similarity_threshold)
    
    # Apply volcano names
    df_volc['VOLCANO'] = df_volc['LOCATION'].map(match_dict)
    
    return df_volc


def validate_sample_completeness(df: pd.DataFrame, required_elements: List[str], 
                               min_completeness: float = 0.5) -> pd.DataFrame:
    """
    Filter samples based on data completeness.
    
    Args:
        df: DataFrame containing geochemical data
        required_elements: List of elements that should be present
        min_completeness: Minimum fraction of required elements that must be present
        
    Returns:
        Filtered DataFrame
    """
    df_valid = df.copy()
    
    # Calculate completeness for each sample
    completeness = df_valid[required_elements].notna().sum(axis=1) / len(required_elements)
    
    # Filter samples
    df_valid = df_valid[completeness >= min_completeness]
    
    return df_valid


def remove_outliers_iqr(df: pd.DataFrame, elements: List[str], 
                       iqr_factor: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers using the Interquartile Range method.
    
    Args:
        df: DataFrame containing geochemical data
        elements: List of elements to check for outliers
        iqr_factor: Factor for outlier detection (default: 1.5)
        
    Returns:
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    for element in elements:
        if element in df_clean.columns:
            Q1 = df_clean[element].quantile(0.25)
            Q3 = df_clean[element].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - iqr_factor * IQR
            upper_bound = Q3 + iqr_factor * IQR
            
            # Set outliers to NaN
            df_clean[element] = df_clean[element].where(
                (df_clean[element] >= lower_bound) & (df_clean[element] <= upper_bound)
            )
    
    return df_clean


# Helper functions for geochemical classifications

def _class_sio2(sio2: float) -> str:
    """Classify sample based on SiO2 content."""
    if pd.isna(sio2):
        return np.nan
    
    if sio2 <= 0.45:
        return 'ultrabasic'
    elif sio2 <= 0.52:
        return 'basic'
    elif sio2 <= 0.63:
        return 'intermediate'
    else:
        return 'acid'


def _class_alkalic(sio2: float, na2o: float, k2o: float) -> str:
    """Classify sample based on alkalinity."""
    if pd.isna(sio2) or sio2 > 0.774:
        return np.nan
    
    ta = na2o + k2o if pd.notna(na2o) and pd.notna(k2o) else np.nan
    if pd.isna(ta):
        return np.nan
    
    boundary_x = [0.392, 0.40, 0.432, 0.45, 0.48, 0.50, 0.537, 0.55, 0.60, 0.65, 0.774]
    boundary_y = [0, 0.04, 0.02, 0.028, 0.04, 0.0475, 0.06, 0.064, 0.08, 0.096, 0.10]
    ta_threshold = np.interp(sio2, boundary_x, boundary_y)
    
    if ta <= ta_threshold:
        return 'sub-alkalic'
    else:
        return 'alkalic'


def _class_lemaitre(sio2: float, k2o: float) -> str:
    """Classify sample based on K2O content (LeMaitre classification)."""
    if pd.isna(sio2) or pd.isna(k2o) or sio2 < 0.48:
        return np.nan
    
    low_med_boundary = [0.045, -0.0186]  # slope, intercept
    med_high_boundary = [0.095, -0.0336]  # slope, intercept
    
    med_threshold = sio2 * low_med_boundary[0] + low_med_boundary[1]
    max_threshold = sio2 * med_high_boundary[0] + med_high_boundary[1]
    
    if k2o <= med_threshold:
        return 'low-K'
    elif k2o <= max_threshold:
        return 'medium-K'
    else:
        return 'high-K'


def _mg_number(mgo: float, feo: float, fe2o3: float, feot: float) -> float:
    """Calculate Mg number."""
    if pd.isna(mgo):
        return np.nan
    
    if pd.notna(feot):
        iron = feot
    else:
        iron = feo + fe2o3 if pd.notna(feo) and pd.notna(fe2o3) else np.nan
    
    if pd.isna(iron):
        return np.nan
    
    return mgo / (mgo + iron)


def _fe_index(sio2: float, feo: float, fe2o3: float, feot: float, mgo: float) -> float:
    """Calculate Fe index."""
    if pd.isna(mgo):
        return np.nan
    
    if pd.notna(feot):
        iron = feot
    else:
        iron = feo + fe2o3 if pd.notna(feo) and pd.notna(fe2o3) else np.nan
    
    if pd.isna(iron):
        return np.nan
    
    return iron / (iron + mgo)


def _class_fe_index(fe_index: float, sio2: float) -> str:
    """Classify sample based on Fe index."""
    if pd.isna(fe_index) or pd.isna(sio2):
        return np.nan
    
    threshold = 0.46 + (0.005 * sio2)
    
    if fe_index <= threshold:
        return 'magnesian'
    else:
        return 'ferroan'


def _peralkalinity_index(na_mol: float, k_mol: float, al_mol: float) -> float:
    """Calculate peralkalinity index."""
    if pd.isna(na_mol) or pd.isna(k_mol) or pd.isna(al_mol) or al_mol == 0:
        return np.nan
    
    return (na_mol + k_mol) / al_mol


def _similarity(a: str, b: str) -> float:
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, a, b).ratio()


def _match_names(location: str, volcano_list: List[str], threshold: float) -> str:
    """Match location to volcano name using fuzzy matching."""
    location_features = location.split(' / ')
    candidates = {}
    
    for feature in location_features:
        for volcano in volcano_list:
            similarity = _similarity(volcano, feature)
            if similarity > threshold:
                candidates[volcano] = similarity
    
    if len(candidates) == 0:
        # Check for volcanic complex/field/center
        volc_complex = [item for item in location_features if 'VOLCANIC COMPLEX' in item]
        volc_field = [item for item in location_features if 'VOLCANIC FIELD' in item]
        volc_center = [item for item in location_features if 'VOLCANIC CENTER' in item]
        
        if volc_complex:
            return volc_complex[0]
        elif volc_field:
            return volc_field[0]
        elif volc_center:
            return volc_center[0]
        else:
            return np.nan
    elif len(candidates) > 1:
        return max(candidates, key=candidates.get)
    else:
        return list(candidates.keys())[0]


def _switch_suffixes(string: str) -> str:
    """Switch comma-separated suffixes in volcano names."""
    if ',' in string:
        pieces = string.split(', ')
        return f'{pieces[1]} {pieces[0]}'
    else:
        return string
