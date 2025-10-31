"""
Geochemical Sample Class

This module provides a GeochemicalSample class for handling individual geochemical samples
from the GEOROC database. The class encapsulates all sample-specific operations including
cleaning, transformation, and classification.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import warnings

# Suppress performance warnings
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


class GeochemicalSample:
    """
    A class to represent a single geochemical sample from the GEOROC database.
    
    This class encapsulates all operations that can be performed on a single sample,
    including data cleaning, transformations, and geochemical classifications.
    
    Attributes:
        sample_id (str): Unique identifier for the sample
        raw_data (pd.Series): Raw geochemical data
        cleaned_data (pd.Series): Cleaned geochemical data
        transformed_data (Dict): Dictionary containing various transformations
        classifications (Dict): Geochemical classifications
        metadata (Dict): Additional sample metadata
    """
    
    # Element lists and constants
    ELEMENTS = ['SIO2', 'TIO2', 'B2O3', 'AL2O3', 'CR2O3', 'FE2O3', 'FEO', 'FEOT', 'CAO', 'MGO',
                'MNO', 'NIO', 'K2O', 'NA2O', 'P2O5', 'H2O', 'CO2', 'CO1', 'F', 'CL', 'CL2', 'OH',
                'CH4', 'SO2', 'SO3', 'SO4', 'S', 'LI', 'BE', 'B', 'C', 'NA', 'MG',
                'AL', 'P', 'K', 'CA', 'SC', 'TI', 'V', 'CR', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN',
                'GA', 'GE', 'AS', 'SE', 'BR', 'RB', 'SR', 'Y', 'ZR', 'NB', 'MO', 'RU', 'RH', 'PD',
                'AG', 'CD', 'IN', 'SN', 'SB', 'TE', 'I', 'CS', 'BA', 'LA', 'CE', 'PR', 'ND', 'SM',
                'EU', 'GD', 'TB', 'DY', 'HO', 'ER', 'TM', 'YB', 'LU', 'HF', 'TA', 'W', 'RE', 'OS',
                'IR', 'PT', 'AU', 'HG', 'TL', 'PB', 'BI', 'TH', 'U']
    
    PPM_ELEMENTS = ['LI', 'BE', 'B', 'C', 'NA', 'MG', 'AL', 'P', 'K', 'CA', 'SC', 
                    'TI', 'V', 'CR', 'MN', 'FE', 'CO', 'NI', 'CU', 'ZN', 'GA', 'GE', 'AS', 'SE',
                    'BR', 'RB', 'SR', 'Y', 'ZR', 'NB', 'MO', 'RU', 'RH', 'PD', 'AG', 'CD', 'IN',
                    'SN', 'SB', 'TE', 'I', 'CS', 'BA', 'LA', 'CE', 'PR', 'ND', 'SM', 'EU', 'GD',
                    'TB', 'DY', 'HO', 'ER', 'TM', 'YB', 'LU', 'HF', 'TA', 'W', 'RE', 'OS', 'IR',
                    'PT', 'AU', 'HG', 'TL', 'PB', 'BI', 'TH', 'U']
    
    REE_ELEMENTS = ['LA', 'CE', 'PR', 'ND', 'SM', 'EU', 'GD', 'TB', 'DY', 'HO', 'ER', 'TM', 'YB', 'LU']
    REE_ELEMENTS_LAMBDA = ['LA', 'CE', 'PR', 'ND', 'SM', 'GD', 'TB', 'DY', 'HO', 'ER', 'TM', 'YB', 'LU']  # exclude Eu
    
    ELEMENTS_MASS = {
        'SIO2': 60.08, 'TIO2': 79.87, 'B2O3': 69.62, 'AL2O3': 101.96, 'CR2O3': 151.99, 'FE2O3': 159.69,
        'FEO': 71.85, 'FEOT': 87.85, 'CAO': 56.08, 'MGO': 40.30, 'MNO': 70.94, 'NIO': 74.71,
        'K2O': 94.20, 'NA2O': 61.98, 'P2O5': 141.94, 'H2O': 18.02, 'CO2': 44.01, 'CO1': 28.01,
        'F': 18.99, 'CL': 35.45, 'CL2': 70.90, 'OH': 17.01, 'CH4': 16.04, 'SO2': 64.07,
        'SO3': 80.06, 'SO4': 96.06, 'S': 32.07, 'LI': 6.94, 'BE': 9.01, 'B': 10.81, 'C': 12.01,
        'NA': 22.99, 'MG': 24.31, 'AL': 26.98, 'P': 30.97, 'K': 39.10, 'CA': 40.08, 'SC': 44.96,
        'TI': 47.87, 'V': 50.94, 'CR': 52.00, 'MN': 54.94, 'FE': 55.85, 'CO': 58.93, 'NI': 58.69,
        'CU': 63.55, 'ZN': 65.38, 'GA': 69.72, 'GE': 72.63, 'AS': 74.92, 'SE': 78.97, 'BR': 79.90,
        'RB': 85.47, 'SR': 87.62, 'Y': 88.91, 'ZR': 91.22, 'NB': 92.91, 'MO': 95.94, 'RU': 101.07,
        'RH': 102.91, 'PD': 106.42, 'AG': 107.87, 'CD': 112.41, 'IN': 114.82, 'SN': 118.71,
        'SB': 121.76, 'TE': 127.60, 'I': 126.90, 'CS': 132.91, 'BA': 137.33, 'LA': 138.91,
        'CE': 140.12, 'PR': 140.91, 'ND': 144.24, 'SM': 150.36, 'EU': 151.96, 'GD': 157.25,
        'TB': 158.93, 'DY': 162.50, 'HO': 164.93, 'ER': 167.26, 'TM': 168.93, 'YB': 173.05,
        'LU': 174.97, 'HF': 178.49, 'TA': 180.95, 'W': 183.84, 'RE': 186.21, 'OS': 190.23,
        'IR': 192.22, 'PT': 195.08, 'AU': 196.97, 'HG': 200.59, 'TL': 204.38, 'PB': 207.2,
        'BI': 208.98, 'TH': 232.04, 'U': 238.03
    }
    
    def __init__(self, sample_data: Union[pd.Series, Dict], sample_id: Optional[str] = None):
        """
        Initialize a GeochemicalSample.
        
        Args:
            sample_data: Geochemical data as pandas Series or dictionary
            sample_id: Optional unique identifier for the sample. If not provided,
                      will attempt to use GEOROC_ID from sample_data.
        """
        if isinstance(sample_data, dict):
            self.raw_data = pd.Series(sample_data)
        else:
            self.raw_data = sample_data.copy()
        
        # Auto-detect sample_id from GEOROC_ID if not explicitly provided
        if sample_id is None:
            if 'GEOROC_ID' in self.raw_data.index and pd.notna(self.raw_data.get('GEOROC_ID')):
                self.sample_id = str(self.raw_data['GEOROC_ID'])
            else:
                self.sample_id = f"sample_{id(self)}"
        else:
            self.sample_id = sample_id
            
        self.cleaned_data = None
        self.transformed_data = {}
        self.classifications = {}
        self.metadata = {}
        
    def clean(self, loi_threshold: float = 3.5) -> 'GeochemicalSample':
        """
        Clean the sample data by removing zeros, negatives, and samples with high LOI.
        
        Args:
            loi_threshold: Maximum LOI value to retain (default: 3.5)
            
        Returns:
            Self for method chaining
        """
        data = self.raw_data.copy()
        
        # Replace zeros with NaN
        data = data.replace(0, np.nan)
        
        # Replace negative concentrations with NaN
        for element in self.ELEMENTS:
            if element in data.index and pd.notna(data[element]) and data[element] < 0:
                data[element] = np.nan
        
        # Drop if SiO2 is not present
        if 'SIO2' not in data.index or pd.isna(data['SIO2']):
            raise ValueError("Sample must have SiO2 concentration")
        
        # Drop if LOI is too high
        if 'LOI' in data.index and pd.notna(data['LOI']) and data['LOI'] > loi_threshold:
            raise ValueError(f"Sample LOI ({data['LOI']:.2f}) exceeds threshold ({loi_threshold})")
        
        self.cleaned_data = data
        return self
    
    def convert_ppm_to_wt(self) -> 'GeochemicalSample':
        """
        Convert ppm elements to weight percent.
        
        Returns:
            Self for method chaining
        """
        if self.cleaned_data is None:
            raise ValueError("Sample must be cleaned before conversion")
        
        data = self.cleaned_data.copy()
        for element in self.PPM_ELEMENTS:
            if element in data.index and pd.notna(data[element]):
                data[element] = data[element] / 10000
        
        self.cleaned_data = data
        return self
    
    def normalize_totals(self) -> 'GeochemicalSample':
        """
        Normalize all elements to 100% total.
        
        Returns:
            Self for method chaining
        """
        if self.cleaned_data is None:
            raise ValueError("Sample must be cleaned before normalization")
        
        data = self.cleaned_data.copy()
        
        # Calculate total
        total = sum(data[element] for element in self.ELEMENTS if element in data.index and pd.notna(data[element]))
        
        if total == 0:
            raise ValueError("Cannot normalize: total concentration is zero")
        
        # Normalize elements
        for element in self.ELEMENTS:
            if element in data.index and pd.notna(data[element]):
                data[f'{element}_norm'] = data[element] / total
        
        data['calc_TOTAL'] = total
        self.cleaned_data = data
        return self
    
    def calculate_molar_fractions(self) -> 'GeochemicalSample':
        """
        Calculate molar fractions for all elements.
        
        Returns:
            Self for method chaining
        """
        if self.cleaned_data is None:
            raise ValueError("Sample must be cleaned before molar calculation")
        
        data = self.cleaned_data.copy()
        
        # Calculate moles
        mol_total = 0
        mol_data = {}
        
        for element in self.ELEMENTS:
            if element in data.index and pd.notna(data[element]):
                moles = data[element] / self.ELEMENTS_MASS[element]
                mol_data[f'{element}_mol'] = moles
                mol_total += moles
        
        # Calculate molar fractions
        for element in self.ELEMENTS:
            if element in data.index and pd.notna(data[element]):
                data[f'{element}_molfrac'] = mol_data[f'{element}_mol'] / mol_total
        
        self.cleaned_data = data
        return self
    
    def clr_transform(self) -> 'GeochemicalSample':
        """
        Apply center log-ratio transformation.
        
        Returns:
            Self for method chaining
        """
        if self.cleaned_data is None:
            raise ValueError("Sample must be cleaned before CLR transformation")
        
        data = self.cleaned_data.copy()
        
        # Get normalized elements
        norm_elements = [f'{element}_norm' for element in self.ELEMENTS 
                       if f'{element}_norm' in data.index and pd.notna(data[f'{element}_norm'])]
        
        if len(norm_elements) == 0:
            raise ValueError("No normalized elements found for CLR transformation")
        
        # Extract values as numpy array and filter out any non-positive values
        norm_values = np.array([data[col] for col in norm_elements], dtype=float)
        norm_values = norm_values[norm_values > 0]  # Only positive values for log
        
        if len(norm_values) == 0:
            raise ValueError("No positive normalized values found for CLR transformation")
        
        # Calculate geometric mean
        log_values = np.log(norm_values)
        geom_mean = np.exp(log_values.mean())
        
        # Apply CLR transformation
        for element in self.ELEMENTS:
            norm_col = f'{element}_norm'
            if norm_col in data.index and pd.notna(data[norm_col]) and data[norm_col] > 0:
                data[f'{element}_clr'] = np.log(data[norm_col] / geom_mean)
        
        data['geom_mean'] = geom_mean
        self.cleaned_data = data
        return self
    
    def classify_sio2(self) -> str:
        """
        Classify sample based on SiO2 content.
        
        Returns:
            Classification string
        """
        if self.cleaned_data is None:
            raise ValueError("Sample must be cleaned before classification")
        
        sio2_norm = self.cleaned_data.get('SIO2_norm', np.nan)
        if pd.isna(sio2_norm):
            return np.nan
        
        if sio2_norm <= 0.45:
            return 'ultrabasic'
        elif sio2_norm <= 0.52:
            return 'basic'
        elif sio2_norm <= 0.63:
            return 'intermediate'
        else:
            return 'acid'
    
    def calculate_mg_number(self) -> float:
        """
        Calculate Mg number.
        
        Returns:
            Mg number
        """
        if self.cleaned_data is None:
            raise ValueError("Sample must be cleaned before calculation")
        
        mgo = self.cleaned_data.get('MGO_norm', np.nan)
        feo = self.cleaned_data.get('FEO_norm', np.nan)
        fe2o3 = self.cleaned_data.get('FE2O3_norm', np.nan)
        feot = self.cleaned_data.get('FEOT_norm', np.nan)
        
        if pd.isna(mgo):
            return np.nan
        
        if pd.notna(feot):
            iron = feot
        else:
            iron = feo + fe2o3 if pd.notna(feo) and pd.notna(fe2o3) else np.nan
        
        if pd.isna(iron):
            return np.nan
        
        return mgo / (mgo + iron)
    
    def get_element_concentration(self, element: str, unit: str = 'wt') -> float:
        """
        Get element concentration in specified unit.
        
        Args:
            element: Element symbol
            unit: Unit type ('wt', 'norm', 'clr', 'molfrac')
            
        Returns:
            Element concentration
        """
        if self.cleaned_data is None:
            raise ValueError("Sample must be cleaned before getting concentrations")
        
        if unit == 'wt':
            return self.cleaned_data.get(element, np.nan)
        elif unit == 'norm':
            return self.cleaned_data.get(f'{element}_norm', np.nan)
        elif unit == 'clr':
            return self.cleaned_data.get(f'{element}_clr', np.nan)
        elif unit == 'molfrac':
            return self.cleaned_data.get(f'{element}_molfrac', np.nan)
        else:
            raise ValueError(f"Unknown unit: {unit}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert sample to dictionary.
        
        Returns:
            Dictionary representation of the sample
        """
        result = {
            'sample_id': self.sample_id,
            'raw_data': self.raw_data.to_dict() if self.raw_data is not None else {},
            'cleaned_data': self.cleaned_data.to_dict() if self.cleaned_data is not None else {},
            'classifications': self.classifications,
            'metadata': self.metadata
        }
        return result
    
    def __repr__(self) -> str:
        """String representation of the sample."""
        return f"GeochemicalSample(id={self.sample_id}, cleaned={self.cleaned_data is not None})"
