"""
Unit tests for the geochemical analysis package.

This module contains unit tests for all the main functions and classes
in the geochemical analysis package.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.sample import GeochemicalSample
from src.data_cleaning import clean_data, ci_norm_ree, convert_ppm2wtpc
from src.data_transformations import normalise_totals, molar_fraction, clr_transform
from src.regression_analysis import linregress_group, linear_regressions


class TestGeochemicalSample(unittest.TestCase):
    """Test cases for the GeochemicalSample class."""
    
    def setUp(self):
        """Set up test data."""
        self.sample_data = {
            'SIO2': 50.0,
            'AL2O3': 15.0,
            'FEO': 8.0,
            'MGO': 5.0,
            'CAO': 10.0,
            'NA2O': 3.0,
            'K2O': 2.0,
            'LOI': 2.0,
            'LA': 10.0,
            'CE': 20.0
        }
        self.sample = GeochemicalSample(self.sample_data, sample_id="test_sample")
    
    def test_sample_creation(self):
        """Test sample creation."""
        self.assertEqual(self.sample.sample_id, "test_sample")
        self.assertIsNotNone(self.sample.raw_data)
        self.assertEqual(self.sample.raw_data['SIO2'], 50.0)
    
    def test_clean_method(self):
        """Test data cleaning."""
        self.sample.clean()
        self.assertIsNotNone(self.sample.cleaned_data)
        self.assertEqual(self.sample.cleaned_data['SIO2'], 50.0)
    
    def test_convert_ppm_to_wt(self):
        """Test ppm to weight percent conversion."""
        self.sample.clean()
        self.sample.convert_ppm_to_wt()
        # REE elements should be converted
        self.assertAlmostEqual(self.sample.cleaned_data['LA'], 0.001, places=3)
    
    def test_normalize_totals(self):
        """Test normalization to 100%."""
        self.sample.clean()
        self.sample.normalize_totals()
        # Check that normalized SiO2 exists
        self.assertIn('SIO2_norm', self.sample.cleaned_data.index)
        # Check that total is calculated
        self.assertIn('calc_TOTAL', self.sample.cleaned_data.index)
    
    def test_clr_transform(self):
        """Test center log-ratio transformation."""
        self.sample.clean()
        self.sample.normalize_totals()
        self.sample.clr_transform()
        # Check that CLR transformed SiO2 exists
        self.assertIn('SIO2_clr', self.sample.cleaned_data.index)
    
    def test_classify_sio2(self):
        """Test SiO2 classification."""
        self.sample.clean()
        self.sample.normalize_totals()
        classification = self.sample.classify_sio2()
        self.assertIn(classification, ['ultrabasic', 'basic', 'intermediate', 'acid'])
    
    def test_get_element_concentration(self):
        """Test getting element concentrations."""
        self.sample.clean()
        self.sample.normalize_totals()
        
        sio2_wt = self.sample.get_element_concentration('SIO2', 'wt')
        sio2_norm = self.sample.get_element_concentration('SIO2', 'norm')
        
        self.assertEqual(sio2_wt, 50.0)
        self.assertIsInstance(sio2_norm, float)


class TestDataCleaning(unittest.TestCase):
    """Test cases for data cleaning functions."""
    
    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'SIO2': [50.0, 0.0, 45.0, np.nan],
            'AL2O3': [15.0, 10.0, -5.0, 12.0],
            'FEO': [8.0, 5.0, 6.0, 7.0],
            'LOI': [2.0, 5.0, 1.0, 3.0]
        })
        self.elements = ['SIO2', 'AL2O3', 'FEO']
    
    def test_clean_data(self):
        """Test data cleaning function."""
        cleaned_df = clean_data(self.df, self.elements)
        
        # Should remove row with SiO2 = 0
        self.assertEqual(len(cleaned_df), 3)
        
        # Should remove row with negative AL2O3
        self.assertEqual(len(cleaned_df), 2)
        
        # Should remove row with high LOI
        self.assertEqual(len(cleaned_df), 1)
    
    def test_convert_ppm2wtpc(self):
        """Test ppm to weight percent conversion."""
        ppm_elements = ['LA', 'CE']
        df_with_ppm = pd.DataFrame({
            'LA': [1000, 2000],
            'CE': [500, 1500]
        })
        
        converted_df = convert_ppm2wtpc(df_with_ppm, ppm_elements)
        
        self.assertAlmostEqual(converted_df['LA'].iloc[0], 0.1, places=1)
        self.assertAlmostEqual(converted_df['CE'].iloc[0], 0.05, places=2)


class TestDataTransformations(unittest.TestCase):
    """Test cases for data transformation functions."""
    
    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'SIO2': [50.0, 45.0, 60.0],
            'AL2O3': [15.0, 20.0, 12.0],
            'FEO': [8.0, 10.0, 5.0]
        })
        self.elements = ['SIO2', 'AL2O3', 'FEO']
        self.elements_mass = {'SIO2': 60.08, 'AL2O3': 101.96, 'FEO': 71.85}
    
    def test_normalise_totals(self):
        """Test normalization to 100%."""
        normalized_df = normalise_totals(self.df, self.elements)
        
        # Check that normalized columns exist
        for element in self.elements:
            self.assertIn(f'{element}_norm', normalized_df.columns)
        
        # Check that totals are calculated
        self.assertIn('calc_TOTAL', normalized_df.columns)
        
        # Check that normalized values sum to 1
        total_norm = sum(normalized_df[f'{element}_norm'].iloc[0] for element in self.elements)
        self.assertAlmostEqual(total_norm, 1.0, places=5)
    
    def test_molar_fraction(self):
        """Test molar fraction calculation."""
        normalized_df = normalise_totals(self.df, self.elements)
        molar_df = molar_fraction(normalized_df, self.elements, self.elements_mass)
        
        # Check that molar fraction columns exist
        for element in self.elements:
            self.assertIn(f'{element}_molfrac', molar_df.columns)
        
        # Check that molar fractions sum to 1
        total_molfrac = sum(molar_df[f'{element}_molfrac'].iloc[0] for element in self.elements)
        self.assertAlmostEqual(total_molfrac, 1.0, places=5)
    
    def test_clr_transform(self):
        """Test center log-ratio transformation."""
        normalized_df = normalise_totals(self.df, self.elements)
        clr_df = clr_transform(normalized_df, self.elements)
        
        # Check that CLR columns exist
        for element in self.elements:
            self.assertIn(f'{element}_clr', clr_df.columns)
        
        # Check that CLR values sum to 0 (approximately)
        total_clr = sum(clr_df[f'{element}_clr'].iloc[0] for element in self.elements)
        self.assertAlmostEqual(total_clr, 0.0, places=5)


class TestRegressionAnalysis(unittest.TestCase):
    """Test cases for regression analysis functions."""
    
    def setUp(self):
        """Set up test data."""
        self.df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'x': [1, 2, 3, 1, 2, 3],
            'y': [2, 4, 6, 1, 3, 5]
        })
    
    def test_linregress_group(self):
        """Test linear regression on a group."""
        group = self.df[self.df['group'] == 'A']
        result = linregress_group(group, 'x', 'y')
        
        # Check that result contains expected keys
        expected_keys = ['x_y_slope', 'x_y_intercept', 'x_y_r2', 'x_y_pval', 'x_y_stderr', 'x_y_n']
        for key in expected_keys:
            self.assertIn(key, result.index)
        
        # Check that slope is approximately 2 (perfect correlation)
        self.assertAlmostEqual(result['x_y_slope'], 2.0, places=1)
        self.assertAlmostEqual(result['x_y_r2'], 1.0, places=1)
    
    def test_linear_regressions(self):
        """Test linear regressions grouped by feature."""
        result_df = linear_regressions(self.df, 'group', 'x', 'y')
        
        # Should have results for both groups
        self.assertEqual(len(result_df), 2)
        self.assertIn('group', result_df.columns)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestGeochemicalSample))
    test_suite.addTest(unittest.makeSuite(TestDataCleaning))
    test_suite.addTest(unittest.makeSuite(TestDataTransformations))
    test_suite.addTest(unittest.makeSuite(TestRegressionAnalysis))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")
