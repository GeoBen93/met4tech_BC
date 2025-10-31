# Geochemical Analysis Package

A comprehensive Python package for cleaning, transforming, and analyzing geochemical data from the GEOROC database. This package provides both functional and object-oriented approaches to geochemical data analysis.

## Features

- **Data Cleaning**: Remove invalid values, normalize REE elements, convert units
- **Data Transformations**: Apply compositional transformations (normalization, CLR, molar fractions)
- **Geochemical Classifications**: Add SiO2, alkalinity, and other geochemical classifications
- **Regression Analysis**: Perform statistical regression analyses with parallel processing
- **Visualization**: Create specialized plots for geochemical data and results
- **Object-Oriented Design**: GeochemicalSample class for individual sample handling

## Installation

1. Clone or download this repository
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Using the GeochemicalSample Class

```python
from src import GeochemicalSample

# Create a sample from raw data
sample_data = {'SIO2': 50.0, 'AL2O3': 15.0, 'FEO': 8.0, ...}
sample = GeochemicalSample(sample_data, sample_id="sample_001")

# Apply transformations
sample.clean().convert_ppm_to_wt().normalize_totals().clr_transform()

# Get element concentrations
sio2_clr = sample.get_element_concentration('SIO2', 'clr')
mg_number = sample.calculate_mg_number()
```

### Using Functional Approach

```python
from src import clean_data, clr_transform, regress_parallel

# Clean data
df_clean = clean_data(df_raw, elements)

# Apply transformations
df_transformed = clr_transform(df_clean, elements)

# Perform regression analysis
results = regress_parallel(df_transformed, 'VOLCANO', independent_vars, dependent_vars)
```

## Package Structure

```
src/
├── __init__.py              # Package initialization and exports
├── sample.py                # GeochemicalSample class
├── data_cleaning.py         # Data cleaning and validation functions
├── data_transformations.py # Compositional transformation functions
├── regression_analysis.py  # Statistical regression analysis functions
└── plotting.py             # Visualization functions

tests/
└── test_geochemical_analysis.py  # Unit tests

examples/
└── workflow_demo.ipynb     # Comprehensive workflow demonstration
```

## Modules

### GeochemicalSample Class (`sample.py`)
- Encapsulates individual geochemical samples
- Methods for cleaning, transformation, and classification
- Object-oriented approach to sample handling

### Data Cleaning (`data_cleaning.py`)
- `clean_data()`: Remove invalid values and samples
- `ci_norm_ree()`: Normalize REE elements to chondrite values
- `convert_ppm2wtpc()`: Convert ppm to weight percent
- `add_geochemical_classifications()`: Add geochemical classification schemes
- `add_volcano_names()`: Fuzzy matching for volcano names

### Data Transformations (`data_transformations.py`)
- `normalise_totals()`: Normalize to 100% totals
- `molar_fraction()`: Calculate molar fractions
- `clr_transform()`: Apply center log-ratio transformation
- `apply_compositional_transformation()`: Apply various transformations
- `batch_transform_samples()`: Process multiple samples

### Regression Analysis (`regression_analysis.py`)
- `regress_parallel()`: Parallel regression analysis across groups
- `linregress_group()`: Linear regression for individual groups
- `extract_result()`: Extract specific regression parameters
- `transpose_to_dict()`: Reorganize results for analysis
- `compare_grouping_methods()`: Compare different grouping approaches

### Plotting (`plotting.py`)
- `plot_regression_boxplots()`: Boxplots for regression slopes
- `plot_survival_function()`: Survival function plots
- `plot_element_concentrations()`: Element concentration visualizations
- `plot_correlation_matrix()`: Correlation matrix heatmaps
- `plot_regression_summary()`: Summary plots for regression results

## Workflow Example

See `examples/workflow_demo.ipynb` for a comprehensive demonstration of the package capabilities.

## Testing

Run the unit tests to verify functionality:

```bash
cd tests
python test_geochemical_analysis.py
```

## Dependencies

- pandas >= 1.3.0
- numpy >= 1.21.0
- scipy >= 1.7.0
- statsmodels >= 0.13.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- joblib >= 1.1.0
- tqdm >= 4.62.0

## Benefits of the Modular Approach

1. **Reusable Functions**: Each function can be imported and used independently
2. **Object-Oriented Design**: GeochemicalSample class provides intuitive interface
3. **Parallel Processing**: Efficient handling of large datasets
4. **Comprehensive Analysis**: Full suite of geochemical analysis tools
5. **Easy Extension**: Simple to add new functionality
6. **Better Organization**: Clear separation of concerns
7. **Improved Testing**: Individual functions can be tested separately
8. **Documentation**: Each function has clear docstrings and type hints

## Contributing

1. Follow the existing code structure and naming conventions
2. Add comprehensive docstrings to new functions
3. Include unit tests for new functionality
4. Update this README with new features

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Ben Clarke - ben.clarke@example.com

## Acknowledgments

- Based on functions developed in GEOROC analysis notebooks
- Inspired by compositional data analysis principles
- Built for the MET4TECH project



