import functime
import polars as pl

def features_extractor(df: pl.DataFrame, time_col: str, time_index_col: str, value_col: str, features_names: list = None, lags: list[int] = [10,20,30]):
    """
    Extract time series features from a Polars DataFrame using functime.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing the time series data
    time_col : str
        Name of the column containing the actual timestamps
    time_index_col : str 
        Name of the column containing the time series index/identifier
    value_col : str
        Name of the column containing the values to extract features from
    features_names : list, optional
        List of feature names to extract. If None, extracts all available features
    lags : list[int], default [10,20,30]
        List of lag values to use for lagged features

    Returns
    -------
    pl.DataFrame
        DataFrame containing the extracted time series features

    Notes
    -----
    The function extracts statistical and time series features using the functime library.
    Features include metrics like absolute energy, autocorrelation, entropy measures etc.
    The data is first sorted by time_index_col and time_col before feature extraction.
    """    
    # Sort by time if time_col is specified
    if time_col:
        df = df.sort(by=[time_index_col, time_col])

    # Get all available feature extractors if features_names not specified
    if features_names is None:
        features_names = [
            'absolute_energy', 
            'absolute_maximum', 
            'absolute_sum_of_changes', 
            'autocorrelation', 
            'benford_correlation', 
            'binned_entropy', 
            'c3',
            'cid_ce', 
            'count_above',
            'count_above_mean', 
            'count_below', 
            'count_below_mean', 
            'first_location_of_maximum',
            'first_location_of_minimum', 
            'harmonic_mean', 
            'has_duplicate', 
            'has_duplicate_max',
            'has_duplicate_min', 
            'index_mass_quantile', 
            'large_standard_deviation', 
            'last_location_of_maximum', 
            'last_location_of_minimum', 
            'lempel_ziv_complexity',
            'max_abs_change',
            'mean_abs_change',
            'mean_change', 
            'mean_n_absolute_max', 
            'mean_second_derivative_central',
            'number_crossings', 
            'number_peaks', 
            'percent_reoccurring_points',
            'percent_reoccurring_values', 
            'permutation_entropy',
            'range_change', 
            'range_count', 
            'range_over_mean',
            'ratio_beyond_r_sigma', 
            'ratio_n_unique_to_length', 
            'root_mean_square', 
            'sum_reoccurring_points', 
            'sum_reoccurring_values', 
            'symmetry_looking', 
            'time_reversal_asymmetry_statistic', 
            'var_gt_std', 
            'variation_coefficient'
        ]

    # Define base parameters for each feature
    features_params = {}
    
    # Features with multiple parameter sets
    for lag in lags:
        features_params[f'autocorrelation_npar{lag}'] = {'feature': 'autocorrelation', 'params': {'n_lags': lag}}
    
    features_params['binned_entropy'] = {'feature': 'binned_entropy', 'params': {'bin_count': 10}}
    features_params['c3'] = {'feature': 'c3', 'params': {'n_lags': 3}}
    features_params['count_above'] = {'feature': 'count_above', 'params': {'threshold': 0.0}}
    features_params['count_below'] = {'feature': 'count_below', 'params': {'threshold': 0.0}}
    features_params['index_mass_quantile'] = {'feature': 'index_mass_quantile', 'params': {'q': 0.5}}
    features_params['large_standard_deviation'] = {'feature': 'large_standard_deviation', 'params': {'ratio': 0.25}}
    features_params['lempel_ziv_complexity'] = {'feature': 'lempel_ziv_complexity', 'params': {'threshold': 0, 'as_ratio': True}}
    features_params['longest_streak_above'] = {'feature': 'longest_streak_above', 'params': {'threshold': 0}}
    features_params['longest_streak_below'] = {'feature': 'longest_streak_below', 'params': {'threshold': 0}}
    features_params['mean_n_absolute_max'] = {'feature': 'mean_n_absolute_max', 'params': {'n_maxima': 5}}
    features_params['number_crossings'] = {'feature': 'number_crossings', 'params': {'crossing_value': 0.0}}
    features_params['number_peaks'] = {'feature': 'number_peaks', 'params': {'support': 3}}
    features_params['permutation_entropy'] = {'feature': 'permutation_entropy', 'params': {'tau': 1, 'n_dims': 3}}
    features_params['range_change'] = {'feature': 'range_change', 'params': {'percentage': True}}
    features_params['range_count'] = {'feature': 'range_count', 'params': {'lower': -1, 'upper': 1, 'closed': 'left'}}
    features_params['ratio_beyond_r_sigma'] = {'feature': 'ratio_beyond_r_sigma', 'params': {'ratio': 0.25}}
    features_params['streak_length_stats'] = {'feature': 'streak_length_stats', 'params': {'above': True, 'threshold': 0}}
    features_params['symmetry_looking'] = {'feature': 'symmetry_looking', 'params': {'ratio': 0.25}}
    
    for lag in lags:
        features_params[f'time_reversal_asymmetry_statistic_npar{lag}'] = {'feature': 'time_reversal_asymmetry_statistic', 'params': {'n_lags': lag}}
    
    features_params['var_gt_std'] = {'feature': 'var_gt_std', 'params': {'ddof': 1}}
    features_params['approximate_entropy'] = {'feature': 'approximate_entropy', 'params': {'run_length': 2, 'filtering_level': 0.2, 'scale_by_std': True}}
    
    for lag in lags:
        features_params[f'augmented_dickey_fuller_npar{lag}'] = {'feature': 'augmented_dickey_fuller', 'params': {'n_lags': lag}}
        features_params[f'autoregressive_coefficients_npar{lag}'] = {'feature': 'autoregressive_coefficients', 'params': {'n_lags': lag}}
    
    features_params['cwt_coefficients'] = {'feature': 'cwt_coefficients', 'params': {'widths': (2, 5, 10, 20), 'n_coefficients': 14}}
    features_params['friedrich_coefficients'] = {'feature': 'friedrich_coefficients', 'params': {'polynomial_order': 3, 'n_quantiles': 30}}
    features_params['fourier_entropy'] = {'feature': 'fourier_entropy', 'params': {'n_bins': 10}}
    features_params['number_cwt_peaks'] = {'feature': 'number_cwt_peaks', 'params': {'max_width': 5}}
    features_params['sample_entropy'] = {'feature': 'sample_entropy', 'params': {'ratio': 0.2, 'm': 2}}
    features_list = [i.split('_npar')[0] for i in features_params.keys()]
    
    if features_names is not None:
        for feature in features_names:
            if feature not in features_list:  # Skip already defined features with parameters
                features_params[feature] = {'feature': feature, 'params': {}}
    
    print(features_params)
    agg_exprs = {
        feature_name: getattr(pl.col(value_col).ts, config['feature'])(**config['params'])
        for feature_name, config in features_params.items()
        if (features_names is None) or (feature_name in features_names) or (config['feature'] in features_names)
    }

    return df.group_by(time_index_col).agg(**agg_exprs)
