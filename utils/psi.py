import numpy as np

def population_stability_index(base, current, bins=10):
    """
    Calculate the Population Stability Index (PSI) between two distributions.
    base: array-like, reference distribution
    current: array-like, comparison distribution
    bins: int or sequence, number of bins or bin edges
    Returns: float, PSI value
    """
    base = np.asarray(base)
    current = np.asarray(current)
    # Remove nan values
    base = base[~np.isnan(base)]
    current = current[~np.isnan(current)]
    # Bin edges
    if isinstance(bins, int):
        bin_edges = np.histogram_bin_edges(np.concatenate([base, current]), bins=bins)
    else:
        bin_edges = np.asarray(bins)
    base_hist, _ = np.histogram(base, bins=bin_edges)
    curr_hist, _ = np.histogram(current, bins=bin_edges)
    # Convert to proportions
    base_prop = base_hist / (base_hist.sum() + 1e-8)
    curr_prop = curr_hist / (curr_hist.sum() + 1e-8)
    # Avoid log(0) and division by zero
    base_prop = np.where(base_prop == 0, 1e-8, base_prop)
    curr_prop = np.where(curr_prop == 0, 1e-8, curr_prop)
    psi = np.sum((curr_prop - base_prop) * np.log(curr_prop / base_prop))
    return psi
