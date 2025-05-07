# This is a module that automatically generates plots

# %%

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.odr import ODR, Model, Data
from scipy.stats import norm
import re
from sklearn.linear_model import LinearRegression
from copy import deepcopy
from auto_exp import read_existing_exp_file
from decimal import Decimal
import math
from scipy import stats
import statsmodels.api as sm

DEACTIVATE_PLOTS = False

# %%

def drop_old_experiments(df):
    # Keep only the last duplicate of each column, where two columns are
    # duplicate if everything is the same, execpt for 'filename', 'avg_time', 'std',
    # 'min', and 'max'

    keep_cols = ["timestamp", "avg_time", "std", "min", "max", "index", "exp_name"]
    drop_cols = [col for col in df.columns if col not in keep_cols]
    df = df.sort_values(by=['timestamp'], ascending=True)
    return df.drop_duplicates(subset=drop_cols, keep='last')

def load_all_experiments(drop_old = True):
    # Collect all the files with extension .csv inside the directory '../data/experiments/'

    filename_pairs = set()
    for filename in os.listdir("../data/experiments"):
        if filename.endswith(".csv"):
            filename_pairs.add(filename)

    common_filenames = filename_pairs

    # Load all the files as pandas dataframes and merge them in a single file

    dfs = []
    for filename in common_filenames:
        df = read_existing_exp_file(f"../data/experiments/{filename}")
        dfs.append(df)

    df_merged = pd.concat(dfs, ignore_index=True)


    # Remove all rows haveing value "m" euqal to 0

    df_merged = df_merged[df_merged['m'] != 0]

    df_merged = df_merged.sort_values(by=['timestamp'], ascending=True)

    if drop_old:
        df_merged = drop_old_experiments(df_merged)


    # Drop all rows having -1 in the 'avg_time' column

    df_merged = df_merged[df_merged['std'] != -1]

    # Drop all rows having 'rd1' or 'rd2' in the 'filename' column that also
    # have a value of 'timestamp' lower than '2024-07-04'

    df_merged = df_merged[~df_merged['filename'].str.contains('rd1|rd2') | (df_merged['timestamp'] >= '2024-07-04')]

    df_merged['avg_time'] = df_merged['avg_time'].astype(float) / 1000
    df_merged['std'] = df_merged['std'].astype(float) / 1000
    df_merged['min'] = df_merged['min'].astype(float) / 1000
    df_merged['max'] = df_merged['max'].astype(float) / 1000

    return df_merged

# I want to group all the rows with the same filename, same value n and same value m. Each
# group has the filename containing "bfct", "dfs", "gor", "mbfct", "rd1", "rd2", "djtest".
# Distinguish between groups that have "aug" in the filename and those that don't, and
# distinguish between groups that have a different value of n or different value of m.

def group_by_instance(df):
  """
  Groups a pandas dataframe based on filename substrings, presence of "aug", and unique values in columns n and m.

  Args:
      df (pandas.DataFrame): The dataframe to be grouped.
      filename_substrings (list): List of substrings to consider from filenames (excluding overlaps).

  Returns:
      pandas.GroupBy: Grouped dataframe object.
  """

  filename_substrings = ["bfct", "dfs", "gor", "rd1", "rd2", "djtest"]
#   cols_list = list(df.columns)
#   bcf_params = cols_list[cols_list.index('use_lazy') : cols_list.index('avg_time')]

  # Filter out filenames containing substrings within other substrings (e.g., "mbfct" shouldn't be grouped with "bfct").
#   filtered_df = df[df.filename.str.cat(filename_substrings, sep='|', na_rep='').isin(filename_substrings)]

  # Create a custom lambda function to define the grouping logic
  def group_by_func(row):
        row = df.loc[row]

        substrings = sorted([substring for substring in filename_substrings if substring in row.filename], key=len, reverse=True)
        if len(substrings) == 0:
            substring = 'NA'
        else:
            substring = substrings[0]

        return (
        # Group by filename substring (excluding overlaps)
        substring,
        # Group by presence of "aug" (boolean or converted to boolean)
        'shift' if re.search(r'shift_[a-z]+_', row.exp_name) else 'aug' if 'aug' in row.filename else 'original',
        # Group by unique values in columns n and m
        # row['n'], row['m'],
        # row.source,
        # '' if row.alg_name != 'BCF' else '-'.join([str(row[param]) for param in bcf_params]),
        )

  # Group the dataframe based on the custom function
  return df.groupby(group_by_func)
#   return filtered_df.groupby(group_by_func)

def group_by_alg(df):
    cols_list = list(df.columns)
    bcf_params = cols_list[cols_list.index('use_lazy') : cols_list.index('avg_time')]

    def group_by_func(row):
        row = df.loc[row]

        return (
            row.alg_name,
            '' if row.alg_name != 'BCF' else '-'.join([str(row[param]) for param in bcf_params]),
            )

    return df.groupby(group_by_func)


def orth_weigh(x, y, x_weights=None, y_weights=None):
    
    if x_weights is None:
        x_weights = np.ones(len(x))
    
    if y_weights is None:
        y_weights = np.ones(len(y))

    # Define the orthogonal regression model
    def orthogonal_regression(B, x):
        return B[0] * x + B[1]

    # Create the ODR data object
    data = Data(x, y, wd=x_weights, we=y_weights)

    # Create the ODR model object
    model = Model(orthogonal_regression)

    # Create the ODR object
    odr = ODR(data, model, beta0=[1.0, 0.0])

    # Run the ODR regression
    results = odr.run()

    # Get the fitted parameters
    parameters = results.beta

    # Get the standard errors of the parameters
    parameter_errors = results.sd_beta

    return parameters[0], parameters[1], parameter_errors[0], parameter_errors[1]

def linear_regression(x, y, y_err=None):
    # Ensure inputs are NumPy arrays
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if y_err is not None:
        y_err = np.array(y_err, dtype=float)

    if y_err is not None:
        # Weighted linear regression using np.polyfit
        w = 1 / (y_err ** 2)
        x1, const = np.polyfit(x, y, 1, w=w)
        
        # Calculate errors
        n = len(x)
        wx = w * x
        wy = w * y
        mean_x = np.sum(wx) / np.sum(w)
        mean_y = np.sum(wy) / np.sum(w)
        s_xx = np.sum(w * (x - mean_x)**2)
        s_yy = np.sum(w * (y - mean_y)**2)
        
        y_pred = const + x1 * x
        residuals = y - y_pred
        chi_square = np.sum((residuals / y_err)**2)
        reduced_chi_square = chi_square / (n - 2)
        
        x1_err = np.sqrt(reduced_chi_square / s_xx)
        const_err = x1_err * np.sqrt(np.sum(x**2) / n)
    else:
        # Non-weighted linear regression
        const, x1, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Calculate errors
        n = len(x)
        mean_x = np.mean(x)
        s_xx = np.sum((x - mean_x)**2)
        
        y_pred = const + x1 * x
        residuals = y - y_pred
        residual_variance = np.sum(residuals**2) / (n - 2)
        
        x1_err = np.sqrt(residual_variance / s_xx)
        const_err = x1_err * np.sqrt(np.sum(x**2) / n)

    return x1, const, x1_err, const_err

# def linear_regression(x, y, y_err=None):
#     # Sample data
#     x = np.array(x)
#     y = np.array(y)  # Adjusted y values with more variability
    
#     if y_err is None:
#         y_err = np.ones(len(y))
    
#     weights = 1 / y_err ** 2

#     # Calculate weights based on the inverse of the variance of the errors
#     errors = y - np.mean(y)
#     error_variance = np.var(errors)
#     weights = 1 / error_variance

#     # Fit weighted least squares regression model
#     X = sm.add_constant(X)
#     model = sm.WLS(y, X, weights=weights)
#     results = model.fit()

def linear_regression2(x, y):
    x = np.array(x)
    y = np.array(y)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope, intercept, 0.0001, 0.0001

def weighted_linear_regression(x, y, y_err=None):
    # Convert input lists to numpy arrays
    x = np.array(x)
    y = np.array(y)
    if y_err is not None:
        y_err = np.array(y_err)
    
    # Define weights
    if y_err is None:
        weights = np.ones_like(y)
    else:
        weights = 1 / (y_err ** 2)
    
    # Calculate the weighted averages
    w = weights
    W = np.sum(w)
    wx = np.sum(w * x)
    wy = np.sum(w * y)
    wx2 = np.sum(w * x * x)
    wxy = np.sum(w * x * y)

    # Calculate slope (x1) and intercept (const)
    denominator = W * wx2 - wx ** 2
    x1 = (W * wxy - wx * wy) / denominator
    const = (wx2 * wy - wx * wxy) / denominator

    # Calculate errors in slope and intercept
    x1_err = np.sqrt(W / denominator)
    const_err = np.sqrt(wx2 / denominator)
    
    return x1, const, x1_err, const_err

# def stats_linreg(x, y, y_err=None):


def log_regression(x, y, x_error=None, y_error=None, model='linear', loglog=False):
    x_copy = np.array(deepcopy(x))
    y_copy = np.array(deepcopy(y))
    if x_error is not None:
        x_error_copy = np.array(deepcopy(x_error))
    else:
        x_error_copy = None
    if y_error is not None:
        y_error_copy = np.array(deepcopy(y_error))
    else:
        y_error_copy = None
    if y_error is not None:
        y_weights = 1 / y_error_copy ** 2
    else:
        y_weights = None

    # x_copy = np.log10(x_copy).reshape(-1, 1)
    x_copy = np.log10(x_copy)
    x_loglog = np.log10(x_copy)
    y_copy = np.log10(y_copy)

    # If we assume that the y errors are very small compared tu the y values, we can
    # treat them as the same without doing the log

    # # Create and fit the linear regression model
    # model = LinearRegression()
    # model.fit(x_copy, y_copy)

    # # Get the intercept and slope
    # intercept = model.intercept_
    # slope = model.coef_[0]

    if model == 'orthogonal':
        slope, intercept, slope_stderr, intercept_stderr = orth_weigh(x_copy, y_copy, x_weights=x_error_copy, y_weights=y_weights)
    elif model == 'linear':
        # slope, intercept, slope_stderr, intercept_stderr = linear_regression2(x_copy, y_copy)
        # slope, intercept, slope_stderr, intercept_stderr = weighted_linear_regression(x_copy, y_copy, y_err=y_error_copy)
        x_copy = sm.add_constant(x_copy)
        model = sm.OLS(y_copy, x_copy).fit()

        alpha = 0.05
        print(x_copy, y_copy)
        print(model.summary())
        print(model.conf_int(alpha=alpha))

        intercept = model.params[0]
        slope = model.params[1]

        intercept_inter = model.conf_int(alpha=alpha)[0, :]
        slope_inter = model.conf_int(alpha=alpha)[1, :]

        intercept_stderr = (intercept_inter[1] - intercept_inter[0]) / 2
        slope_stderr = (slope_inter[1] - slope_inter[0]) / 2

    else:
        raise ValueError(f"Invalid model: {model}")


    # intercept = 10 ** intercept

    return intercept, slope, intercept_stderr, slope_stderr

def format_numbers(a, b):
    # Find the number of decimal places in b
    decimal_places = len(str(b).split('.')[1])
    
    # Format a and b with the same number of decimal places
    a = format(a, '.{}f'.format(decimal_places))
    b = format(b, '.{}f'.format(decimal_places))
    
    # Return the formatted strings
    return a, b

def round_value_and_error(val, err, force_int=False):
  """Rounds an error value and a corresponding value to appropriate 
  significant digits and truncates them to the last digit of the error.

  Args:
    val: The value to be rounded.
    err: The error value to be rounded.

  Returns:
    A tuple containing the rounded and truncated value and error as strings.
  """

  # Handle cases where err is zero to avoid division by zero errors
  if err == 0:
    return str(round(val, 0)), '0'

  # Determine the position of the most significant digit in err
  msd_pos = -int(math.floor(math.log10(abs(err))))

  digit_pos = 0
  while True:
    # Adjust rounding position if the most significant digit is 1 or 2
    this_digit = str(abs(err))[digit_pos]
    if False and this_digit in ['1', '2']:
        msd_pos += 1
        break
    elif this_digit in ['0', '.']:
        digit_pos += 1
    else:
        break


  # Round err to the determined significant digit
  rounded_err = round(err, msd_pos)

  # Round val to the same decimal place as the rounded err
  rounded_val = round(val, msd_pos)

  # Determine the number of decimal places in rounded_err for truncation
  decimal_places = len(str(rounded_err).split('.')[1]) if '.' in str(rounded_err) else 0

  # Truncate both rounded_val and rounded_err to the determined decimal places
  truncated_val = round(rounded_val, decimal_places)
  truncated_err = round(rounded_err, decimal_places)

#   if truncated_err > 2.9:
  if truncated_err > 0.9 or (force_int and truncated_err > 0.4):
    return str(int(round(truncated_val))), str(int(round(truncated_err)))

  return format_numbers(truncated_val, truncated_err)

def produce_plot(algs_data, name, filename=None, deactivate_plots=None):

    if deactivate_plots is None:
        deactivate_plots = DEACTIVATE_PLOTS

    if not deactivate_plots:
        # Define the colormap
        cmap = plt.get_cmap('tab20')

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Set the background color to white
        ax.set_facecolor('white')

    df = pd.DataFrame(columns=['intercept', 'err_intercept_+', 'err_intercept_-', 'slope', 'err_slope'])

    # Iterate over each set of points and interpolations
    for i, (alg_name, alg_elems) in enumerate(algs_data.items()):
        if not deactivate_plots:
            # Get the color for the current group from the colormap
            color = cmap(i % cmap.N)
            color = cmap((i  * 2) % cmap.N)

        x, y, y_err = [], [], []
        for elem in alg_elems:
            x.append(elem['edges'])
            y.append(elem['avgtime'])
            y_err.append(elem['std'])
        
        # print(x, y)

        if any(elem == 0 for elem in y_err):
            log10_intercept, slope, log10_intercept_err, slope_err = log_regression(x, y)
        else:
            log10_intercept, slope, log10_intercept_err, slope_err = log_regression(x, y, y_error=y_err)

        intercept = 10 ** log10_intercept

        max_intercept = 10 ** (log10_intercept + log10_intercept_err)
        min_intercept = 10 ** (log10_intercept - log10_intercept_err)
        intercept_err = (max_intercept - min_intercept) / 2  # not too correct mathematically

        slope_round, slope_err_round = round_value_and_error(slope, slope_err)

        df = df.append({
            'alg_name': alg_name,
            'intercept': intercept,
            'err_intercept_+': max_intercept - intercept,
            'err_intercept_-': intercept - min_intercept,
            'slope': slope_round,
            'err_slope': slope_err_round,
        }, ignore_index=True)

        if not deactivate_plots:
            # Plot the points
            # ax.scatter(x, y, label='_nolegend_', color=color)
            ax.errorbar(x, y, yerr=y_err, label='_nolegend_', fmt='o', color=color, linestyle='', markersize=2)
        
        x_range = np.linspace(min(x), max(x), 100)
        y_range = intercept * x_range ** slope
        
        # Plot the interpolation line
        # regression = f'{alg_name} (({intercept:.2g} ± {intercept_err:.2g}) * x^({slope_round} ± {slope_err_round}))'
        regression = f'{alg_name} ({intercept:.2e} $m^{{ {slope_round} \pm {slope_err_round} }}$)'
        if not deactivate_plots:
            ax.plot(x_range, y_range, label=regression, color=color)
            

    if not deactivate_plots:
        # Set the x-axis and y-axis to log scale
        ax.set_xscale('log')
        ax.set_yscale('log')

        # Add labels and legend
        ax.set_xlabel('#edges')
        # ax.set_ylabel('milliseconds')
        ax.set_ylabel('seconds')
        # legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.5))
        legend = ax.legend()
        ax.add_artist(legend)
        ax.set_title(name)

        plt.rcParams.update({
            "figure.facecolor":  (1.0, 1.0, 1.0, 1.0),  # red   with alpha = 30%
            "axes.facecolor":    (1.0, 1.0, 1.0, 1.0),  # green with alpha = 50%
            "savefig.facecolor": (1.0, 1.0, 1.0, 1.0),  # blue  with alpha = 20%
        })

        # Save the plot as a PDF file
        plt.savefig(f'../data/plots/{name if filename is None else filename}.pdf', transparent=False, bbox_extra_artists=(legend,), bbox_inches='tight')
        # plt.show()

    return df

def get_plottable_points(df):
    ret = []
    for _, row in df.iterrows():
        iter_info = {}
        iter_info['avgtime'] = float(row['avg_time'])
        iter_info['std'] = float(row['std']) / np.sqrt(float(row['reps']))
        iter_info['min'] = float(row['min'])
        iter_info['max'] = float(row['max'])
        iter_info['alg_type'] = row['alg_goal']
        iter_info['alg_name'] = row['alg_name']
        iter_info['graph_filename'] = row['filename']
        iter_info['source'] = int(row['source'])
        iter_info['iterations'] = int(row['reps'])
        iter_info['nodes'] = int(row['n'])
        iter_info['edges'] = int(row['m'])
        ret.append(iter_info)

    return ret

def get_plottable_points_grouped(df, m_tolerance=0.0):
    plottable_points = []
    # Keep track of the indices that we have considered
    considered_indices = set()

    for index, row in df.iterrows():
        # Skip indices that we have already considered
        if index in considered_indices:
            continue

        # Find other rows with 'm' value close to the current row
        m = row['m']
        df_m = df[np.abs(df['m'] - m) / m <= m_tolerance]
        
        # Mark the indices in this grouping as considered
        considered_indices.update(df_m.index)

        if len(df_m) == 0:
            continue

        avg_time_mean = df_m['avg_time'].mean()
        avg_time_std = df_m['avg_time'].std()
        avg_time_min = df_m['avg_time'].min()
        avg_time_max = df_m['avg_time'].max()
        avg_time_count = df_m['avg_time'].count()

        if avg_time_count == 1:
            # Assuming get_plottable_points is defined elsewhere
            plottable_points.append(get_plottable_points(df_m)[0])
        else:
            # Take the following info from the first row of df_m
            first_row = df_m.iloc[0]
            iter_info = {
                'avgtime': avg_time_mean,
                'std': avg_time_std / np.sqrt(avg_time_count),
                'min': avg_time_min,
                'max': avg_time_max,
                'alg_type': first_row['alg_goal'],
                'alg_name': first_row['alg_name'],
                'graph_filename': first_row['filename'],
                'source': int(first_row['source']),
                'iterations': int(first_row['reps']),
                'nodes': int(first_row['n']),
                'edges': int(first_row['m'])
            }
            plottable_points.append(iter_info)
    
    return plottable_points

def transform_dataframe(df):
    # Drop columns that contain 'err'
    df = df.loc[:, ~df.columns.str.contains('err')]
    
    # Initialize an empty dictionary to hold the new columns and values
    new_data = {}
    
    # Iterate over each row in the dataframe
    for _, row in df.iterrows():
        # Get the algorithm name
        alg_name = row['alg_name']
        
        # Iterate over each column in the row
        for col in df.columns:
            if col != 'alg_name':
                # Create the new column name
                new_col_name = f"{alg_name}_{col}"
                
                # Add the value to the new data dictionary
                new_data[new_col_name] = row[col]
    
    # Convert the dictionary to a single-row dataframe
    new_df = pd.DataFrame(new_data, index=[0])
    
    return new_df

def main(deactivate_plots=False):
    global DEACTIVATE_PLOTS

    DEACTIVATE_PLOTS = deactivate_plots

    df = load_all_experiments()
    grouped_instances_df = group_by_instance(df.copy())

    if False:

        # Access groups and their data
        for group_instance, group_data in grouped_instances_df:
            print(f"Group Instance: {group_instance}")
            algs_data = {}
            # Algs data will be of the form: {'alg1': [{'avgtime' : time, 'std' : std, 'min': min, 'max': max}, {'avgtime' : time, 'std' : std, 'min': min, 'max': max}], 'alg2': [{'avgtime' : time, 'std' : std, 'min': min, 'max': max}, {'avgtime' : time, 'std' : std, 'min': min, 'max': max}]}
            grouped_algs = group_by_alg(group_data.copy())
            for group_alg, group_algs_data in grouped_algs:
                print(f"Group Algorithm: {group_alg}")
                alg_name = group_alg[0] if group_alg[1] == '' else group_alg[0] + '-' + group_alg[1]
                algs_data[alg_name] = get_plottable_points(group_algs_data)

            produce_plot(algs_data, '_'.join(str(elem) for elem in group_instance))
        
        # Get the rows where:
        # - filename matches the regex 'shift3_'
        # - 'exp_name' starts with 'shift_bcf'

        df3 = df[df['filename'].str.contains('shift3_')]
        df3 = df3[df3['exp_name'].str.startswith('shift_bcf')]


        df4 = df[df['filename'].str.contains('shift4_')]
        df4 = df4[df4['exp_name'].str.startswith('shift_bcf')]

        # Get the rows where:
        # - filename matches the regex 'shift_'
        # - 'exp_name' starts with 'shift_bcf'
        # - 'cutedgesseed' = 111

        df0 = df[df['filename'].str.contains('shift_')]
        df0 = df0[df0['exp_name'].str.startswith('shift_bcf')]
        df0 = df0[df0['cutedgesseed'] == 111]

        algs_data = {'BCF-Shift-New-3': get_plottable_points(df3),
                    'BCF-Shift-New-4': get_plottable_points(df4),
                    'BCF-Shift-Old': get_plottable_points(df0)}

        produce_plot(algs_data, 'new_shift_vs_old')

    # log vs log2

    x_vals = [5 * 10**5, 1 * 10 ** 6, 2 * 10 ** 6, 5 * 10 ** 6, 1 * 10 ** 7, 2 * 10 ** 7]
    # x_vals = [10 ** x for x in range(1, 7)]
    log1 = [x * np.log2(x) for x in x_vals]
    log2 = [x * np.log2(x) ** 2 for x in x_vals]
    log3 = [x * np.log2(x) ** 3 for x in x_vals]

    # Create a pandas DataFrame with the columns 'avg_time', 'std', 'min', 'max', 'alg_type', 'alg_name', 'graph_filename', 'source', 'iterations', 'nodes', 'edges'
    # Each row is an element from either log1 or log2
    # Set 'alg_name' to 'log1' or 'log2', respectively, for each row
    # Set 'avg_time' to the corresponding value from either log1 or log2
    # Set 'std' to 0
    # Set 'm' to the corresponding value of x_vals
    # Set all the other columns to the same value 'None'

    def get_df_log(x_vals, log_vals):
        df_log = pd.DataFrame()
        df_log['alg_name'] = ['log'] * len(log_vals)
        df_log['avg_time'] = log_vals
        df_log['std'] = [0] * len(df_log)
        df_log['reps'] = [1] * len(df_log)
        df_log['m'] = x_vals
        df_log['n'] = x_vals
        df_log['min'] = x_vals
        df_log['max'] = x_vals
        df_log['alg_type'] = [0] * len(df_log)
        df_log['alg_goal'] = [0] * len(df_log)
        df_log['graph_filename'] = [0] * len(df_log)
        df_log['filename'] = [0] * len(df_log)
        df_log['source'] = [0] * len(df_log)
        df_log['iterations'] = [0] * len(df_log)
        df_log['nodes'] = [0] * len(df_log)
        df_log['edges'] = [0] * len(df_log)
        return df_log

    df_log1 = get_df_log(x_vals, log1)
    df_log2 = get_df_log(x_vals, log2)
    df_log3 = get_df_log(x_vals, log3)

    algs_data = {
        'log1': get_plottable_points_grouped(df_log1),
        'log2': get_plottable_points_grouped(df_log2),
        'log3': get_plottable_points_grouped(df_log3)
    }

    produce_plot(algs_data, '$m \log m$ vs $m \log^2 m$ vs $m \log^3 m$', 'log1_vs_log2')

    # Shift BCF VS GOR

    plottable_points_alg = {}
    for alg_name in ['BCF', 'GOR']:
        df_alg = df[df['alg_name'] == alg_name]
        # Select the rows of df_alg where 'exp_name' starts with 'shiftv'
        df_alg = df_alg[df_alg['exp_name'].str.startswith('shiftv')]
        plottable_points = []
        for i in range(1, 7):   # each i represents a different graph size
            df_alg_i = df_alg[df_alg['exp_name'].str.endswith(f'_{i}')]
            if len(df_alg_i) == 0:
                continue

            avg_time_mean = df_alg_i['avg_time'].mean()
            avg_time_std = df_alg_i['avg_time'].std()
            avg_time_min = df_alg_i['avg_time'].min()
            avg_time_max = df_alg_i['avg_time'].max()

            # Take the following info from the first row of df_alg: 'alg_type', 'alg_name', 'graph_filename', 'source', 'iterations', 'nodes', 'edges'
            row = df_alg_i.iloc[0]
            iter_info = {}
            iter_info['avgtime'] = avg_time_mean
            iter_info['std'] = avg_time_std / np.sqrt(5)
            iter_info['min'] = avg_time_min
            iter_info['max'] = avg_time_max
            iter_info['alg_type'] = row['alg_goal']
            iter_info['alg_name'] = row['alg_name']
            iter_info['graph_filename'] = row['filename']
            iter_info['source'] = int(row['source'])
            iter_info['iterations'] = int(row['reps'])
            iter_info['nodes'] = int(row['n'])
            iter_info['edges'] = int(row['m'])
            plottable_points.append(iter_info)
        
        plottable_points_alg[alg_name] = plottable_points
    
    algs_data = {'BCF': plottable_points_alg['BCF'],
                 'GOR': plottable_points_alg['GOR']}

    produce_plot(algs_data, 'SHIFT GOR instances', filename='shift_bcf_vs_gor')

    df_wold = load_all_experiments(drop_old=False)

    # Sshift BCF VS GOR

    df_ssh = df_wold[df_wold['exp_name'].str.startswith('sshift')]
    df_ssh = drop_old_experiments(df_ssh)

    for alg_name in ['BCF', 'GOR']:
        
        df_ssh_alg = df_ssh[df_ssh['alg_name'] == alg_name]
        plottable_points = get_plottable_points_grouped(df_ssh_alg, m_tolerance=0.05)

        algs_data[f'{alg_name}-scaled'] = plottable_points

    produce_plot(algs_data, 'Sshift BCF VS BCF VS GOR', filename='sshift_bcf_vs_gor')

    # Useful function

    def highlight_minmax_col(df, minmax):
        for i, col in enumerate(df.columns):
            if i == 0:
                continue
            min_str = minmax(df[col], key=lambda s: float(s.strip().strip('$').split('\\pm')[0].strip()))
            df[col] = df[col].apply(
                lambda x: '$\\mathbf{' + x[1:-1] + '}$' if str(x) == str(min_str) else x
            )
    
        return df



    # k_factor

    df_kf = df_wold[df_wold['exp_name'].str.startswith('kfact_')]
    df_kf = drop_old_experiments(df_kf)
    df_kf = df_kf[df_kf['timestamp'] > '2024-07-10']
    df_kf = df_kf[df_kf['cutedges'] == 5]
    df_kf = df_kf[df_kf['diam_apprx'] == 0]

    # Collect all the different strings in 'exp_name' after 'kfact'
    instances = df_kf['exp_name'].str.split('_').str[1].unique()

    df_kf_res = []
    df_kf_last = []

    for instance in instances:  # each instance will yield a plot
        df_instance = df_kf[df_kf['exp_name'].str.startswith(f'kfact_{instance}')]

        plottable_points = {}

        # Collect all the different values of "k_factor" in the "k_factor" column
        k_factors = df_instance['k_factor'].unique()

        for k_factor in k_factors:
            df_kf_kf = df_instance[df_instance['k_factor'] == k_factor]

            if len(df_kf_kf) == 0:
                continue

            plottable_points[k_factor] = get_plottable_points(df_kf_kf)
        
        if len(plottable_points) == 0:
            continue

        algs_data = {f'k_factor_{k_factor}': plottable_points[k_factor] for k_factor in plottable_points}

        df_kf_res.append(transform_dataframe(produce_plot(algs_data, f'k_factor_{instance}', deactivate_plots=True)))
        df_kf_res[-1]['instance'] = 'AUG' + instance.upper()

        # df_kf_last.append(pd.DataFrame(columns=['instance'] + [str(k_factor) for k_factor in k_factors]))

        df_kf_last_row = pd.DataFrame({'': '\\bad' + instance.replace('1', 'one').replace('2', 'two')}, index=[0])

        for k_factor in k_factors:
            df_kf_2e7 = df_instance[df_instance['filename'].str.contains('2e7')]
            if len(df_kf_2e7) == 0:
                print('ERROR: 2e7 not found')
                quit()
            
            kf_value = df_kf_2e7[df_kf_2e7['k_factor'] == k_factor]['avg_time'].values[0]
            kf_value = str(int(kf_value))
            df_kf_last_row['$K = ' + str(k_factor) + '$'] = kf_value

        df_kf_last.append(df_kf_last_row)

    df_kf_res = pd.concat(df_kf_res)

    
    # print(df_kf_res.to_latex(index=False))
    df_kf_last = pd.concat(df_kf_last)

    # df_kf_last = highlight_minmax_row(df_kf_last, minmax=min)

    for _, row in df_kf_last.iterrows():
        min_val_index = row.values[1:].astype(int).argmin() + 1
        row.iloc[min_val_index] = '\\mathbf{' + str(row.iloc[min_val_index]) + '}'

    # Apply the conversion function to each cell starting from the second column
    for col in df_kf_last.columns[1:]:
        # df_kf_last[col] = df_kf_last[col].apply(convert_to_latex_sci_notation)

        df_kf_last[col] = '$' + df_kf_last[col] + '$'



    # def convert_to_latex_sci_notation(text):
    #     # Find all integers in the text
    #     match = re.search(r'\d+', text)
    #     if match:
    #         num = int(match.group())
    #         # Convert the number to scientific notation with 2 significant digits
    #         sci_notation = f"{num:.1e}"
    #         # Split the notation into base and exponent
    #         base, exponent = sci_notation.split('e')
    #         # Format the number in LaTeX scientific notation
    #         latex_notation = f"{float(base)} \\cdot 10^{{{int(exponent)}}}"
    #         # Replace the integer in the original string with the LaTeX formatted number
    #         text = text.replace(str(num), latex_notation)
    #     return text

    if False:  # old version
        print(df_kf_last.to_latex(index=False, escape=False, column_format='c' * len(df_kf_last.columns)))

    # k_factor 2e7

    df_kf2e7 = df_wold[df_wold['exp_name'].str.startswith('kfact2e7_')]
    df_kf2e7 = drop_old_experiments(df_kf2e7)

    instances = df_kf2e7['exp_name'].str.split('_').str[1].unique()
    df_kf2e7_res = []

    for instance in instances:
        df_instance = df_kf2e7[df_kf2e7['exp_name'].str.startswith(f'kfact2e7_{instance}')]

        plottable_points = {}

        # Collect all the different values of "k_factor" in the "k_factor" column
        k_factors = df_instance['k_factor'].unique()

        # k_factors = [str(k_factor) if str(k_factor) != '1000000' else '\\infty' for k_factor in k_factors]


        k_to_valerr = {}
        for k_factor in k_factors:
            df_kf2e7_kf = df_instance[df_instance['k_factor'] == k_factor]

            value = df_kf2e7_kf['avg_time'].mean()
            err = df_kf2e7_kf['avg_time'].std() / np.sqrt(len(df_kf2e7_kf))
            value, err = round_value_and_error(value, err, force_int=True)
            k_to_valerr[k_factor] = f'{value} \\pm {err}'


        # Create a one-line DataFrame. Each column is a different value of "k_factor"
        # and it's called "$K=<k_factor>$" where <k_factor> is the value of k_factor
        # The value of each column is of the form "val \\pm err" where val is
        # k_to_valerr[k_factor][0] and err is k_to_valerr[k_factor][1]

        df_kf2e7_last_row = pd.DataFrame.from_dict(k_to_valerr, orient='index').transpose()
        # Rename the column '1000000' to '\\infty'
        df_kf2e7_last_row = df_kf2e7_last_row.rename(columns={1000000: '\\infty'})
        df_kf2e7_last_row.columns = ['$K = ' + (str(k_factor) if str(k_factor) != '1000000' else '\\infty') + '$' for k_factor in k_factors]
        df_kf2e7_last_row[''] = '\\bad' + instance.replace('1', 'one').replace('2', 'two')
        # Move the column '' to the first position
        first_column = df_kf2e7_last_row.pop('')
        df_kf2e7_last_row.insert(0, '', first_column)

        df_kf2e7_res.append(df_kf2e7_last_row)


    df_kf2e7_res = pd.concat(df_kf2e7_res)


    def highlight_min_value(df):
        def extract_value(s):
            try:
                return float(s.split('\\pm')[0].strip())
            except ValueError:
                return np.inf

        def highlight_row(row):
            # Skip the first column
            values = row.iloc[1:].apply(extract_value)
            min_idx = values.idxmin()
            
            # Create a new row with the highlighted minimum
            new_row = row.copy()
            min_value = new_row[min_idx]
            new_row[min_idx] = f"\\mathbf{{{min_value}}}"
            
            return new_row

        # Apply the highlighting to each row
        return df.apply(highlight_row, axis=1)
    
    def highlight_min_values(df):
        def extract_value(s):
            try:
                return float(s.split('\\pm')[0].strip())
            except ValueError:
                return np.inf

        def highlight_row(row):
            # Skip the first column
            values = row.iloc[1:].apply(extract_value)
            min_value = values.min()
            
            # Create a new row with the highlighted minimums
            new_row = row.copy()
            for idx, value in values.items():
                if value == min_value:
                    new_row[idx] = f"\\mathbf{{{row[idx]}}}"
            
            return new_row

        # Apply the highlighting to each row
        return df.apply(highlight_row, axis=1)

    df_kf2e7_res = highlight_min_values(df_kf2e7_res)

    # for _, row in df_kf2e7_res.iterrows():
    #     min_val_index = row.values[1:].astype(int).argmin() + 1
    #     row.iloc[min_val_index] = '\\mathbf{' + str(row.iloc[min_val_index]) + '}'

    # Apply the conversion function to each cell starting from the second column
    for col in df_kf2e7_res.columns[1:]:
        # df_kf2e7_res[col] = df_kf2e7_res[col].apply(convert_to_latex_sci_notation)

        df_kf2e7_res[col] = '$' + df_kf2e7_res[col] + '$'

    print(df_kf2e7_res.to_latex(index=False, escape=False, column_format='c' * len(df_kf2e7_res.columns)))


    # norm_bcf_vs_gor new

    df_norm = df_wold[df_wold['exp_name'].str.match(r'nbvg_\w*_\w*_\d_\d')]

    # We made experiments with a modification in the code and I set the seed to 9876 to
    # mark that something different was happening.
    # df_norm = df_norm[(df_norm['alg_name'] == 'GOR') | ((df_norm['alg_name'] == 'BCF') & (df_norm['cutedgesseed'] == 9876))]
    df_norm = df_norm[(df_norm['alg_name'] == 'GOR') | ((df_norm['alg_name'] != 'BCF') | (df_norm['cutedgesseed'] != 9876))]

    df_norm = drop_old_experiments(df_norm)

    instances = df_norm['exp_name'].str.split('_').str[2].unique()

    for instance in instances:
        for alg_name in ['bcf', 'gor']:
            df_instance_alg = df_norm[df_norm['exp_name'].str.startswith(f'nbvg_{alg_name}_{instance}')]
            if alg_name == 'bcf':
                df_instance_alg = df_instance_alg[df_instance_alg['diam_apprx'] == 1]
            # if instance == 'rand':
            #     plottable_points_alg[alg_name] = get_plottable_points(df_instance_alg)
            #     continue
            plottable_points = []
            for i in range(1, 7):   # each i represents a different graph size
                df_instance_alg_i = df_instance_alg[df_instance_alg['exp_name'].str.endswith(f'_{i}')]
                if len(df_instance_alg_i) == 0:
                    continue

                avg_time_mean = df_instance_alg_i['avg_time'].mean()
                avg_time_std = df_instance_alg_i['avg_time'].std()
                avg_time_min = df_instance_alg_i['avg_time'].min()
                avg_time_max = df_instance_alg_i['avg_time'].max()
                avg_time_count = df_instance_alg_i['avg_time'].count()

                # Take the following info from the first row of df_instance_alg: 'alg_type', 'alg_name', 'graph_filename', 'source', 'iterations', 'nodes', 'edges'
                row = df_instance_alg_i.iloc[0]
                iter_info = {}
                iter_info['avgtime'] = avg_time_mean
                iter_info['std'] = avg_time_std / np.sqrt(avg_time_count)
                iter_info['min'] = avg_time_min
                iter_info['max'] = avg_time_max
                iter_info['alg_type'] = row['alg_goal']
                iter_info['alg_name'] = row['alg_name']
                iter_info['graph_filename'] = row['filename']
                iter_info['source'] = int(row['source'])
                iter_info['iterations'] = int(row['reps'])
                iter_info['nodes'] = int(row['n'])
                iter_info['edges'] = int(row['m'])
                plottable_points.append(iter_info)
            
            plottable_points_alg[alg_name] = plottable_points
        
        algs_data = {f'BCF': plottable_points_alg['bcf'], f'GOR': plottable_points_alg['gor']}

        plot_title = f'AUG {instance.upper()} instances' if instance != 'rand' else f'RANDOM RESTRICTED instances'

        produce_plot(algs_data, plot_title, filename=f'{instance}_bcf_vs_gor')


    # DAG BCF vs GOR

    df_bad = df_wold[df_wold['exp_name'].str.match(r'bad_\w*_\w*_\d_\d')]

    df_bad = drop_old_experiments(df_bad)

    df_bad = df_bad[df_bad['reps'] == 5]

    df_bad['instance'] = df_bad['exp_name'].str.split('_').str[2]
    df_bad['instance'] = '\\bad' + df_bad['instance'].str.lower().replace({'1': 'one', '2': 'two'}, regex=True)
    df_bad['std'] = df_bad['std'] / df_bad['reps'].apply(lambda reps: np.sqrt(reps))
    # df_bad['time_std'] = df_bad.apply(lambda row: f"{row['avg_time']} ± {row['std']}", axis=1)
    # df_bad['time_std'] = df_bad.apply(lambda row: ' ± '.join(str(elem) for elem in round_value_and_error(row['avg_time'], row['std'])), axis=1)
    def round_value_and_error_to_latex_string(value, err):
        value, err = round_value_and_error(value, err)
        return f'${value} \\pm {err}$'
    df_bad['time_std'] = df_bad.apply(lambda row: round_value_and_error_to_latex_string(row['avg_time'], row['std']), axis=1)

    df_bad['alg_name'] = df_bad['alg_name'].apply(lambda x: '\\' + x)

    # Rename column 'instance' with ''
    df_bad = df_bad.rename(columns={'instance': ''})


    # Pivot the DataFrame
    df_bad = df_bad.pivot(index='alg_name', columns='', values=['time_std'])
    df_bad = df_bad['time_std']
    df_bad = df_bad.reset_index()
    df_bad = df_bad.rename(columns={'alg_name': ''})

    df_bad = highlight_minmax_col(df_bad, min)

    # for i, col in enumerate(df_bad.columns):
    #     if i == 0:
    #         continue
    #     min_str_len = max(len(s) for s in df_bad[col])
    #     df_bad[col] = df_bad[col].apply(
    #         lambda x: '$\\mathbf{' + x[1:-1] + '}$' if len(x) == min_str_len else x
    #     )


    print(df_bad.to_latex(index=False, escape=False, column_format='c' * len(df_bad.columns)))


    # diam_apprx

    df_da = df_wold[df_wold['exp_name'].str.startswith('diam')]
    df_da = df_da[df_da['cutedges'] == 5]
    # df_da = df_da[df_da['timestamp'] < '2024-07-16']

    df_da = drop_old_experiments(df_da)

    # Collect all the different strings in 'exp_name' after 'kfact'
    instances = df_da['exp_name'].str.split('_').str[1].unique()

    for instance in instances:  # each instance will yield a plot
        df_instance = df_da[df_da['exp_name'].str.startswith(f'diam_{instance}')]

        plottable_points = {}

        # Collect all the different values of "k_factor" in the "k_factor" column
        diam_apprxs = df_instance['diam_apprx'].unique()

        for diam_apprx in diam_apprxs:
            df_da_da = df_instance[df_instance['diam_apprx'] == diam_apprx]

            if len(df_da_da) == 0:
                continue

            # plottable_points[diam_apprx] = get_plottable_points(df_da_da)
            plottable_points[diam_apprx] = get_plottable_points_grouped(df_da_da)
        
        if len(plottable_points) == 0:
            continue

        algs_data = {f'diam_apprx = {bool(diam_apprx)}': plottable_points[diam_apprx] for diam_apprx in plottable_points}

        instance_type = f'AUG {instance.upper()} instances' if instance != 'rand' else f'RANDOM RESTRICTED instances'

        produce_plot(algs_data, f'Parameter: diam_apprx. {instance_type}', filename=f'diam_apprx_{instance}')

    
    # rand_label

    df_rl = df_wold[df_wold['exp_name'].str.startswith('randl')]

    df_rl = drop_old_experiments(df_rl)

    # Collect all the different strings in 'exp_name' after 'kfact'
    instances = df_rl['exp_name'].str.split('_').str[1].unique()

    for instance in instances:  # each instance will yield a plot
        df_instance = df_rl[df_rl['exp_name'].str.startswith(f'randl_{instance}')]

        plottable_points = {}

        # Collect all the different values of "k_factor" in the "k_factor" column
        rand_labels = df_instance['rand_label'].unique()

        for rand_label in rand_labels:
            df_rl_rl = df_instance[df_instance['rand_label'] == rand_label]

            if len(df_rl_rl) == 0:
                continue

            plottable_points[rand_label] = get_plottable_points(df_rl_rl)
        
        if len(plottable_points) == 0:
            continue

        algs_data = {f'rand_label = {bool(rand_label)}': plottable_points[rand_label] for rand_label in plottable_points}

        instance_type = f'AUG {instance.upper()} instances' if instance != 'rand' else f'RANDOM RESTRICTED instances'

        produce_plot(algs_data, f'Parameter: rand_label. {instance_type}', filename=f'rand_label_{instance}')

    # use_lazy

    df_ul = df_wold[df_wold['exp_name'].str.startswith('uselazy')]

    df_ul = drop_old_experiments(df_ul)

    # Collect all the different strings in 'exp_name' after 'kfact'
    instances = df_ul['exp_name'].str.split('_').str[1].unique()

    for instance in instances:  # each instance will yield a plot
        df_instance = df_ul[df_ul['exp_name'].str.startswith(f'uselazy_{instance}')]

        plottable_points = {}

        # Collect all the different values of "k_factor" in the "k_factor" column
        use_lazies = df_instance['use_lazy'].unique()

        for use_lazy in use_lazies:
            df_ul_ul = df_instance[df_instance['use_lazy'] == use_lazy]

            if len(df_ul_ul) == 0:
                continue

            # plottable_points[use_lazy] = get_plottable_points(df_ul_ul)
            plottable_points[use_lazy] = get_plottable_points_grouped(df_ul_ul, m_tolerance=0.01)
        
        if len(plottable_points) == 0:
            continue

        algs_data = {f'use_lazy = {bool(use_lazy)}': plottable_points[use_lazy] for use_lazy in plottable_points}

        instance_type = f'AUG {instance.upper()} instances' if instance != 'rand' else f'RANDOM RESTRICTED instances'

        produce_plot(algs_data, f'Parameter: use_lazy. {instance_type}', filename=f'use_lazy_{instance}')

    # USA

    df_usa = df_wold[df_wold['exp_name'].str.startswith('usa')]

    df_usa = drop_old_experiments(df_usa)

    algs_names = df_usa['alg_name'].unique()

    df_usa_rows = []

    for alg_name in algs_names:
        df_alg = df_usa[df_usa['alg_name'] == alg_name]

        if len(df_alg) == 0:
            continue

        df_usa_row = pd.DataFrame({'': '\\' + alg_name.upper()}, index=[0])

        exp_names = sorted(df_alg['exp_name'].unique())
        exp_names = [exp_name.split('_')[-1] for exp_name in exp_names]

        for exp_name in exp_names:
            avg_time = df_alg[df_alg['exp_name'] == f'usa_{alg_name.lower()}_{exp_name}']['avg_time'].values[0]
            std = df_alg[df_alg['exp_name'] == f'usa_{alg_name.lower()}_{exp_name}']['std'].values[0]
            avg_time, std = round_value_and_error(avg_time, std)
            df_usa_row['\\usa{' + str(exp_name) + '}'] = f'${avg_time} \pm {std}$'

        df_usa_rows.append(df_usa_row)

    df_usa_last = pd.concat(df_usa_rows)


    df_usa_last = highlight_minmax_col(df_usa_last, minmax=min)
    
    # def format_error_string(string):
    #     """
    #     Extracts integer values from a string, formats them in scientific notation,
    #     and returns the formatted string.

    #     Args:
    #         string: String containing "integer value \pm integer value".

    #     Returns:
    #         Formatted string in LaTeX scientific notation.
    #     """

    #     match = re.search(r"(\d+)\s*\pm\s*(\d+)", string)
    #     if match:
    #         value, error = int(match.group(1)), int(match.group(2))
    #         exponent = len(str(value)) - 1
    #         formatted_value = f"({value/10**exponent:.2f} \\pm {error/10**exponent:.1f}) 10^{{{exponent}}}"
    #         return string.replace(match.group(0), formatted_value)
    #     else:
    #         return string
    #     return s

    # for col in df_usa_last.columns[1:]:
    #     df_usa_last[col] = df_usa_last[col].apply(format_error_string)

    print(df_usa_last.to_latex(index=False, escape=False, column_format='c' * len(df_usa_last.columns)))


# %%

if __name__ == '__main__':
    # main(deactivate_plots=True)
    main(deactivate_plots=False)

# %% MANUAL EXPERIMENTS

df = load_all_experiments(drop_old=False)
# %%
