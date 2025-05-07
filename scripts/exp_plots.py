# %%
import os

# Collect all the groups of filenames in the folders ../data/experiments and ../data/queries with the same name
# and put them in a list

filename_pairs = set()
for filename in os.listdir("../data/experiments"):
    if filename.endswith(".txt"):
        filename_pairs.add(filename)

filename_pairs2 = set()
for filename in os.listdir("../data/queries"):
    if filename.endswith(".txt"):
        filename_pairs2.add(filename)

common_filenames = filename_pairs & filename_pairs2
# %%
import re

# Group all filenames that have the same name and differ only by a substring that can be "_bf", "_bcf" or "_gor"
# All the single files will have a single plot

algs = ["bf", "bcf", "gor", "bfct"]
grouped = {}

for filename in common_filenames:
    tent_filename = str(filename)
    # Remove the ".txt" extension
    tent_filename = re.sub(r"\.txt$", "", tent_filename)
    inner_string = '(?:(?:[^a-zA-Z0-9])|(?:$)))|(?:_'.join(algs)  # _inner_string(?:(?:[^a-zA-Z0-9])|(?:$))
    unlabeled_group = f'(?:_{inner_string}(?:(?:[^a-zA-Z0-9])|(?:$)))'
    # for alg in algs:
    #     tent_filename = re.sub(fr"(_{alg})(?!.*\1)", "", tent_filename)
    tent_filename = re.sub(fr"({unlabeled_group})(?!.*(?:{unlabeled_group}))", "", tent_filename)
    if tent_filename in grouped:
        grouped[tent_filename].append(filename)
    else:
        grouped[tent_filename] = [filename]

# %%
def extract_times(filename):
    with open(filename, 'r') as f:
        string = f.read()
    # print(string)
    times_strings = ['Average', 'std', 'min', 'max']
    all_times = []
    for time_string in times_strings:
        pattern = fr'{time_string}: (\d+(?:\.\d+)?(?:e[+-]\d+)?) ms'
        matches = re.findall(pattern, string)
        all_times.append([float(match) for match in matches])
    
    # Transposing the list of lists
    transp_all_times = []
    for i in range(len(all_times[0])):
        transp_all_times.append([all_times[j][i] for j in range(len(all_times))])

    return transp_all_times

# %%
# timestamp,algorithm type (SSSP or negcycle),specific algorithm used,graph filename (no path),#nodes,#edges,iterations,avgtime,std,min,max
def extract_info(filename):
    with open(filename, 'r') as f:
        string = f.read().splitlines()

    # print(string)

    data = []
    for line in string:
        if line.strip() == '': continue
        splited_line = line.split()
        if splited_line[1] != 'time': continue

        iter_data = {'alg_type': splited_line[0],
                     'alg_name': splited_line[2],
                     'graph_filename': splited_line[3],
                     'source': int(splited_line[4]),
                     'iterations': int(splited_line[5])}

        # Read the first line from the file of the graph and count the number of lines

        with open(splited_line[3], 'r') as f:
            # Read the first line
            iter_data['nodes'] = int(f.readline())
            # Count the number of lines
            iter_data['edges'] = sum(1 for line in f)

        data.append(iter_data)


    return data

# %%
from scipy.odr import ODR, Model, Data
from scipy.stats import norm

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

# %%
import numpy as np
from sklearn.linear_model import LinearRegression
from copy import deepcopy
def log_regression(x, y, x_error=None, y_error=None):
    x_copy = deepcopy(x)
    y_copy = deepcopy(y)
    x_error_copy = deepcopy(x_error)
    y_error_copy = deepcopy(y_error)

    # x_copy = np.log10(x_copy).reshape(-1, 1)
    x_copy = np.log10(x_copy)
    y_copy = np.log10(y_copy)

    # If we assume that the y errors are very small compared tu the y values, we can
    # treat them as the same without doing the log

    # # Create and fit the linear regression model
    # model = LinearRegression()
    # model.fit(x_copy, y_copy)

    # # Get the intercept and slope
    # intercept = model.intercept_
    # slope = model.coef_[0]

    slope, intercept, slope_stderr, intercept_stderr = orth_weigh(x_copy, y_copy, x_weights=x_error_copy, y_weights=y_error_copy)


    # intercept = 10 ** intercept

    return intercept, slope, intercept_stderr, slope_stderr

# %%
import matplotlib.pyplot as plt

def produce_plot(algs_data, name):
    # Define the colormap
    cmap = plt.get_cmap('tab20')

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Iterate over each set of points and interpolations
    for i, (alg_name, alg_elems) in enumerate(algs_data.items()):
        # Get the color for the current group from the colormap
        color = cmap(i % cmap.N)

        x, y, y_err = [], [], []
        for elem in alg_elems:
            x.append(elem['nodes'])
            y.append(elem['avgtime'])
            y_err.append(elem['std'])
        
        # print(x, y)
        
        log10_intercept, slope, log10_intercept_err, slope_err = log_regression(x, y)
        # log10_intercept, slope, log10_intercept_err, slope_err = log_regression(x, y, y_error=y_err)

        intercept = 10 ** log10_intercept

        max_intercept = 10 ** (log10_intercept + log10_intercept_err)
        min_intercept = 10 ** (log10_intercept - log10_intercept_err)
        intercept_err = (max_intercept - min_intercept) / 2  # not too correct mathematically

        # Plot the points
        # ax.scatter(x, y, label='_nolegend_', color=color)
        ax.errorbar(x, y, yerr=y_err, label='_nolegend_', fmt='o', color=color, linestyle='', markersize=4)
        
        x_range = np.linspace(min(x), max(x), 100)
        y_range = intercept * x_range ** slope
        
        # Plot the interpolation line
        ax.plot(x_range, y_range, label=f'{alg_name} (({intercept:.2g} +- {intercept_err:.2g}) * x^({slope:.2g} +- {slope_err:.2g}))', color=color)
            

    # Set the x-axis and y-axis to log scale
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Add labels and legend
    ax.set_xlabel('# nodes')
    ax.set_ylabel('times')
    ax.legend()
    ax.set_title(name)

    # Save the plot as a PNG file
    plt.savefig(f'../data/plots/{name}.png')
    # plt.show()


# %%
from tqdm import tqdm

# for group_name, group_elems in {'mbfct_trend': ['mbfct_trend_bcf.txt', 'mbfct_trend_bf.txt']}.items():
for group_name, group_elems in tqdm(grouped.items()):
    algs_data = {}
    for filename in group_elems:
        times = extract_times(f"../data/experiments/{filename}")
        info = extract_info(f"../data/queries/{filename}")

        # print(times)
        # print(info)

        if len(times) != len(info):
            print(f"{filename} has {len(times)} times and {len(info)} info")
            continue

        for iter_info, iter_time in zip(info, times):
            iter_info['avgtime'] = iter_time[0]
            iter_info['std'] = iter_time[1] / np.sqrt(iter_info['iterations'])
            iter_info['min'] = iter_time[2]
            iter_info['max'] = iter_time[3]
            if iter_info['alg_name'] in algs_data:
                algs_data[iter_info['alg_name']].append(iter_info)
            else:
                algs_data[iter_info['alg_name']] = [iter_info]

    # print(algs_data)

    produce_plot(algs_data, group_name)
# %%
