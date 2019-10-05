import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as _stats
import pandas as pd
import numpy as np

def print_values_of_type(data, data_type):
    cat_vars = data.select_dtypes(include=data_type)
    for att in cat_vars:
        print(att, data[att].unique())

def print_missing_values(data):
    fig = plt.figure(figsize=(10, 7))
    mv = {}
    for var in data:
        mv[var] = data[var].isna().sum()
        bar_chart(plt.gca(), mv.keys(), mv.values(), 'Number of missing values per variable', var,
                       'nr. missing values')
    fig.tight_layout()
    plt.show()

def print_attribute_distribution(data, attribute):
    plt.figure(figsize=(5, 4))
    plt.plot(data[attribute])
    plt.show()

def singular_boxplot(data):
    columns = data.select_dtypes(include='number').columns
    rows, cols = choose_grid(len(columns))
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
    i, j = 0, 0
    for n in range(len(columns)):
        axs[i, j].set_title('Boxplot for %s' % columns[n])
        axs[i, j].boxplot(data[columns[n]].dropna().values)
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    fig.tight_layout()
    plt.show()

def bar_chart(ax: plt.axes, xvalues: list, yvalues: list, title: str, xlabel: str, ylabel: str, percentage=False):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.set_xticklabels(xvalues, rotation=90, fontsize='small')
    if percentage:
        ax.set_ylim(0.0, 1.0)
    ax.bar(xvalues, yvalues, edgecolor='grey')

def hist_each_numeric_var(data):
    columns = data.select_dtypes(include='number').columns
    rows, cols = choose_grid(len(columns))
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
    i, j = 0, 0
    for n in range(len(columns)):
        axs[i, j].set_title('Histogram for %s' % columns[n])
        axs[i, j].set_xlabel(columns[n])
        axs[i, j].set_ylabel("probability")
        axs[i, j].set_yscale('log')
        axs[i, j].hist(data[columns[n]].dropna().values, 'auto')
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    fig.tight_layout()
    plt.show()

def hist_categorical_var(data, category):
    columns = data.select_dtypes(include=category).columns
    rows, cols = choose_grid(len(columns))
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
    i, j = 0, 0
    for n in range(len(columns)):
        counts = data[columns[n]].dropna().value_counts(normalize=True)
        bar_chart(axs[i, j], counts.index, counts.values, 'Histogram for %s' % columns[n], columns[n],
                       'probability')
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    fig.tight_layout()
    plt.show()

def display_best_fit_var(data):
    columns = data.select_dtypes(include='number').columns
    rows, cols = choose_grid(len(columns))
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
    i, j = 0, 0
    for n in range(len(columns)):
        axs[i, j].set_title('Histogram with trend for %s' % columns[n])
        axs[i, j].set_ylabel("probability")
        axs[i, j].set_yscale('log')
        sns.distplot(data[columns[n]].dropna().values, norm_hist=True, ax=axs[i, j], axlabel=columns[n])
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    fig.tight_layout()
    plt.show()

def choose_grid(nr):
    return nr // 4 + 1, 4

def fit_different_distributions(data):
    columns = data.select_dtypes(include='number').columns
    rows, cols = choose_grid(len(columns))
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
    i, j = 0, 0
    for n in range(len(columns)):
        histogram_with_distributions(axs[i, j], data[columns[n]].dropna(), columns[n])
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)
    fig.tight_layout()
    plt.show()

def compute_known_distributions(x_values, n_bins) -> dict:
    distributions = dict()
    # Gaussian
    mean, sigma = _stats.norm.fit(x_values)
    distributions['Normal(%.1f,%.2f)' % (mean, sigma)] = _stats.norm.pdf(x_values, mean, sigma)
    # LogNorm
    sigma, loc, scale = _stats.lognorm.fit(x_values)
    distributions['LogNor(%.1f,%.2f)'%(np.log(scale),sigma)] = _stats.lognorm.pdf(x_values, sigma, loc, scale)
    # Exponential
    loc, scale = _stats.expon.fit(x_values)
    distributions['Exp(%.2f)' % (1 / scale)] = _stats.expon.pdf(x_values, loc, scale)
    # SkewNorm
    # a, loc, scale = _stats.skewnorm.fit(x_values)
    # distributions['SkewNorm(%.2f)'%a] = _stats.skewnorm.pdf(x_values, a, loc, scale)
    return distributions

def histogram_with_distributions(ax: plt.Axes, series: pd.Series, var: str):
    values = series.sort_values().values
    n, bins, patches = ax.hist(values, 20, density=True, edgecolor='grey')
    distributions = compute_known_distributions(values, bins)
    multiple_line_chart(ax, values, distributions, 'Best fit for %s' % var, var, 'probability')

def multiple_line_chart(ax: plt.Axes, xvalues: list, yvalues: dict, title: str, xlabel: str, ylabel: str, percentage=False):
    legend: list = []
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    if percentage:
        ax.set_ylim(0.0, 1.0)

    for name, y in yvalues.items():
        ax.plot(xvalues, y)
        legend.append(name)
    ax.legend(legend, loc='best', fancybox = True, shadow = True)

def granularity(data):
    columns = data.select_dtypes(include='number').columns
    rows = len(columns)
    cols = 5
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
    bins = range(5, 100, 20)
    for i in range(len(columns)):
        for j in range(len(bins)):
            axs[i, j].set_title('Histogram for %s' % columns[i])
            axs[i, j].set_xlabel(columns[i])
            axs[i, j].set_ylabel("probability")
            axs[i, j].set_yscale('log')
            axs[i, j].hist(data[columns[i]].dropna().values, bins[j])
    fig.tight_layout()

    plt.show()

# MULTIVARIABLE CORELATION
def sparsity(data):
    columns = data.select_dtypes(include='number').columns
    rows, cols = len(columns) - 1, len(columns) - 1
    plt.figure()
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)
    for i in range(len(columns)):
        var1 = columns[i]
        for j in range(i + 1, len(columns)):
            var2 = columns[j]
            axs[i, j - 1].set_title("%s x %s" % (var1, var2))
            axs[i, j - 1].set_xlabel(var1)
            axs[i, j - 1].set_ylabel(var2)
            axs[i, j - 1].scatter(data[var1], data[var2])
    fig.tight_layout()
    plt.show()

def correlation_analysis(data):
    fig = plt.figure(figsize=[12, 12])
    corr_mtx = data.corr()
    sns.heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
    plt.title('Correlation analysis')
    plt.show()


