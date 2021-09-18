import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_missing(missing):
    # Histogram of missing values
    plt.style.use('seaborn-white')
    plt.figure(figsize=(7, 5))
    plt.hist(missing, bins=np.linspace(0, 1, 11), edgecolor='k', color='red',linewidth=1.5)
    plt.xticks(np.linspace(0, 1, 11))
    plt.xlabel('Missing-Fraction', size=14)
    plt.ylabel('Number of Features', size=14)
    plt.title("Histogram", size=16)
    plt.show()


def plot_unique(unique):
    unique.plot.hist(edgecolor='k', figsize=(7, 5))
    plt.ylabel('Frequency', size=14)
    plt.xlabel('Unique Values', size=14)
    plt.title('Number of Unique Values Histogram', size=16)
    plt.show()

def plot_collinear(corr_matrix,record_coll):

    corr_matrix_plot = corr_matrix.loc[list(set(record_coll['corr_feature'])),list(set(record_coll['drop_feature']))]
    title = "Correlations Above Threshold"
    f, ax = plt.subplots(figsize=(10, 8))

    # Diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with a color bar
    sns.heatmap(corr_matrix_plot, cmap=cmap, center=0,linewidths=.25, cbar_kws={"shrink": 0.6})

    # Set the ylabels
    ax.set_yticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[0]))])
    ax.set_yticklabels(list(corr_matrix_plot.index), size=int(160 / corr_matrix_plot.shape[0]))

    # Set the xlabels
    ax.set_xticks([x + 0.5 for x in list(range(corr_matrix_plot.shape[1]))])
    ax.set_xticklabels(list(corr_matrix_plot.columns), size=int(160 / corr_matrix_plot.shape[1]))
    plt.title(title, size=14)
    plt.show()

def plot_feature_importances(feature_impt,plot_n=15):
    # Need to adjust number of features if greater than the features in the data
    if plot_n > feature_impt.shape[0]:
        plot_n = feature_impt.shape[0] - 1

    plt.figure(figsize=(10, 6))
    ax = plt.subplot()

    ax.barh(list(reversed(list(feature_impt.index[:plot_n]))),feature_impt['normalized_importance'][:plot_n],align='center', edgecolor='k')

    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(feature_impt.index[:plot_n]))))
    ax.set_yticklabels(feature_impt['feature'][:plot_n], size=12)

    # Plot labeling
    plt.xlabel('Normalized Importance', size=16)
    plt.title('Feature Importances', size=18)
    plt.show()

    # Cumulative importance plot
    plt.figure(figsize=(6, 4))
    plt.plot(list(range(1, len(feature_impt) + 1)), feature_impt['cumulative_importance'],'r-')
    plt.xlabel('Number of Features', size=14)
    plt.ylabel('Cumulative Importance', size=14)
    plt.title('Cumulative Feature Importance', size=16)

