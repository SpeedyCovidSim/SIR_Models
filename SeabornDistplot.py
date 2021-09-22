from matplotlib.colors import Colormap
import seaborn as sns
from matplotlib import pyplot as plt
from seaborn.palettes import color_palette

def seabornDist(x_vector, hue_vector, hue_order_vector, title, outputFileName="plot", Display=True, save=True):
    '''
    Inputs
    t              : Array of times at which events have occured


    Outputs
    png            :
    '''
    # set up figure # histogram of Reff in AL4
    sns.set()
    sns.set_style('ticks')
    sns.set_color_codes('pastel')
    f = plt.figure(figsize=[6,5],dpi=300)
    # palette = ['#A1C9F4', '#FF9F9B', '#8DE5A1', '#B9F2F0']
    palette = ['#A1C9F4', '#FF9F9B','#8DE5A1', '#B9F2F0']

    # Seaborn.hist(x=models_df.R_eff_after_al[conditionIndexes], bins=16, weights=ones(length(conditionIndexes))/length(conditionIndexes))

    # sns.displot(x=x_vector, hue=hue_vector, common_norm=False, cut=0, bw_adjust=0.7, stat="density", kind="kde", fill=True, palette=sns.color_palette(palette, len(palette)))

    sns.histplot(x=x_vector, hue=hue_vector, hue_order=hue_order_vector, bins=12, common_norm=False, multiple='layer', kde=True, kde_kws={'cut':0, 'bw_adjust':0.7}, element='step', stat="probability", fill=True, palette=sns.color_palette(palette, len(palette)), )

    # else
    # end
    plt.xlabel("Reff")
    plt.ylabel("Probability")

    plt.title(title)
    plt.tight_layout(h_pad=0.01)
    # plt.legend(loc = 'upper right')

    # if Display:
    #     # required to display graph on plots.
    #     plt.show()

    if save:
        # Save graph as pngW
        plt.savefig(outputFileName)
    plt.close()

# pastel_colors = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF', '#DEBB9B', '#FAB0E4', '#CFCFCF', '#FFFEA3', '#B9F2F0']

def plotReffHist(x_vector, title, xlab, outputFileName="plot", Display=True, save=True):

    # set up figure # histogram of Reff in AL4
    sns.set()
    sns.set_style('ticks')
    sns.set_color_codes('pastel')
    f = plt.figure(figsize=[6,4],dpi=300)
    # palette = ['#A1C9F4', '#FF9F9B', '#8DE5A1', '#B9F2F0']
    # palette = ['#A1C9F4', '#FF9F9B','#8DE5A1', '#B9F2F0']

    # Seaborn.hist(x=models_df.R_eff_after_al[conditionIndexes], bins=16, weights=ones(length(conditionIndexes))/length(conditionIndexes))

    # sns.displot(x=x_vector, hue=hue_vector, common_norm=False, cut=0, bw_adjust=0.7, stat="density", kind="kde", fill=True, palette=sns.color_palette(palette, len(palette)))

    sns.histplot(x=x_vector, bins=18, kde=True, kde_kws={'cut':0, 'bw_adjust':0.7}, element='bars', stat="probability", fill=True)

    # else
    # end
    plt.xlabel(xlab)
    plt.ylabel("Probability")

    plt.title(title)
    plt.tight_layout(h_pad=0.01)
    # plt.legend(loc = 'upper right')

    # if Display:
    #     # required to display graph on plots.
    #     plt.show()

    if save:
        # Save graph as pngW
        plt.savefig(outputFileName)

    plt.close()

# seabornDist([1,2,2,2,3,3,4], [1,1,1,1,1,2,2], 'hi', 'nope', True, False)

# print(sns.color_palette('pastel', as_cmap=['g']))
