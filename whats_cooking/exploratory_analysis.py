from __future__ import division

import sys
import os
from math import pi

import numpy as np
import pandas as pd

from collections import Counter
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.5)

# =========
# FUNCTIONS
# =========
def create_folder(complete_path):
    """
    Function to create a folder.

    Parameter
    ---------
    complete_path : str
        Complete path of the new folder.

    Returns
    -------
    Create the new folder.
    """
    if not os.path.exists(complete_path):
        os.makedirs(complete_path)

    return 0

def clean_cuisine_names(cuisine_names):
    """
    String manipulation of cuisine names.

    Parameter:
    ---------
    cuisine_names : list
        List containg the cuisine names.

    Returns:
    -------
    clean_names : list
        List with the with the new names.
    """
    clean_names = []

    for i, name in enumerate(cuisine_names):
        new_name = name.title()

        if new_name.find('_') > 0:
            new_name = new_name.replace('_', ' ')

        clean_names.append(new_name)

    return clean_names

def parallel_counting(data):
    """
    Auxiliary function for parallel counting.
    """
    return data.map(Counter).sum()

def ingredients_counter(data):
    """
    Function to count the ingredients in parallel fashion.

    Parameter:
    ---------
    data : pandas series
        Pandas Series object with the ingredients for counting.

    Returns:
    -------
    ingredients_count : pandas series
        Series with count for each ingredient.

    Note:
    ----
    The ingredients are returned in descending order
    """
    # Let's make this counter process parallel
    # using the 'multiprocessing' library
    cores = cpu_count()

    # separate data into chunks for the parallel processing
    data_chunks = np.array_split(data, cores)

    pool = Pool(cores)
    counter_list = pool.map(parallel_counting, data_chunks)
    pool.close()

    ingredients_count = pd.Series(sum(counter_list, \
    Counter())).sort_values(ascending=False)

    return ingredients_count

def cuisine_ingredients(data_frame, cuisine_name):
    """
    This function gets the specific ingredients for a particular cuisine.

    Parameters:
    ----------
    data_frame : pandas dataframe
        The original dataset.
    cuisine_name : str
        Cuisine name as written in the original dataset.
            - mexican
            - italian
            - cajun_creole ...

    Returns:
    -------
    data : pandas series
        Pandas Series with the ingredients for the specific cuisine.
    """
    data = data_frame['ingredients'][data_frame['cuisine']==cuisine_name]
    return pd.Series(data)

def ingredients_radar_plot(data1, data2, cuisine_names, ingredients_names):
    """
    Function to create and plot a radar graph comparing
    ingredients of two different cuisines.

    Parameters:
    ----------
    data1 : pandas series
        Series with the ingredients of the first cuisine.
    data2 : pandas series
        Series with the ingredients of the seocnd cuisine.
    cuisine_names : list
        List cotainig the names of each cuisine ti comparison.
    ingredients_names : list
        List containing the names of teh ingredients that one
        would like to compare.

    Returns:
    -------
    Radar Plot on Screen.

    Note:
    ----
    The qunatity of ingredients is normalized
    """
    # string manipulation of cuisine names
    clean_names = []
    for i, name in enumerate(ingredients_names):
        name = name.title() # capitalize each word

        if name.find('_') > 0: # found an underscore in the string name
            name = name.replace('_', ' ') # replace it with a space
        clean_names.append(name)

    # PART 1: Create Background
    N = len(ingredients_names)

    # What will be the angle of each axis in the plot?
    # R: plot / number of variables.
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles.append(angles[0]) # We need to repeat the first values
                             # to close the circular graph

    # Initialise the spider plot
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels yet
    plt.xticks(angles[:-1], clean_names)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks(color="grey", size=13)

    # PART 2: Add plots
    # Ind1
    values = list(data1[ingredients_names] / sum(data1[ingredients_names]))
    values.append(values[0])
    ax.plot(angles, values, linewidth=1, linestyle='solid',
            label=cuisine_names[0], color='b')
    ax.fill(angles, values, 'b', alpha=0.1)

    # Ind2
    values = list(data2[ingredients_names] / sum(data2[ingredients_names]))
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid',
            label=cuisine_names[1], color='r')
    ax.fill(angles, values, 'r', alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1))

if __name__ == '__main__':

    # =======
    # FOLDERS
    # =======
    package_path = os.path.dirname(os.getcwd())
    data_path = package_path + '/data/'

    # create folder for figures
    fig_path = package_path + '/figures/'
    create_folder(fig_path)

    # =========
    # LOAD DATA
    # =========
    input_name = 'train.json'
    input_file = data_path + input_name

    df = pd.read_json(input_file)

    # get the total number of recipes
    n_recipes = df.shape[0]
    print('>> Data <<')
    print('    The training dataset has %i recipes.\n' % (n_recipes))

    # ========
    # CUISINES
    # ========
    cuisine = df['cuisine'].value_counts()
    n_cuisines = cuisine.nunique()
    print('>> Cuisines <<')
    print('    This dataset has %i different cuisines.' % n_cuisines)

    cuisine_names = list(cuisine.index)
    cuisine_values = list(cuisine.values)
    cuisine_clean_names = clean_cuisine_names(cuisine_names)


    # cuisines bar plot
    fig_file = fig_path + 'cuisines_barplot.pdf'

    plt.figure(figsize=(10, 7))
    sns.barplot(x=cuisine_values,
                y=cuisine_clean_names,
                edgecolor=(0, 0, 0),
                linewidth=1)
    plt.xlabel('Counts')
    plt.ylabel('Cuisines')
    plt.savefig(fig_file, bbox_inches='tight', dpi=1200)
    plt.close()

    # cuisines pie chart
    fig_file = fig_path + 'cuisines_piechart.pdf'
    top_cuisines = 5
    short_cuisine_values = cuisine_values[0:top_cuisines]
    short_cuisine_values.append(sum(cuisine_values[top_cuisines:]))
    short_cuisine_names = cuisine_clean_names[0:top_cuisines]
    short_cuisine_names.append(u'Others')

    plt.figure(figsize=(7, 7))
    explode = list(np.zeros(top_cuisines)) # explode the last slice ('Others')
    explode.append(0.08)

    wedgeprops={"edgecolor":"k", 'linewidth': 1} # edges properties

    plt.pie(short_cuisine_values, labels=short_cuisine_names, startangle=30,
            autopct='%1.1f%%', explode=explode, wedgeprops=wedgeprops)
    plt.title('Cuisines')
    plt.tight_layout()
    plt.axis('equal')
    plt.savefig(fig_file, bbox_inches='tight', dpi=1200)
    plt.close()

    # ===========
    # INGREDIENTS
    # ===========
    # df['n_ingredients'] = df['ingredients'].str.len()
    #
    # mean_ingredients = df.groupby(['cuisine'])['n_ingredients'].mean()
    # std_ingredients = df.groupby(['cuisine'])['n_ingredients'].std()
    #
    # # string manipulation of cuisine names
    # cuisine_names = []
    #
    # for name in mean_ingredients.index:
    #     name = name.title() # capitalize each word
    #
    #     if name.find('_') > 0: # found an underscore in the string name
    #         name = name.replace('_', ' ') # replace it with a space
    #     cuisine_names.append(name)
    #
    # # mean ingredients barplot
    # fig_file = fig_path + 'mean_ingredients_barplot.pdf'
    # plt.figure(figsize=(10,7))
    # sns.barplot(x=mean_ingredients.values,
    #             xerr=std_ingredients.values,
    #             y=cuisine_names,
    #             edgecolor=(0,0,0),
    #             linewidth=1,
    #             error_kw=dict(ecolor='gray', lw=1, capsize=3, capthick=1))
    # plt.ylabel('Cuisine')
    # plt.xlabel('Mean Ingredients')
    # plt.savefig(fig_file, bbox_inches='tight', dpi=1200)
    # plt.close()
    #
    # # counting ingredients from the entire dataset
    # ingredients_count = ingredients_counter(df['ingredients'])
    #
    # # getting the top ingredients in the whole dataset
    # top_common = 15
    # top_ingredients_names = list(ingredients_count[:top_common].index)
    # top_ingredients_values = list(ingredients_count[:top_common].values)
    #
    # # string manipulation of cuisine names
    # clean_names = []
    # for i, name in enumerate(top_ingredients_names):
    #     name = name.title() # capitalize each word
    #
    #     if name.find('_') > 0: # found an underscore in the string name
    #         name = name.replace('_', ' ') # replace it with a space
    #     clean_names.append(name)
    #
    # # top ingredients barplot
    # fig_file = fig_path + 'top_ingredients_barplot.pdf'
    # plt.figure(figsize=(10,7))
    # sns.barplot(x=top_ingredients_values,
    #             y=clean_names,
    #             edgecolor=(0,0,0),
    #             linewidth=1)
    # plt.ylabel('Ingredients')
    # plt.xlabel('Counts')
    # plt.title('Top %i Most Used Ingredients' % int(top_common))
    # plt.savefig(fig_file, bbox_inches='tight', dpi=1200)
    # plt.close()

    # selecting ingredients for pair of cuisines
