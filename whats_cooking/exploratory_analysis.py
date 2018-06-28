from __future__ import division

import sys
import os

import numpy as np
import pandas as pd

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

    # string manipulation of cuisine names
    for i, name in enumerate(cuisine_names):
        capital_name = name.title()
        cuisine_names[i] = capital_name

        if capital_name.find('_') > 0:
            clean_name = capital_name.replace('_', ' ')
            cuisine_names[i] = clean_name

    # cuisines bar plot
    fig_file = fig_path + 'cuisines_barplot.eps'

    plt.figure(figsize=(10, 7))
    sns.barplot(x=cuisine_values,
                y=cuisine_names,
                edgecolor=(0, 0, 0),
                linewidth=1)
    plt.xlabel('Counts')
    plt.ylabel('Cuisines')
    plt.savefig(fig_file, format='eps', bbox_inches='tight', dpi=1200)
    plt.close()

    # cuisines pie chart
    fig_file = fig_path + 'cuisines_piechart.eps'
    top_cuisines = 5
    short_cuisine_values = cuisine_values[0:top_cuisines]
    short_cuisine_values.append(sum(cuisine_values[top_cuisines:]))
    short_cuisine_names = cuisine_names[0:top_cuisines]
    short_cuisine_names.append(u'Others')

    plt.figure(figsize=(7,7))
    explode = list(np.zeros(top_cuisines)) # explode the last slice ('Others')
    explode.append(0.08)

    wedgeprops={"edgecolor":"k", 'linewidth': 1} # edges properties

    plt.pie(short_cuisine_values, labels=short_cuisine_names, startangle=30,
            autopct='%1.1f%%', explode=explode, wedgeprops=wedgeprops)
    plt.title('Cuisines')
    plt.tight_layout()
    plt.axis('equal')
    plt.savefig(fig_file, format='eps', bbox_inches='tight', dpi=1200)
    plt.close()
