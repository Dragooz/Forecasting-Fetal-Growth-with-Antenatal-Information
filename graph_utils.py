import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math

#Graph visulization:
def visualize_cat(cols, df):
    '''
    visulizing categorical data.
    '''

    fig, ax = plt.subplots(math.ceil(len(cols)/2) , 2, figsize=(25,35))

    for index, col in enumerate(cols):
        b = sns.countplot(y=col, data=df, ax=ax[index//2, index%2], 
                      order = df[col].value_counts().index)
        b.set_xlabel('', fontsize = 20.0) # X label
        b.set_ylabel(col, fontsize = 20.0) # Y label
        b.tick_params(labelsize=20)

    fig.tight_layout()
    plt.show()

def visualize_bool(cols, df):
    #visulizing categorical data.
    fig, ax = plt.subplots(math.ceil(len(cols)/2) , 2, figsize=(20,5))

    for index, col in enumerate(cols):
        b = sns.countplot(y=col, data=df, ax=ax[index//2, index%2], 
                      order = df[col].value_counts().index)
        b.set_xlabel('', fontsize = 20.0) # X label
        b.set_ylabel(col, fontsize = 20.0) # Y label
        b.tick_params(labelsize=20)

    fig.tight_layout()
    plt.show()

def visualize_val(cols, df):
    #visulizing value data.
    fig, ax = plt.subplots(math.ceil(len(cols)/2) , 2, figsize=(25,15))

    for index, col in enumerate(cols):
        b = sns.boxplot(x=col, data=df, ax=ax[index//2, index%2])
        b.set_xlabel('value', fontsize = 20.0) # X label
        b.set_title(col)
        b.tick_params(labelsize=20)

    fig.tight_layout()
    plt.show()

def visualize_val_all_df(value_cols, df_22, df_32, df_35, hue=None):
    for i in range(len(section)-1):
        cols = value_cols[section[i]: section[i+1]]
        fig, ax = plt.subplots(len(cols), 3, figsize=(35,35))
        for index, col in enumerate(cols):
            sns.set_palette('flare')
            a = sns.boxplot(x=col, data=df_22, ax=ax[index, 0], hue=hue)
            a.set_xlabel('', fontsize = 20.0) # X label
            a.set_ylabel(col, fontsize=20.0)
            a.tick_params(labelsize=20)

            sns.set_palette("pastel")
            b = sns.boxplot(x=col, data=df_32, ax=ax[index, 1], hue=hue)
            b.set_xlabel('', fontsize = 20.0) # X label
            b.set_ylabel('', fontsize=20.0)
            b.tick_params(labelsize=20)

            sns.set_palette("muted")
            c = sns.boxplot(x=col, data=df_35, ax=ax[index, 2], hue=hue)
            c.set_xlabel('', fontsize = 20.0) # X label
            c.set_ylabel('', fontsize=20.0)
            c.tick_params(labelsize=20)

    fig.tight_layout()
    plt.show()
    
def visualize_cat_all_df(cols, df_22, df_32, df_35, hue=None):
    '''visulizing categorical data'''
    fig, ax = plt.subplots(len(cols) , 3, figsize=(35,25))

    for index, col in enumerate(cat_cols):
        sns.set_palette("pastel")
        b = sns.countplot(y=col, data=df_22, ax=ax[index, 0], hue=hue)
        b.set_xlabel('', fontsize = 20.0) # X label
        b.set_ylabel(col, fontsize = 20.0) # Y label
        b.tick_params(labelsize=20)
        
        sns.set_palette("pastel")
        b = sns.countplot(y=col, data=df_32, ax=ax[index, 1], hue=hue)
        b.set_xlabel('', fontsize = 20.0) # X label
        b.set_ylabel('', fontsize = 20.0) # Y label
        b.tick_params(labelsize=20)
        
        sns.set_palette("pastel")
        b = sns.countplot(y=col, data=df_35, ax=ax[index, 2], hue=hue)
        b.set_xlabel('', fontsize = 20.0) # X label
        b.set_ylabel('', fontsize = 20.0) # Y label
        b.tick_params(labelsize=20)

    fig.tight_layout()
    plt.show()
