import pandas  as pd
from matplotlib import pyplot as plt 
import numpy as np
import numpy.typing as npt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots




def Scatter_plot(x:npt.ArrayLike , y:npt.ArrayLike , x_label:str =None, y_label:str =None , fig_size=(7,5) , title:str=None ,ax_grid=True , s:npt.ArrayLike =20):
    fig, ax = plt.subplots(figsize=fig_size)
    fig.set_facecolor('oldlace')
    ax.set_facecolor('oldlace')
    plt.tick_params(
        axis='both',          
        which='both',      
        bottom=False,      
        top=False,         
        left=False,
    )
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    plt.scatter(x , y , color='tab:blue' , alpha=0.3 , s=s )
    ax.grid(ax_grid)
    plt.show()

def Box_plot(Df:pd.DataFrame , Columns : list , figsize=(20,10)):
    fig , ax = plt.subplots(int(np.ceil(len(Columns)/3)) , 3,figsize=figsize)
    ax = ax.flatten()
    fig.set_facecolor('oldlace')
    plt.tick_params(
        
    )
    for col in Columns : 
        ax[Columns.index(col)].set_facecolor('oldlace')
        ax[Columns.index(col)].set_title(col)
        ax[Columns.index(col)].tick_params(
            axis='both',          
            which='both',      
            bottom=False,      
            top=False,         
            left=False,
            # labelleft=False,
        )
        sns.boxplot(Df[col] , ax=ax[Columns.index(col)] , color='tab:blue' )
def hist_plot( dataframe:pd.DataFrame,columns_list:list[str], figsize=(20,10) , log_scale:bool = True)-> None:

    N= len(columns_list)
    rows = int(np.floor(np.sqrt(N)))
    cols= int(np.ceil(N/rows))
    fig, axes = plt.subplots(rows, cols, figsize = figsize)
    fig.set_facecolor('oldlace')


    if N==1:
        axes = [axes] 
    else:
        axes=axes.flatten()

    for i, col in enumerate (columns_list):
        if log_scale ==False:
            sns.histplot(ax= axes[i],x= col, data= dataframe, kde= True , log_scale=False)
        elif log_scale ==True :
            if dataframe[col].min()>0:
                sns.histplot(ax= axes[i],x= col, data= dataframe, kde= True , log_scale=True)
            elif dataframe[col].min()<=0:
                sns.histplot(ax= axes[i],x= col, data= dataframe, kde= True , log_scale=False)
        axes[i].set_facecolor('oldlace')
        axes[i].set_title(f'{col}')
        axes[i].set_ylabel('')
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='y')
        axes[i].grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
        axes[i].tick_params(axis='both')
        sns.despine(ax=axes[i], top=True, right=True)
        

    fig.suptitle('histograms of Selected Columns', fontweight='bold')

    for j in range(N , len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def corr_plot(df : pd.DataFrame , columns : list[str]  ,figsize = (10 , 10)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor('oldlace')
    ax.set_facecolor('oldlace')
    sns.heatmap(df[columns].corr(), annot=True, ax=ax, fmt='.2f',cmap= 'vlag')
    plt.show()

def Count_plot(df : pd.DataFrame , Columns: list, hue:npt.ArrayLike , figsize =(20,10)):
    fig , axes = plt.subplots(int(np.ceil(len(Columns)/2)) , 2,figsize=figsize)
    fig.set_facecolor('oldlace')
    N= len(Columns)
    axes=axes.flatten()
    for i , Cat in enumerate(Columns) :
        sns.countplot(df,x=Cat, ax=axes[i] , hue=hue )
        axes[i].set_facecolor('oldlace')
        axes[i].set_title(f'{Cat}')
        axes[i].set_ylabel('')
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='y')
        axes[i].grid(axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
        axes[i].tick_params(axis='both')
        sns.despine(ax=axes[i], top=True, right=True)
    fig.suptitle('CountPlot of Selected Categories', fontweight='bold')

    for j in range(N , len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()