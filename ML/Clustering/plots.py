import pandas  as pd
from matplotlib import pyplot as plt 
import numpy as np
import numpy.typing as npt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler , MinMaxScaler 
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer


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
def hist_plot( dataframe:pd.DataFrame,columns_list:list[str], figsize=(20,10))-> None:

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
        sns.histplot(ax= axes[i],x= col, data= dataframe, kde= True)
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

def corr_plot(df : pd.DataFrame , columns : list[str] ,figsize = (10 , 10)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.set_facecolor('oldlace')
    ax.set_facecolor('oldlace')
    sns.heatmap(df[columns].corr(), annot=True, ax=ax, fmt='.2f',cmap= 'vlag')
    plt.show()

def Clusterability(data:pd.DataFrame):
    #PCA
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(data) 
    Scatter_plot(x_pca[:, 0], x_pca[:, 1], x_label= 'Principal Component 1' , y_label='Principal Component 2' , title='PCA')
    #TSNE
    tsne = TSNE(
    n_components=2,
    max_iter=1000)
    x_tsne = tsne.fit_transform(data)
    Scatter_plot(x_tsne[:, 0], x_tsne[:, 1], x_label= 'TSNE Component 1' , y_label='TSNE Component 2' , title='TSNE')

#Clustrability:

#Elbow method
def fn_calculate_wcss(data_points, k_max):
    ls_k = []
    ls_wcss = []

    for k in range(1, k_max + 1):
        model_kmeans_k = KMeans(
            n_clusters=k,
            max_iter=1000,
            random_state=0
        ).fit(data_points)

        ls_k.append(k)
        ls_wcss.append(model_kmeans_k.inertia_)

    fig , ax = plt.subplots()
    ax.plot(ls_k, ls_wcss, c='r', marker='x')

    fig.set_facecolor('oldlace')
    plt.xticks(range(k_max))
    ax.set_facecolor('oldlace')
    ax.set_title('WCSS per Cluster')
    ax.set_xlabel('Cluster Count')
    ax.set_ylabel('WCSS')
    plt.show()


#Silhouette Score

def fn_calculate_silhouette(data_points, k_max):
    ls_k = []
    ls_sil = []

    for k in range(2, k_max + 1):
        model_kmeans_k = KMeans(
            n_clusters=k,
            max_iter=100,
            random_state=0
        ).fit(data_points)

        ls_k.append(k)
        ls_sil.append(silhouette_score(
            X=data_points,
            labels=model_kmeans_k.labels_,
            metric='euclidean'
        ))

    fig ,ax = plt.subplots()
    ax.plot(ls_k, ls_sil, c='r', marker='x')
    fig.set_facecolor('oldlace')
    plt.xticks(range(k_max))
    ax.set_facecolor('oldlace')
    ax.set_title('Silhouette Value per Cluster')
    ax.set_xlabel('Cluster Count')
    ax.set_ylabel('Silhouette Value')
    plt.show()

#Final Clusters

def Cluster_plot(df:pd.DataFrame , features:list ,cluster_column_name:str ):
    #Data for PCA plot
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(df[features]) 
    Cluster_plot_pca=pd.DataFrame({
        'x':x_pca[:, 0],
        'y':x_pca[:, 1],
        'cluster':df[cluster_column_name]
    })


    Cluster_plot_pca['cluster']=Cluster_plot_pca['cluster'].apply(lambda x : str(x))
    #Data for TSNE plot
    tsne = TSNE(
        n_components=2,
        max_iter=1000)
    x_tsne = tsne.fit_transform(df[features]) 
    Cluster_plot_tsne=pd.DataFrame({
        'x':x_tsne[:, 0],
        'y':x_tsne[:, 1],
        'cluster':df[cluster_column_name]
    })
    Cluster_plot_tsne['cluster']=Cluster_plot_tsne['cluster'].apply(lambda x : str(x))

    #drawing plots
    fig2 =px.scatter(Cluster_plot_pca, x="x", y="y", color="cluster" , title='PCA')
    fig1 = px.scatter(Cluster_plot_tsne, x="x", y="y", color="cluster" , title='TSNE' )
    

    fig1.show()
    fig2.show()