import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
from networkx import DiGraph

from matplotlib.patches import FancyArrowPatch, Circle, Ellipse




def draw_network(G,pos,ax,sg=None, labels=None, weight_coeff=1):

    for n in G:
        if labels:
            text = labels[n]
        else:
            text = '^_^'
        c=Ellipse(xy=pos[n],width=0.05 + 0.005*len(text), height=0.05, color='pink')
        ax.add_patch(c)
                  
        G.nodes[n]['patch']=c
        
    for (u,v) in G.edges:
        n1=G.nodes[u]['patch']
        n2=G.nodes[v]['patch']
        
        color='k'
        
        lw = G[u][v]['weight'] * weight_coeff
        
        rad = 0.1 + 1/(0.5 + lw)
        
        e = FancyArrowPatch(n1.center,n2.center,patchA=n1,patchB=n2,
                            arrowstyle='-|>',
                            connectionstyle='arc3,rad=%s'%rad,
                            mutation_scale=15,
                            lw=lw,
                            alpha=0.5,
                            color=color)

        ax.add_patch(e)
        
    for n in G:
        if labels:
            text = labels[n]
        else:
            text = '^_^'
        c=Ellipse(xy=pos[n],width=0.05 + 0.005*len(text), height=0.05, color='pink')
        ax.add_patch(c)
        
        x,y=pos[n]
        ax.text(x-len(text)*0.00275, y, text, fontsize=10)
        
    return e



class Graph():
    def __init__(self, df, id_col, stage_col, user_col):
        self.df = df
        self.id_col = id_col
        self.user_col = user_col
        self.stage_col = stage_col
        self.stages = self.df[self.stage_col].unique()
        self.act_graph = None
        self.usr_graph = None
        
    def create_activities_graph(self):
        stages = self.df[self.stage_col].unique()
        graph = DiGraph() 
        for stage in stages:
            graph.add_node(stage)
        self.df['from'] = self.df.groupby([self.id_col])[self.stage_col].shift(1)

        edges = self.df.loc[:, ['from', self.stage_col]].drop_duplicates().dropna().values

        graph.add_edges_from(edges)
        self.act_graph = graph
        return graph

    def create_users_graph(self):
        users = self.df[self.user_col].unique()
        graph = DiGraph() 
        for user in users:
            graph.add_node(user)
        self.df['from_user'] = self.df.groupby([self.id_col])[self.user_col].shift(1)

        edges = self.df.loc[:, ['from_user', self.user_col]].drop_duplicates().dropna().values.astype(int)
        graph.add_edges_from(edges)
        self.usr_graph = graph
        return graph
    
    def draw_activities(self, alpha=0.03, labels=None, layout=None, info=None, node_size=700, style='fancy'):

        activity_matrix = np.zeros((len(self.stages), len(self.stages)))
        
        for _, row in self.df.iterrows():
            if(pd.isna(row['from'])):
                continue
                
            x_from = int(row['from'])
            x_stag = int(row[self.stage_col])
            activity_matrix[x_from, x_stag] += 1

        self.activity_matrix = activity_matrix
        
        graph = DiGraph()
        
#calculating nodes order
        if layout:
            graph.add_nodes_from(self.act_graph.nodes)
            pos = layout(graph)
        else:
            mat = self.activity_matrix.copy()

            for i in range(mat.shape[0]):
                mat[i,i] = -1
            nodes = []
            nodes.append(np.argmax(mat) // mat.shape[0])
            mat[:, nodes[0]] = -1

            while(True):
                node = np.argmax(mat[nodes[-1],:])
                mat[:,node] = -1
                nodes.append(node)

                if len(nodes) == mat.shape[0]:
                    break 

            graph.add_nodes_from(nodes)
            
            pos = {}
            for i, node in enumerate(nodes[:-2]):
                x = 0.5
                y = 1 - (i + 1) / (len(nodes[:-2]) + 1)
                pos[node] = [x, y]
                
            for i, node in enumerate(nodes[-2:]):
                x = (i + 1) / (2 + 1)
                pos[node] = [x, y]

        for edge in self.act_graph.edges:
            graph.add_edge(edge[0], edge[1], weight=1 + alpha*activity_matrix[int(edge[0]), int(edge[1])])
            
        weights = [graph[u][v]['weight'] for u,v in graph.edges]
            
        
# drawing graph

        
        plt.figure(figsize=(16,16))
        plt.title("Граф перехода активностей")
        if style == 'fancy':

            ax=plt.gca()
            draw_network(graph,pos,ax, labels=labels, weight_coeff=100/self.df.shape[0])
            plt.axis('equal')
            plt.axis('off')

            plt.show()
        else:
            
            nx.draw(graph, pos=pos, width=weights, with_labels=True, font_size=10, 
                    node_size=node_size, node_shape='D',
                    node_color='c',  edge_color='pink', labels=labels)
            if info=='edges':
                labels = {}
                for edge in self.act_graph.edges:
                    labels[edge] = int(activity_matrix[int(edge[0]), int(edge[1])])
                nx.draw_networkx_edge_labels(graph, pos=pos, width=weights, with_labels=True, 
                                            font_size=20, node_size=node_size, 
                                            node_color='b',  edge_color='c', edge_labels=labels)
            plt.show()
        
        
        
    def draw_users(self, alpha=0.03, labels=None, layout=None, info=None, node_size=700, style='fancy'):

        self.users = np.unique(self.df[self.user_col])
        users_matrix = np.zeros((len(self.users), len(self.users)))

        for _, row in self.df.iterrows():
            if pd.isna(row['from_user']):
                continue
            else:
                user = int(row[self.user_col])
                prev = int(row['from_user'])
                
                users_matrix[prev, user] += 1
                
        self.users_matrix = users_matrix
        
        
        
        graph = nx.DiGraph()

# calculating nodes order
        if layout:
            graph.add_nodes_from(self.usr_graph.nodes)
            pos = layout(graph)
        else:
            mat = self.users_matrix.copy()
            for i in range(mat.shape[0]):
                mat[i,i] = -1

            nodes = []
            nodes.append(np.argmax(mat) // mat.shape[0])
            mat[:, nodes[0]] = -1

            while(True):
                node = np.argmax(mat[nodes[-1],:])
                mat[:,node] = -1
                nodes.append(node)

                if len(nodes) == mat.shape[0]:
                    break 

            graph.add_nodes_from(nodes)
            
            pos = {}
            for i, node in enumerate(nodes[:-2]):
                x = 0.5
                y = 1 - (i + 1) / (len(nodes[:-2]) + 1)
                pos[node] = [x, y]
                
            for i, node in enumerate(nodes[-2:]):
                x = (i + 1) / (2 + 1)
                pos[node] = [x, y]
            
            
        
        for edge in self.usr_graph.edges:
            if self.users_matrix[int(edge[0]), int(edge[1])] > 0:
                graph.add_edge(edge[0], edge[1], weight = 1 + alpha*self.users_matrix[edge[0], edge[1]])
                
              
# drawing graph

        
        plt.figure(figsize=(16,16))
        plt.title("Граф социальных связей")
        
        weights = [graph[u][v]['weight'] for u,v in graph.edges]
        
        if style == 'fancy':

            ax=plt.gca()
            draw_network(graph, pos, ax, labels=labels, weight_coeff=100/self.df.shape[0])
            plt.axis('equal')
            plt.axis('off')

            plt.show()
        else:
            nx.draw(graph, pos=pos, width=weights, with_labels=True, font_size=12, 
                    node_size=node_size, node_shape="D",
                    node_color='pink', edge_color='c', labels=labels)
            if info=='edges':
                lab = {}
                for edge in self.usr_graph.edges:
                    if edge[0] == edge[1]:
                        continue
                    lab[edge] = str(int(self.users_matrix[int(edge[0]), int(edge[1])])) + '<->' \
                    + str(int(self.users_matrix[int(edge[1]), int(edge[0])]))

                nx.draw_networkx_edge_labels(graph, pos=pos, width=weights, with_labels=True, 
                                            font_size=12, node_size=500,
                                            node_color='b',  edge_color='c', edge_labels=lab)
            plt.show()   