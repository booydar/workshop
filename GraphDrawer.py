import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
from networkx import DiGraph

from matplotlib.patches import FancyArrowPatch, Circle, Ellipse




def draw_network(G,pos,ax,sg=None, labels=None, node_color='b', weight_coeff=1):

    for n in G:
        if labels:
            text = labels[n]
        else:
            text = 'foo'
        c=Ellipse(xy=pos[n],width=0.1 + 0.005*len(str(text)), height=0.05, color=node_color)
        ax.add_patch(c)
                  
        G.nodes[n]['patch']=c
        
    for (u,v) in G.edges:
        n1=G.nodes[u]['patch']
        n2=G.nodes[v]['patch']
        
        color='k'
        
        lw = G[u][v]['weight'] * weight_coeff - 0.2
        
        rad = 0.1 + 3/(1 + G[u][v]['weight'])
        
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
        c=Ellipse(xy=pos[n],width=0.1 + 0.005*len(text), height=0.05, color=node_color)
        ax.add_patch(c)
        
        x,y=pos[n]
        ax.text(x-len(text)*0.0045, y, text, fontsize=10)
        ax.autoscale()
        
    return e



class Graph():
    def __init__(self, df, id_col, stage_col, user_col, time_col=None):
        self.df = df
        self.id_col = id_col
        self.user_col = user_col
        self.stage_col = stage_col
        self.time_col = time_col
        self.stages = self.df[self.stage_col].unique()
        self.act_graph = None
        self.usr_graph = None
        
    def create_activities_graph(self):
        stages = self.df[self.stage_col].unique()
        graph = DiGraph() 
        for stage in stages:
            graph.add_node(stage)
        self.df['from'] = self.df.groupby([self.id_col])[self.stage_col].shift(1)

        edges = self.df.loc[:, ['from', self.stage_col]].drop_duplicates().dropna().values.astype(int)

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
    
    def draw_activities(self, alpha=0.03, labels=None, layout=None, info=None, node_size=700, style='fancy', main_nodes=4):

        activity_matrix = np.zeros((len(self.stages), len(self.stages)))
        
        for _, row in self.df.iterrows():
            if(pd.isna(row['from'])):
                continue
                
            x_from = int(row['from'])
            x_stag = int(row[self.stage_col])
            activity_matrix[x_from, x_stag] += 1

        self.activity_matrix = activity_matrix
        
        graph = DiGraph()
        
#calculating nodes order and position
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
            
            if main_nodes:
                num_main = main_nodes
            else:
                num_main = len(nodes)*3//5

            pos = {}
            for i, node in enumerate(nodes[:num_main]):
                x = 0.5
                y = 1 - (i + 1) / (len(nodes[:num_main]) + 3)
                pos[node] = [x, y]
                
            y = 1 - (i + 2) / (len(nodes[:num_main]) + 2)
            for i, node in enumerate(nodes[num_main:]):
                x = (i + 1) / (len(nodes) - num_main + 1)
                pos[node] = [x, y]

        for edge in self.act_graph.edges:
            graph.add_edge(edge[0], edge[1], weight=1 + alpha*activity_matrix[int(edge[0]), int(edge[1])])
            
        weights = [graph[u][v]['weight'] for u,v in graph.edges]

#calculating additional information
        if info == 'time' and self.time_col:
                        
            act_time_matrix = np.array([pd.Timedelta(0)]*(len(self.stages)**2)).reshape(len(self.stages), len(self.stages))
            self.df['prev_act_time'] = self.df[self.time_col].shift(1)
            for (_, row) in self.df.iterrows():
                if not pd.isna(row['from']):
                    act_time_matrix[int(row['from']), row[self.stage_col]] += row[self.time_col] - row['prev_act_time']
            for i in range(len(self.stages)):
                for j in range(len(self.stages)):
                    if self.activity_matrix[i,j] != 0:
                        act_time_matrix[i,j] /= self.activity_matrix[i,j]
                        act_time_matrix[i,j] = pd.Timedelta(act_time_matrix[i,j])
            self.act_time_matrix = act_time_matrix
            
        
# drawing graph

        
        plt.figure(figsize=(16,16))
        plt.title("Граф перехода активностей")
        if style == 'fancy':

            ax=plt.gca()
            draw_network(graph,pos,ax, node_color='c', labels=labels, weight_coeff=50/self.df.shape[0])
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
                    
                    if edge[0] == edge[1]:
                        continue
                    labels[edge] = str(int(self.activity_matrix[int(edge[0]), int(edge[1])]))
                nx.draw_networkx_edge_labels(graph, pos=pos, width=weights, with_labels=True, 
                                            font_size=12, node_size=node_size, label_pos=0.25,
                                            node_color='b',  edge_color='c', edge_labels=labels)
            elif info=='time':
                labels = {}
                for edge in self.act_graph.edges:
                    
                    if edge[0] == edge[1]:
                        continue
                    time = self.act_time_matrix[int(edge[0]), int(edge[1])]
                    if time == pd.Timedelta(0):
                        time = ''
                    labels[edge] = str(time)
                    
                nx.draw_networkx_edge_labels(graph, pos=pos, width=weights, with_labels=True, 
                                            font_size=10, node_size=node_size, label_pos=0.25,
                                            node_color='b',  edge_color='c', edge_labels=labels)
                
            plt.show()
        
        
        
    def draw_users(self, alpha=0.03, labels=None, layout=None, info='time', node_size=700, style='fancy', main_nodes=4):

#calculating user interractions
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

# calculating nodes order and position
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
            

            if main_nodes:
                num_main = main_nodes
            else:
                num_main = len(nodes)*3//5
            pos = {}
            for i, node in enumerate(nodes[:num_main]):
                x = 0.5
                y = 1 - (i + 1) / (len(nodes[:num_main]) + 3)
                pos[node] = [x, y]
                
            y = 1 - (i + 2) / (len(nodes[:num_main]) + 2)
            for i, node in enumerate(nodes[num_main:]):
                x = (i + 1) / (len(nodes) - num_main + 1)
                pos[node] = [x, y]
            
            
        
        for edge in self.usr_graph.edges:
            if self.users_matrix[int(edge[0]), int(edge[1])] > 0:
                graph.add_edge(edge[0], edge[1], weight = 1 + alpha*self.users_matrix[edge[0], edge[1]])

#calculating additional information
        if info == 'time' and self.time_col:
            usr_time_matrix = np.array([pd.Timedelta(0)]*(len(self.stages)**2)).reshape(len(self.stages), len(self.stages))
            self.df['prev_user_time'] = self.df[self.time_col].shift(1)
            for (_, row) in self.df.iterrows():
                if not pd.isna(row['from_user']):
                    usr_time_matrix[int(row['from_user']), row[self.user_col]] += row[self.time_col] - row['prev_user_time']
            for i in range(len(self.users)):
                for j in range(len(self.users)):
                    if self.users_matrix[i,j] != 0:
                        usr_time_matrix[i,j] /= self.users_matrix[i,j]
                        usr_time_matrix[i,j] = pd.Timedelta(usr_time_matrix[i,j])
                        
            self.usr_time_matrix = usr_time_matrix

            
                
# drawing graph

        
        plt.figure(figsize=(16,16))
        plt.title("Граф социальных связей")
        
        weights = [graph[u][v]['weight'] for u,v in graph.edges]
        
        if style == 'fancy':

            ax=plt.gca()
            draw_network(graph, pos, ax, labels=labels, node_color='pink', weight_coeff=50/self.df.shape[0])
            plt.axis('equal')
            plt.axis('off')

            plt.show()
        else:
            nx.draw(graph, pos=pos, width=weights, with_labels=True, font_size=12, 
                    node_size=node_size, node_shape="D",
                    node_color='pink', edge_color='c', labels=labels)
            if info=='edges':
                labels = {}
                for edge in self.usr_graph.edges:
                    if edge[0] == edge[1]:
                        continue
                    labels[edge] = str(int(self.users_matrix[int(edge[0]), int(edge[1])]))
                nx.draw_networkx_edge_labels(graph, pos=pos, width=weights, with_labels=True, 
                                            font_size=12, node_size=500, label_pos = 0.25,
                                            node_color='b',  edge_color='c', edge_labels=labels)
                
            elif info=='time':
                labels = {}
                for edge in self.usr_graph.edges:
                    
                    if edge[0] == edge[1]:
                        continue
                    time = self.usr_time_matrix[int(edge[0]), int(edge[1])]
                    if time == pd.Timedelta(0):
                        time = ''
                    labels[edge] = str(time)
                nx.draw_networkx_edge_labels(graph, pos=pos, width=weights, with_labels=True, 
                                            font_size=10, node_size=node_size, label_pos=0.25,
                                            node_color='b',  edge_color='c', edge_labels=labels)
            plt.show()   
            
