import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import networkx as nx
from networkx import DiGraph



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

        edges = self.df.loc[:, ['from_user', self.user_col]].drop_duplicates().dropna().values
        graph.add_edges_from(edges)
        self.usr_graph = graph
        return graph
    
    def draw_activities(self, alpha=0.03, labels=None, layout=None, info=None):

        activity_matrix = np.zeros((len(self.stages)+1, len(self.stages)+1))
        
        for _, row in self.df.iterrows():
            if(pd.isna(row['from'])):
                continue
                
            x_from = int(row['from'])
            x_stag = int(row[self.stage_col])
            activity_matrix[x_from, x_stag] += 1

        self.activity_matrix = activity_matrix
        
        graph = DiGraph()
        graph.add_nodes_from(self.act_graph.nodes)
        
        plt.figure(figsize=(16,8))
        plt.title("Граф перехода активностей")
        
        if(info=='nodes'):
            print("Всего {} видов действий".format(len(self.stages)))
        elif(info=='time'):
            print("Среднее время действия:")
            df = self.df.sort_values('datetime')
            df['datetime_shift'] = df.datetime.shift(1)
            for stage in np.unique(df[self.stage_col]):
                df_ = df.loc[df[self.stage_col]==stage]
                mean = (df_.datetime[1:] - df_.datetime_shift[1:]).mean()
                print(mean, ' - ', labels[stage])
        elif(info=='all'):
            print("Всего {} видов действий\n".format(len(self.stages)))
            print("Среднее время действия:")
            df = self.df.sort_values('datetime')
            df['datetime_shift'] = df.datetime.shift(1)
            for stage in np.unique(df[self.stage_col]):
                df_ = df.loc[df[self.stage_col]==stage]
                mean = (df_.datetime[1:] - df_.datetime_shift[1:]).mean()
                print(mean, ' - ', labels[stage])
            info='edges'
        
        
        for edge in self.act_graph.edges:
            graph.add_edge(edge[0], edge[1], weight=1 + alpha*activity_matrix[int(edge[0]), int(edge[1])])
            
        weights = [graph[u][v]['weight'] for u,v in graph.edges]
        
        if layout:
            pos = layout(graph)
        else:
            pos = None
        
        nx.draw(graph, pos=pos, width=weights, with_labels=True, font_size=20, node_size=500, node_shape='d',
                    node_color='c',  edge_color='pink', labels=labels)
        if info=='edges':
            labels = {}
            for edge in self.act_graph.edges:
                labels[edge] = int(activity_matrix[int(edge[0]), int(edge[1])])
            nx.draw_networkx_edge_labels(graph, pos=pos, width=weights, with_labels=True, 
                                        font_size=20, node_size=500, node_shape='d',
                                        node_color='b',  edge_color='c', edge_labels=labels)
        plt.show()
        
    def draw_users(self, alpha=0.03, labels=None, layout=None, info=None):

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
        graph.add_nodes_from(self.usr_graph.nodes)
        
        for edge in self.usr_graph.edges:
            if self.users_matrix[int(edge[0]), int(edge[1])] > 0:
                graph.add_edge(edge[0], edge[1], weight = 1 + alpha*self.users_matrix[int(edge[0]), int(edge[1])])
        
        plt.figure(figsize=(16,8))
        plt.title("Граф социальных связей")

        if(info=='nodes'):
            print("Всего {} пользователей".format(len(self.users)))
        elif(info=='time'):
            print("Среднее время пользователя:")
            df = self.df.sort_values('datetime')
            df['datetime_shift'] = df.datetime.shift(1)
            for user in np.unique(df[self.user_col]):
                df_ = df.loc[df[self.user_col]==user]
                mean = (df_.datetime[1:] - df_.datetime_shift[1:]).mean()
                print(mean, ' - ', labels[user])
                
        elif(info=='all'):
            print("Всего {} пользователей\n".format(len(self.users)))
            print("Среднее время пользователя:")
            df = self.df.sort_values('datetime')
            df['datetime_shift'] = df.datetime.shift(1)
            for user in np.unique(df[self.user_col]):
                df_ = df.loc[df[self.user_col]==user]
                mean = (df_.datetime[1:] - df_.datetime_shift[1:]).mean()
                print(mean, ' - ', labels[user])
            info='edges'
        
        weights = [graph[u][v]['weight'] for u,v in graph.edges]
       
        if layout:
            pos = layout(graph)
        else:
            pos = None
        nx.draw(graph, pos=pos, width=weights, with_labels=True, font_size=12, node_size=500, node_shape='d',
                node_color='pink', edge_color='c', labels=labels)
        if info=='edges':
            lab = {}
            for edge in self.usr_graph.edges:
                if edge[0] == edge[1]:
                    continue
                lab[edge] = str(int(self.users_matrix[int(edge[0]), int(edge[1])])) + '<->' \
                + str(int(self.users_matrix[int(edge[1]), int(edge[0])]))
                
            nx.draw_networkx_edge_labels(graph, pos=pos, width=weights, with_labels=True, 
                                        font_size=12, node_size=500, node_shape='d',
                                        node_color='b',  edge_color='c', edge_labels=lab)
        plt.show()       