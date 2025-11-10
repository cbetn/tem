#graaph visualizer
import networkx as nx
import matplotlib.pyplot as plt


class GraphVisualizer:
    def __init__(
        self,
        edges, 
        pos=None,
        weighted=None,
        solution_edges=None,
        is_solution_directed=False,
    ):
        self.graph = nx.Graph()
        self.weighted = weighted
        self.solution_edges = solution_edges
        self.is_solution_directed = is_solution_directed
        if weighted is None:
            self.graph.add_edges_from(edges)
        else:
            self.graph.add_weighted_edges_from(edges)
        self.pos = (
            pos if pos else nx.spring_layout(self.graph)
        )  # Use provided position or generate new

    def draw_graph(self, title="Graph", save_path=None):
        plt.figure(figsize=(6, 4))
        nx.draw(
            self.graph,
            self.pos,
            with_labels=True,
            node_color="lightblue",
            edge_color="gray",
            node_size=500,
            font_size=10,
        )

        if self.weighted is not None:
            edge_labels = {
                (u, v): d["weight"] for u, v, d in self.graph.edges(data=True)
            }
            nx.draw_networkx_edge_labels(self.graph, self.pos, edge_labels=edge_labels)

        if self.solution_edges:
            nx.draw_networkx_edges(
                self.graph,
                pos=self.pos,
                edgelist=self.solution_edges,
                edge_color="green",
                width=4,
                arrows=self.is_solution_directed,
                arrowstyle="->",
                arrowsize=20,
                connectionstyle="arc3,rad=0.2",
            )

        plt.title(title)

        if save_path:
            plt.savefig(save_path)  # Save the figure if a path is provided
        else:
            plt.show()


#BFS
from collections import deque
from graph_visualizer import GraphVisualizer

import networkx as nx
import matplotlib.pyplot as plt


class Node:
    def __init__(self, state, parent=None):
        self.STATE = state
        self.PARENT = parent

    def __str__(self):
        return f"{self.STATE}"


def GOAL_TEST(node, goal):
    return node.STATE == goal


def CHILD_NODES(node, G):
    children = []
    for neighbor in G.neighbors(node.STATE):
        children.append(Node(state=neighbor, parent=node))
    return children


def SOLUTION(node):
    path = []
    while node:
        path.append(node.STATE)
        node = node.PARENT
    return list(reversed(path))


def BFS(G, start, goal):  # G - Adjancency List, start - vertex. A,B,.. goal - vertex: C

    node = Node(start, parent=None)
    frontier = deque([node])
    explored = set()

    while frontier:
        node = frontier.popleft()
        explored.add(node.STATE)

        if GOAL_TEST(node, goal):
            return SOLUTION(node)

        for child in CHILD_NODES(node, G):
            if (child.STATE not in explored) and all(
                child.STATE != n.STATE for n in frontier
            ):
                frontier.append(child)
    return None


G_adj = {"A": ["B", "D", "E"], "B": ["C", "D"], "D": ["F"]}

G = nx.Graph(G_adj)
pos = nx.spring_layout(G)

solution = BFS(G, start="A", goal="C")
print(solution)


## Ploting the solution
path_edges = list(zip(solution, solution[1:]))
graph = GraphVisualizer(G.edges, pos=pos,solution_edges=path_edges,is_solution_directed=True)
graph.draw_graph()



#DFS
from graph_visualizer import GraphVisualizer
import networkx as nx

# ----- Node class -----
class Node:
    def __init__(self, state, parent=None):
        self.STATE = state
        self.PARENT = parent

    def __str__(self):
        return f"{self.STATE}"

# ----- Goal Test -----
def GOAL_TEST(node, goal):
    return node.STATE == goal

# ----- Generate child nodes -----
def CHILD_NODES(node, G):
    children = []
    for neighbor in G.neighbors(node.STATE):
        children.append(Node(state=neighbor, parent=node))
    return children

# ----- Build solution path -----
def SOLUTION(node):
    path = []
    while node:
        path.append(node.STATE)
        node = node.PARENT
    return list(reversed(path))

# ----- DFS Algorithm -----
def DFS(G, start, goal):
    node = Node(start, parent=None)
    frontier = [node]        # Stack for DFS
    explored = set()

    while frontier:
        node = frontier.pop()        # Remove last (LIFO)
        explored.add(node.STATE)

        if GOAL_TEST(node, goal):
            return SOLUTION(node)

        # Add child nodes in reverse order (so leftmost is expanded first)
        for child in reversed(CHILD_NODES(node, G)):
            if (child.STATE not in explored) and all(child.STATE != n.STATE for n in frontier):
                frontier.append(child)
    return None

# ----- Graph creation -----
G_adj = {"A": ["B", "D", "E"], "B": ["C", "D"], "D": ["F"]}
G = nx.Graph(G_adj)
pos = nx.spring_layout(G)

# ----- Run DFS -----
solution = DFS(G, start="A", goal="C")
print("DFS Path:", solution)

# ----- Plot the solution -----
if solution:
    path_edges = list(zip(solution, solution[1:]))
    graph = GraphVisualizer(G.edges, pos=pos,
                            solution_edges=path_edges,
                            is_solution_directed=True)
    graph.draw_graph(title="DFS Traversal")


#UFS
import heapq
import networkx as nx
from graph_visualizer import GraphVisualizer
import matplotlib.pyplot as plt

class Node:
    def __init__(self, state, parent=None,cost=0):
        self.STATE = state
        self.PARENT = parent
        self.COST=cost

    def __lt__(self,other):
        return self.COST < other.COST


def GOAL_TEST(node, goal):
    return node.STATE == goal


def CHILD_NODES(node, G):
    children = []
    for neighbor in G.neighbors(node.STATE):
        weight=G[node.STATE][neighbor].get("weight",1)
        child=Node(state=neighbor,parent=node,cost=node.COST+weight)
        children.append(child)
    return children


def SOLUTION(node):
    path = []
    while node:
        path.append(node.STATE)
        node = node.PARENT
    return list(reversed(path))

def UCS(G,start,goal):
    node=Node(start,parent=None,cost=0)
    frontier=[(node.COST,node)]
    explored={}
    while frontier:
        _,node=heapq.heappop(frontier)
        if node.STATE in explored and explored[node.STATE]<= node.COST:
            continue
        explored[node.STATE]=node.COST
        if GOAL_TEST(node,goal):
            return SOLUTION(node),node.COST
        for child in CHILD_NODES(node,G):
            heapq.heappush(frontier,(child.COST,child))
    return None,float("inf")
G_adj = [('A','B',2),('A','D',1),('A','E',4),('B','C',5),('B','D',2),('D','F',3)]
G=nx.Graph()
G.add_weighted_edges_from(G_adj)
pos=nx.spring_layout(G)
solution,total_cost=UCS(G,start='A',goal='C')
print(solution)
print(total_cost)

path_edges=list(zip(solution,solution[1:]))
graph= GraphVisualizer(G.edges(data=True),pos=pos,solution_edges=path_edges,is_solution_directed=True)
graph.draw_graph("UCS")

#hill climbing
import networkx as nx
from graph_visualizer import GraphVisualizer

def heuristic(node, goal, pos):
    # Using Euclidean distance as heuristic
    (x1, y1) = pos[node]
    (x2, y2) = pos[goal]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def HILL_CLIMBING(G, start, goal, pos):
    current = start
    path = [current]
    
    while True:
        neighbors = list(G.neighbors(current))
        if not neighbors:
            break

        # Calculate heuristics for all neighbors
        h_values = {n: heuristic(n, goal, pos) for n in neighbors}
        best_neighbor = min(h_values, key=h_values.get)

        # Print heuristic details (optional for understanding)
        print(f"Current: {current}, h={heuristic(current, goal, pos):.2f}")
        print(f"Neighbors: {h_values}")

        # If best neighbor is not better, stop (local minima)
        if heuristic(best_neighbor, goal, pos) >= heuristic(current, goal, pos):
            print("Reached local minimum or plateau.")
            break

        # Move to the better neighbor
        current = best_neighbor
        path.append(current)

        if current == goal:
            print("Goal reached!")
            break

    return path


# Example graph (same style as A*)
edges = [
    ("A", "B", 1),
    ("A", "D", 3),
    ("B", "C", 1),
    ("D", "F", 4),
    ("B", "D", 2),
    ("E", "A", 2),
]

G = nx.Graph()
G.add_weighted_edges_from(edges)
pos = nx.spring_layout(G)

solution = HILL_CLIMBING(G, start="A", goal="C", pos=pos)
print("Path found by Hill Climbing:", solution)

# Visualization
if solution:
    path_edges = list(zip(solution, solution[1:]))
    graph = GraphVisualizer(
        edges,
        pos=pos,
        weighted=True,
        solution_edges=path_edges,
        is_solution_directed=True,
    )
    graph.draw_graph(title="Hill Climbing Path")


#A STAR
import heapq
from graph_visualizer import GraphVisualizer

import networkx as nx
import matplotlib.pyplot as plt


class Node:
    def __init__(self, state, parent=None,g=(),h=()):
        self.STATE = state
        self.PARENT = parent
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self,other):
        return self.f < other.f

    def __str__(self):
        return f"{self.STATE}"

def heuristic(node,goal,pos):
    (x1,y1) = pos[node]
    (x2,y2) = pos[goal]
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5

def GOAL_TEST(node, goal):
    return node.STATE == goal

def SOLUTION(node):
    path = []
    while node:
        path.append(node.STATE)
        node = node.PARENT
    return list(reversed(path))


def A_STAR(G, start, goal,pos):  # G - Adjancency List, start - vertex. A,B,.. goal - vertex: C
    node = Node(start,None,g=0,h = heuristic(start,goal,pos))
    frontier = []
    heapq.heappush(frontier,node)
    explored = set()

    while frontier:
        node = heapq.heappop(frontier)

        if GOAL_TEST(node, goal):
            return SOLUTION(node)

        explored.add(node.STATE)

        for neighbor in G.neighbors(node.STATE):
            weight = G[node.STATE][neighbor].get('weight',1)
            g_cost = node.g + weight
            h_cost = heuristic(neighbor,goal,pos)
            child = Node(neighbor,node,g=g_cost,h=h_cost)
            
            if neighbor not in explored:
                heapq.heappush(frontier,child)
    return None


G_adj = {
    ("A","B",1),
    ("A","D",3),
    ("B","C",1),
    ("D","F",4),
    ("B","D",2),
    ("E","A",2),
}

G = nx.Graph()
G.add_weighted_edges_from(G_adj)
pos = nx.spring_layout(G)

solution = A_STAR(G, start="A", goal="C",pos=pos)
print(solution)


## Ploting the solution
if solution:
    path_edges = list(zip(solution, solution[1:]))
    graph = GraphVisualizer(G_adj, pos=pos,weighted=True,solution_edges=path_edges,is_solution_directed=True,)
    graph.draw_graph()

