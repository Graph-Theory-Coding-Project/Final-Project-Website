from flask import Flask, render_template
import folium
import pandas as pd
from geopy.distance import geodesic
import networkx as nx
from queue import PriorityQueue

app = Flask(__name__)

# Load and preprocess data
df_preprocessing = pd.read_csv("airports-code@public.csv", sep=";")
df_preprocessing1 = df_preprocessing.drop(df_preprocessing.columns[[8, 9]], axis=1)
df_preprocessing1[['latitude', 'longitude']] = df_preprocessing1['coordinates'].str.split(',', expand=True)
df_preprocessing1['latitude'] = pd.to_numeric(df_preprocessing1['latitude'], errors='coerce')
df_preprocessing1['longitude'] = pd.to_numeric(df_preprocessing1['longitude'], errors='coerce')
df = df_preprocessing1.drop(df_preprocessing1.columns[[5, 6]], axis=1)

chosen_airports = [
    'Tokyo Haneda International', 'Soekarno-Hatta International', 'John F Kennedy Intl',
    'Sydney', 'Beverley Springs', 'Ezeiza Ministro Pistarini', 'El Prat De Llobregat',
    'Gimpo International', 'Sheremetyevo'
]

df_chosen_airports = df[df['Airport Name'].isin(chosen_airports)]
airport_locations = list(zip(df_chosen_airports['latitude'], df_chosen_airports['longitude']))
airport_names = df_chosen_airports['Airport Name'].tolist()

# Build the graph
G = nx.Graph()
for i in range(len(airport_locations)):
    G.add_node(airport_names[i], pos=airport_locations[i])

for i in range(len(airport_locations) - 1):
    for j in range(i + 1, len(airport_locations)):
        start_coords = airport_locations[i]
        end_coords = airport_locations[j]
        distance_km = geodesic(start_coords, end_coords).km
        G.add_edge(airport_names[i], airport_names[j], weight=distance_km)

# Prim's algorithm to find MST
def addEdge(node, visited, pq, graph):
    visited[node] = True
    edges = graph.edges(node)
    for neighbor, edge_data in graph[node].items():
        if not visited[neighbor]:
            pq.put((edge_data['weight'], node, neighbor, graph.nodes[node]['pos'], graph.nodes[neighbor]['pos']))

def prims(graph, start):
    visited = {node: False for node in graph.nodes}
    pq = PriorityQueue()
    addEdge(start, visited, pq, graph)
    mst_edges = []
    mst_cost = 0
    while not pq.empty():
        edge, startNode, endNode, startCoords, endCoords = pq.get()
        if visited[endNode]:
            continue
        visited[endNode] = True
        mst_edges.append((startNode, endNode, edge, startCoords, endCoords))
        mst_cost += edge
        addEdge(endNode, visited, pq, graph)
    return mst_edges, mst_cost

mst_edges, _ = prims(G, airport_names[0])

# Construct adjacency list and perform pre-order traversal
def construct_adjacency_list(mst_edges):
    adjacency_list = {}
    for startNode, endNode, edge, _, _ in mst_edges:
        if startNode not in adjacency_list:
            adjacency_list[startNode] = []
        if endNode not in adjacency_list:
            adjacency_list[endNode] = []
        adjacency_list[startNode].append((endNode, edge))
        adjacency_list[endNode].append((startNode, edge))
    return adjacency_list

mst_adj_list = construct_adjacency_list(mst_edges)
visited = {node: False for node in mst_adj_list.keys()}
path = []

def preorderWalk(node, visited, adj_list, path):
    visited[node] = True
    path.append(node)
    for neighbor, _ in adj_list[node]:
        if not visited[neighbor]:
            preorderWalk(neighbor, visited, adj_list, path)

preorderWalk(airport_names[0], visited, mst_adj_list, path)
path.append(airport_names[0])  # Complete Eulerian cycle

# Generate the map
def generate_map():
    m_path = folium.Map(location=[airport_locations[0][0], airport_locations[0][1]], zoom_start=3)
    for i in range(len(path) - 1):
        start_index = airport_names.index(path[i])
        end_index = airport_names.index(path[i + 1])
        start_coords = airport_locations[start_index]
        end_coords = airport_locations[end_index]
        route = [start_coords, end_coords]
        folium.PolyLine(locations=route, color='blue', weight=2.5, opacity=0.8).add_to(m_path)
        folium.Marker(location=start_coords, popup=f"{path[i]}").add_to(m_path)
        if i == len(path) - 2:
            folium.Marker(location=end_coords, popup=f"{path[i+1]}").add_to(m_path)
    m_path.save('static/m_path_map.html')

# Route to display the map
@app.route('/')
def index():
    generate_map()
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
