import json
import networkx as nx
import matplotlib.pyplot as plt

NODE_SIZE = 5500

def visualize(file_name):
    print("Starting visualization...")
    
    G = nx.DiGraph()

    with open(f"./data/annotations/{file_name}.json", encoding="utf8") as f:

        # Extract data from json
        data = json.load(f)

        # Extract nodes and edges
        sentiments = data["sentiments"] 
        characters, negative_edges, neutral_edges, positive_edges =  [], [], [], []

        for item in sentiments.items():
            character_from = item[0]
            characters.append(character_from)

            for sentiment in item[1].items():
                character_to = sentiment[0]
                score = sentiment[1]

                if character_from != character_to:
                    G.add_edge(character_from, character_to)
                    if score > 0:
                        positive_edges.append((character_from, character_to))
                    elif score < 0:
                        negative_edges.append((character_from, character_to))
                    else:
                        neutral_edges.append((character_from, character_to))

        # Draw DiGraph
        plt.title('Character sentiments visualization')
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size = NODE_SIZE)
        nx.draw_networkx_labels(G, pos)

        nx.draw_networkx_edges(G, 
                               pos, 
                               width=2,
                               edgelist=negative_edges, 
                               edge_color='r',
                               arrows=True, arrowsize=15, node_size=NODE_SIZE,
                               connectionstyle='arc3, rad = 0.05')
        
        nx.draw_networkx_edges(G, 
                               pos, 
                               width=2,
                               edgelist=positive_edges, 
                               edge_color='g', 
                               arrows=True, arrowsize=15, node_size=NODE_SIZE,
                               connectionstyle='arc3, rad = 0.05')
        
        nx.draw_networkx_edges(G, 
                               pos, 
                               edgelist=neutral_edges, 
                               arrows=True, arrowsize=15, node_size=NODE_SIZE,
                               connectionstyle='arc3, rad = 0.05')
        plt.savefig('vis1.png', bbox_inches='tight', dpi=500)

    print("Ending visualization...")