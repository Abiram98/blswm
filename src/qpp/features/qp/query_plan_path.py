import networkx as nx
import dgl
import numpy as np
import torch as th
from qpp.features.nodes.node import FilterNode,  TriplePattern2, TriplePattern3, FilterNode02
from qpp.features.nodes.path_node import PathNode, PathNode2
from typing import Any as FeaturizerBase



def rel_lookup(rel):
    match rel:
        case "SS":
            return 0
        case "SP":
            return 1
        case "SO":
            return 2
        case "PS":
            return 3
        case "PP":
            return 4
        case "PO":
            return 5
        case "OS":
            return 6
        case "OP":
            return 7
        case "OO":
            return 8
        case "filter":
            return 9
        case "OPTIONAL":
            return 10
        case "Optional":
            return 10
        case "SingleTripleOrCatesian":
            return 11
        case _:
            raise Exception("undefined "+ rel)


class QueryGraph:
    """Standalone query graph class for SPARQL query representation."""
    
    max_relations = 13
    
    def __init__(self, query_graph):
        QueryGraph.max_relations = 13
        self.process(query_graph)

    @staticmethod
    def set_max_rels(number=13):
        QueryGraph.max_relations = number

    def process(self, data):
        self.data = data
        self.triples: list[TriplePattern2 | PathNode] = list()
        self.filters: list[FilterNode] = list()

        self.nodes = list()
        self.edges = list()
        self.node2obj = {}
        self.node2type_str = {}  # Track node type as string
        for node in data['nodes']:
            match node['nodeType']:
                case "TP":
                    n = TriplePattern3(node)
                    node_type_str = "TP"
                case "PP":
                    n = PathNode2(node)
                    node_type_str = "PP"
                case "FILTER":
                    n = FilterNode02(node)
                    node_type_str = "FILTER"
                case _:
                    raise Exception('Unknown node type: ' + node['nodeType'])
            nodeid = node['nodeId']
            self.nodes.append(nodeid)
            self.node2obj[nodeid] = n
            self.node2type_str[nodeid] = node_type_str

        self.G = self.networkx()

    def networkx(self):
        G = nx.MultiDiGraph()
        for e in self.data['edges']:
            G.add_edge(e[0], e[1], rel_type=rel_lookup(e[2]))
        G.add_nodes_from(self.nodes)
        return G
    
    def feature(self, featurizer: FeaturizerBase):
        """
        Featurize all nodes in the query graph.
        
        Args:
            featurizer: Featurizer object that converts nodes to feature vectors
        """
        self.node_features = {}
        for n_id in self.nodes:
            features = featurizer.featurize(self.node2obj[n_id])
            # Convert numpy arrays to torch tensors if necessary
            if isinstance(features, np.ndarray):
                features = th.from_numpy(features).float()
            
            # Map node type string to integer: TP=0, PP=1, FILTER=2
            node_type_str = self.node2type_str.get(n_id, "TP")
            node_type_int = {"TP": 0, "PP": 1, "FILTER": 2}.get(node_type_str, 0)
            
            self.node_features[n_id] = {
                "node_features": features,
                "node_type": node_type_int
            }

        nx.set_node_attributes(self.G, self.node_features)

    def to_dgl(self):
        """
        Convert the query graph to a DGL graph.
        Prerequisite: feature() must be called first.
        
        Returns:
            dgl.DGLGraph: Graph with node features, node types, and edge types
        """
        try:
            dgl_graph = dgl.from_networkx(
                self.G, edge_attrs=["rel_type"], node_attrs=["node_features", "node_type"]
            )
        except Exception:
            dgl_graph = dgl.from_networkx(self.G, node_attrs=["node_features", "node_type"])
            dgl_graph.edata["rel_type"] = th.tensor(np.array([]), dtype=th.int64)
        
        dgl_graph = dgl.add_self_loop(dgl_graph)
        return dgl_graph

if __name__ == "__main__":
    query = '''SELECT   ?var1 WHERE { ?var1 <http://www.wikidata.org/prop/direct/P569> ?var2 . FILTER ( ( ( ?var2 >= "1911-00-00T00:00:00Z"^^<http://www.w3.org/2001/XMLSchema#dateTime> ) && ( ?var2 <= "1911-13-31T00:00:00Z"^^<http://www.w3.org/2001/XMLSchema#dateTime> ) ) ) . ?var1 <http://www.wikidata.org/prop/direct/P570> ?var3 . FILTER ( ( ( ?var3 >= "1929-00-00T00:00:00Z"^^<http://www.w3.org/2001/XMLSchema#dateTime> ) && ( ?var3 <= "1929-13-31T00:00:00Z"^^<http://www.w3.org/2001/XMLSchema#dateTime> ) ) ) . OPTIONAL { ?var1 <http://www.w3.org/2000/01/rdf-schema#label> ?var4Label . } OPTIONAL { ?var1 <http://www.w3.org/2000/01/rdf-schema#alias> ?var5 . } FILTER ( ( LCASE ( STR ( ?var4Label ) ) IN ( "string1" ) || LCASE ( STR ( ?var5 ) ) IN ( "string1" ) ) ) . } '''
    ...