import random

import pandas as pd
from src.processing.attributed_graph import AttributedGraph


class GenericGraph(AttributedGraph):
    def __init__(
        self, main_dir, false_edge_gen, attributes_file="", sample_training_set=False
    ):
        """
        Assumes derived classes will create a networkx graph object along with
        following
        1) self.unique_relations = set()
        2) self.node_attr_dfs = dict()
        3) self.node_types = dict()
        4) self.G  = nx.Graph()
        """
        super().__init__()
        self.main_dir = main_dir
        self.false_edge_gen = false_edge_gen  # false edge generation pattern/double/basic
        # self.load_graph(main_dir + "/kg_final.txt")
        self.load_graph(main_dir + "/train.txt")
        self.load_graph(main_dir + "/valid.txt")
        self.load_graph(main_dir + "/test.txt")
        self.create_normalized_node_id_map()

        # get product attributes
        self.node_attr = dict()
        self.node_attr_list = []

        if attributes_file == "":
            for node_id in self.G.nodes():
                self.node_attr[node_id] = dict()
                self.node_attr[node_id]["defattr"] = "True"
            self.node_attr_list = ["defattr"]
        else:
            raise NotImplementedError

        self.node_attr_dfs = {
            "deftype": pd.DataFrame.from_dict(self.node_attr, orient="index")
        }
        self.sample_training_set = sample_training_set
        self.get_nodeid2rowid()
        self.get_rowid2nodeid()
        self.get_rowid2vocabid_map()
        self.set_relation_id()

    def generate_link_prediction_dataset(self, outdir, fr_valid_edges, fr_test_edges):

        print("In GenericGraph link Prediction Generation")
        false_edge_gen = self.false_edge_gen
        self.train_edges = self.load_test_edges(self.main_dir + "/train.txt")
        num_positive_edges = len(self.train_edges)
        if self.sample_training_set:
            self.train_edges = random.sample(self.train_edges, num_positive_edges / 2)
            num_positive_edges = len(self.train_edges)
        if false_edge_gen == "pattern":
            print("Generating false edges by pattern")
            self.train_edges = self.generate_false_edges3(self.train_edges)
        elif false_edge_gen == "double":
            print("Generating false edges by random false target and source selection")
            self.train_edges = self.generate_false_edges2(self.train_edges)
        else:
            print("Generating false edges by random false target")
            self.train_edges = self.generate_false_edges(self.train_edges)

        print(
            "Positive training edge, total number of training edges",
            num_positive_edges,
            len(self.train_edges),
        )
        # json.dump(self.train_edges, open("train_edges.json", "w"))
        self.valid_edges = self.load_test_edges(self.main_dir + "/valid.txt")
        self.test_edges = self.load_test_edges(self.main_dir + "/test.txt")  # test.txt
        return self.train_edges, self.valid_edges, self.test_edges

    def generate_link_prediction_dataset_fixed_train(self, outdir, fr_valid_edges, fr_test_edges):

        print("In GenericGraph link Prediction Generation")
        print("Load train_{}.txt".format(self.false_edge_gen))
        false_edge_gen = self.false_edge_gen
        pos_train_edges = self.load_test_edges(self.main_dir + "/train.txt")
        num_positive_edges = len(pos_train_edges)
        if false_edge_gen == "pattern":
            print("Generating false edges by pattern")
            self.train_edges = self.load_test_edges(self.main_dir + "/train_pattern.txt")
        elif false_edge_gen == "double":
            print("Generating false edges by random false target and source selection")
            self.train_edges = self.load_test_edges(self.main_dir + "/train_double.txt")
        else:
            print("Generating false edges by random false target")
            self.train_edges = self.load_test_edges(self.main_dir + "/train_random.txt")

        print(
            "Positive training edge, total number of training edges",
            num_positive_edges,
            len(self.train_edges),
        )
        # json.dump(self.train_edges, open("train_edges.json", "w"))
        self.valid_edges = self.load_test_edges(self.main_dir + "/valid.txt")
        self.test_edges = self.load_test_edges(self.main_dir + "/test.txt")
        return self.train_edges, self.valid_edges, self.test_edges

    def load_graph(self, filename):
        """
        This method should be implemented by all DataLoader classes to fill in
        values for
        1) self.G
        2) self.node_attr_dfs
        3) self.unique_relations
        4) self.node_types
        """
        with open(filename, "r") as f:
            for line in f:
                arr = line.strip().split(" ")
                if len(arr) >= 3:
                    edge_type = arr[0]
                    source = arr[1]
                    target = arr[2]
                    if len(arr) == 3 or (len(arr) == 4 and arr[3] == "1"):
                        self.G.add_edge(source, target, label=edge_type)
                        self.unique_relations.add(edge_type)
                        self.node_types[source] = "deftype"
                        self.node_types[target] = "deftype"
        return

    """
    def load_node_attributes(self, filename):
        node_attr_list = []
        with open(filename, "r") as f:
            line = f.readline()
            for line in f:
                arr = line.strip().split()
                graph_node_id = arr[0]
                node_attr = [float(x) for x in arr[1:]]
                for i in range(len(node_attr)):
                    self.node_attr[graph_node_id] = dict()
                    node_attr_key = "attr_"+str(i)
                    self.node_attr[graph_node_id][node_attr_key] = node_attr[i]
                    node_attr_list.append(node_attr_key)
        return self.node_attr, list(set(node_attr_list))
    """

    def load_test_edges(self, filename):
        true_and_false_edges = []
        with open(filename, "r") as f:
            for line in f:
                arr = line.strip().split()
                relation = arr[0]
                source = arr[1]
                target = arr[2]
                if len(arr) == 4:
                    true_and_false_edges.append((relation, source, target, arr[3]))
                else:
                    # GATNE dataset have no false edges for training
                    true_and_false_edges.append((relation, source, target, "1"))
        return true_and_false_edges

    def generate_false_edges_for_val_test_dataset(self):
        print("In GenericGraph link Prediction Generation (Valid & Test)")
        false_edge_gen = self.false_edge_gen
        self.valid_edges = self.load_test_edges(self.main_dir + "/valid_no_label.txt")
        self.test_edges = self.load_test_edges(self.main_dir + "/test_no_label.txt")
        num_positive_valid_edges = len(self.valid_edges)
        num_positive_test_edges = len(self.test_edges)
        if self.sample_training_set:
            self.valid_edges = random.sample(self.valid_edges, num_positive_valid_edges / 2)
            self.test_edges = random.sample(self.test_edges, num_positive_test_edges / 2)
            num_positive_valid_edges = len(self.valid_edges)
            num_positive_test_edges = len(self.test_edges)

        if false_edge_gen == "pattern":
            print("Generating false edges by pattern")
            self.valid_edges = self.generate_false_edges3(self.valid_edges)
            self.test_edges = self.generate_false_edges3(self.test_edges)
        elif false_edge_gen == "double":
            print("Generating false edges by random false target and source selection")
            self.valid_edges = self.generate_false_edges2(self.valid_edges)
            self.test_edges = self.generate_false_edges2(self.test_edges)
        else:
            print("Generating false edges by random false target")
            self.valid_edges = self.generate_false_edges(self.valid_edges)
            self.test_edges = self.generate_false_edges(self.test_edges)

        print(
            "Positive valid edge, total number of valid edges",
            num_positive_valid_edges,
            len(self.valid_edges),
        )
        print(
            "Positive test edge, total number of test edges",
            num_positive_test_edges,
            len(self.test_edges),
        )
        # json.dump(self.train_edges, open("train_edges.json", "w"))

        return self.valid_edges, self.test_edges

    def generate_false_edges_for_train_dataset(self):

        print("In GenericGraph link Prediction Generation (Train)")
        false_edge_gen = self.false_edge_gen
        self.train_edges = self.load_test_edges(self.main_dir + "/train.txt")
        num_positive_train_edges = len(self.train_edges)

        if false_edge_gen == "pattern":
            print("Generating false edges by pattern")
            self.train_edges = self.generate_false_edges3(self.train_edges)
        elif false_edge_gen == "double":
            print("Generating false edges by random false target and source selection")
            self.train_edges = self.generate_false_edges2(self.train_edges)
        else:
            print("Generating false edges by random false target")
            self.train_edges = self.generate_false_edges(self.train_edges)

        print(
            "Positive train edge, total number of train edges",
            num_positive_train_edges,
            len(self.train_edges),
        )
        # json.dump(self.train_edges, open("train_edges.json", "w"))

        return self.train_edges

    def get_continuous_cols(self):
        return {"deftype": None}

    def get_wide_cols(self):
        return {"deftype": self.node_attr_list}
