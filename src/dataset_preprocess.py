from src.processing.generic_attributed_graph import GenericGraph
import random
import numpy as np
import os

def get_graph(data_path, false_edge_gen):
    """

    :param data_path:
    :param false_edge_gen: false edge generation pattern/double/basic
    :return:
    """
    print("\n Loading graph...")
    attr_graph = GenericGraph(data_path, false_edge_gen)

    return attr_graph


if __name__ == "__main__":
    seed=20
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    data_path = '/SMiLE/data/FB15k-237'
    false_edge_gen = 'basic'  # pattern/double/basic
    attr_graph = get_graph(data_path, false_edge_gen)

    # train_edges = attr_graph.generate_false_edges_for_train_dataset()  # generate_false_edges_for_val_test_dataset
    # train_file_name = data_path + '/train_{}.txt'.format(false_edge_gen)
    # attr_graph.dump_edges(train_file_name, train_edges, label=True)

    valid_edges, test_edges = attr_graph.generate_false_edges_for_val_test_dataset()
    val_file_name = data_path + '/valid.txt'
    test_file_name = data_path + '/test.txt'
    attr_graph.dump_edges(val_file_name, valid_edges, label=True)
    attr_graph.dump_edges(test_file_name, test_edges, label=True)


