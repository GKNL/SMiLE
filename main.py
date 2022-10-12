import argparse
import json
import os
import time
import random
from typing import List

import numpy as np
import torch

from src.finetuning_BLP import (
    run_finetuning_wkfl2,
    run_finetuning_wkfl3,
    setup_finetuning_input,
)
from src.pretraining_GCL_sch_only import run_pretraining, setup_pretraining_input
from src.processing.context_generator import ContextGenerator
from src.processing.generic_attributed_graph import GenericGraph
from src.processing.generic_schema_graph import GenericSchemaGraph
from src.utils.data_utils import load_pretrained_node2vec, load_pretrained_node2vec_without_idmap
from src.utils.evaluation import run_evaluation_main
from src.utils.link_predict import find_optimal_cutoff, link_prediction_eval
from src.utils.utils import get_name_id_map_from_txt, load_pickle


def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_set_embeddings_details(args):
    if not args.pretrained_embeddings:
        if args.pretrained_method == "compgcn":
            args.pretrained_embeddings = (
                f"{args.emb_dir}/{args.data_name}/"
                f"act_{args.data_name}_{args.node_edge_composition_func}_500.out"
            )
        elif args.pretrained_method == "node2vec":
            args.base_embedding_dim = 128
            args.pretrained_embeddings = (
                f"{args.emb_dir}/{args.data_name}/{args.data_name}.emd"
            )
    else:
        if args.pretrained_method == "compgcn":
            args.base_embedding_dim = 200
        elif args.pretrained_method == "node2vec":
            args.base_embedding_dim = 128
    return args.pretrained_embeddings, args.base_embedding_dim


def get_graph(data_path, false_edge_gen):
    """

    :param data_path:
    :param false_edge_gen: false edge generation pattern/double/basic
    :return:
    """
    print("\n Loading graph...")
    attr_graph = GenericGraph(data_path, false_edge_gen)
    context_gen = ContextGenerator(attr_graph, args.num_walks_per_node)

    return attr_graph, context_gen

def get_schema_graph(data_path, rel_name2id_map, ent_name2id_map, topNfilters=-10):
    """

    :param data_path:
    :return:
    """
    print("\n Loading schema graph...")
    schema_graph = GenericSchemaGraph(data_path, rel_name2id_map, ent_name2id_map, topNfilters)

    return schema_graph

def get_test_edges(paths: List[str], sep: str):
    # edges = set()
    edges = []
    for path in paths:
        with open(path, "r") as f:
            for line in f:
                tokens = line.strip().split(sep)
                etype = tokens[0]
                source = tokens[1]
                destination = tokens[2]
                label = tokens[3]
                edge = (etype, source, destination, label)
                # edges.add(edge)
                edges.append(edge)

    return edges


def main(args):
    gpu_id = args.gpu_id
    torch.cuda.set_device(gpu_id)
    torch.set_num_threads(7)
    setup_seed(20)

    data_path = f"{args.data_path}/{args.data_name}"
    attr_graph, context_gen = get_graph(data_path, args.false_edge_gen)
    attr_graph.dump_stats()

    stime = time.time()
    id_maps_dir = data_path
    ent_name2id = get_name_id_map_from_txt(f"{id_maps_dir}/entityname2id.txt")
    rel_name2id = get_name_id_map_from_txt(f"{id_maps_dir}/relationname2id.txt")
    print(len(ent_name2id), len(rel_name2id))
    num_total_nodes = len(ent_name2id)
    """ Load schema """
    schema_graph = get_schema_graph(data_path, rel_name2id, ent_name2id, args.topNfilters)

    if args.pretrained_method == "compgcn":
        pretrained_node_embedding = load_pickle(args.pretrained_embeddings)
    elif args.pretrained_method == "node2vec":
        # pretrained_node_embedding = load_pretrained_node2vec(
        #     args.pretrained_embeddings, ent2id, args.base_embedding_dim
        # )
        pretrained_node_embedding = load_pretrained_node2vec_without_idmap(
            args.pretrained_embeddings, args.base_embedding_dim, num_total_nodes
        )

    print(
        "No. of nodes with pretrained embedding: ",
        len(pretrained_node_embedding),
    )

    valid_path = data_path + "/valid.txt"
    valid_edges_paths = [valid_path]
    valid_edges = list(get_test_edges(valid_edges_paths, " "))
    test_path = data_path + "/test.txt"
    test_edges_paths = [test_path]
    test_edges = list(get_test_edges(test_edges_paths, " "))
    print("No. edges in test data: ", len(test_edges))

    # print("***************PRETRAINING***************")
    pre_num_batches = setup_pretraining_input(args, attr_graph, context_gen, data_path)
    print("\n Run model for pre-training ...")

    run_pretraining(
        args, attr_graph, pre_num_batches, pretrained_node_embedding, schema_graph
    )

    print("\n Begin evaluation for node prediction...")

    ft_num_batches = setup_finetuning_input(args, attr_graph, context_gen)
    pred_data, true_data = run_finetuning_wkfl2(
        args, attr_graph, ft_num_batches, pretrained_node_embedding,  # ent2id, rel2id
    )

    # print("\n Begin evaluation for link prediction of pretrained nodes...")
    # valid_true_data = np.array(true_data["valid"])
    # threshold = find_optimal_cutoff(valid_true_data, pred_data["valid"])[0]
    # run_evaluation_main(
    #     test_edges, pred_data["test"], true_data["test"], threshold, header="workflow2"
    # )



    # WORKFLOW_3
    print("***************FINETUNING***************")

    print("\n Run model for finetuning (freeze SMiLE weights)...")
    (
        pred_data_test,
        true_data_test,
        pred_data_valid,
        true_data_valid,
    ) = run_finetuning_wkfl3(
        args, attr_graph, ft_num_batches, pretrained_node_embedding  # , ent2id, rel2id
    )

    print("\n Begin evaluation for link prediction...")
    valid_true_data = np.array(true_data_valid)
    threshold = find_optimal_cutoff(valid_true_data, pred_data_valid)[0]
    # save the threshold values for later use
    json.dump(threshold, open(args.outdir + "/thresholds.json", "w"))
    run_evaluation_main(
        test_edges, pred_data_test, true_data_test, threshold, header="workflow3"
    )

    # evaluate after context inference

    etime = time.time()
    elapsed = etime - stime
    print(f"running time(seconds) on {args.data_name} data: {elapsed}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=3, help='GPU ID')
    parser.add_argument("--data_name", default="amazon_s", help="name of the dataset")
    parser.add_argument("--data_path", default="data", help="path to dataset")
    parser.add_argument("--outdir", default="output/default", help="path to output dir")
    parser.add_argument(
        "--pretrained_embeddings", help="absolute path to pretrained embeddings"
    )
    parser.add_argument(
        "--pretrained_method", default="node2vec", help="compgcn|node2vec"
    )

    parser.add_argument('--relation_weight', type=float, default=1.0)
    # schema options
    parser.add_argument('--use_schema', default=False, action='store_true')
    parser.add_argument('--schema_weight', type=float, default=1.0)
    parser.add_argument('--topNfilters', type=int, default=-400)
    parser.add_argument('--max_neg_samples_num', type=int, default=512)

    # Walks options
    parser.add_argument(
        "--beam_width",
        default=2,
        type=int,
        help="beam width used for generating random walks",
    )
    parser.add_argument(
        "--num_walks_per_node", default=1, type=int, help="walks per node"
    )
    parser.add_argument("--walk_type", default="bfs", help="walk type bfs/dfs")
    parser.add_argument("--max_length", default=6, type=int, help="max length of walks")
    parser.add_argument(
        "--n_pred", default=1, help="number of tokens masked to be predicted"
    )
    parser.add_argument(
        "--max_pred", default=1, help="max number of tokens masked to be predicted"
    )

    # Pretraining options
    parser.add_argument("--lr", default=0.0001, type=float, help="learning rate")
    parser.add_argument(
        "--n_epochs", default=10, type=int, help="number of epochs for training"
    )
    parser.add_argument(
        "--checkpoint", default=10, type=int, help="checkpoint for validation"
    )
    parser.add_argument(
        "--base_embedding_dim",
        default=128,
        type=int,
        help="dimension of base embedding",
    )
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="number of data sample in each batch",
    )
    parser.add_argument(
        "--emb_dir",
        default="data",
        type=str,
        help="Used to generate embeddings path if --pretrained_embeddings is not set",
    )
    parser.add_argument(
        "--get_bert_encoder_embeddings",
        default=False,
        help="indicate if need to get node vectors from BERT encoder output, save code "
             "commented out in src/pretraining.py",
    )
    # BERT Layer options
    parser.add_argument(
        "--n_layers", default=4, type=int, help="number of encoder layers in bert"
    )
    parser.add_argument(
        "--d_model", default=200, type=int, help="embedding size in bert"
    )
    parser.add_argument("--d_k", default=64, type=int, help="dimension of K(=Q), V")
    parser.add_argument("--d_v", default=64, type=int, help="dimension of K(=Q), V")
    parser.add_argument("--n_heads", default=4, type=int, help="number of head in bert")
    parser.add_argument(
        "--d_ff",
        default=200 * 4,
        type=int,
        help="4*d_model, FeedForward dimension in bert",
    )
    # GCN Layer options
    parser.add_argument(
        "--is_pre_trained",
        action="store_true",
        help="if there is pretrained node embeddings",
    )
    parser.add_argument(
        "--gcn_option",
        default="no_gcn",
        help="preprocess bert input once or alternate gcn and bert, preprocess|alternate|no_gcn",
    )
    parser.add_argument(
        "--num_gcn_layers", default=2, type=int, help="number of gcn layers before bert"
    )
    parser.add_argument(
        "--node_edge_composition_func",
        default="mult",
        help="options for node and edge compostion, sub|circ_conv|mult|no_rel",
    )

    # Finetuning options
    parser.add_argument("--ft_lr", default=0.001, type=float, help="learning rate")
    parser.add_argument(
        "--ft_batch_size",
        default=128,
        type=int,
        help="number of data sample in each batch",
    )
    parser.add_argument(
        "--ft_checkpoint", default=500, type=int, help="checkpoint for validation"
    )
    parser.add_argument(
        "--ft_d_ff", default=512, type=int, help="feedforward dimension in finetuning"
    )
    parser.add_argument(
        "--ft_layer", default="ffn", help="options for finetune layer: linear|ffn"
    )
    parser.add_argument(
        "--ft_drop_rate", default=0.1, type=float, help="dropout rate in finetuning"
    )
    parser.add_argument(
        "--ft_input_option",
        default="last4_cat",
        help="which output layer from graphbert will be used for finetuning, last|last4_cat|last4_sum",
    )
    parser.add_argument(
        "--false_edge_gen",
        default="double",
        help="false edge generation pattern/double/basic",
    )
    parser.add_argument(
        "--ft_n_epochs", default=10, type=int, help="number of epochs for training"
    )
    parser.add_argument(
        "--path_option",
        default="shortest",
        help="fine tuning context generation: shortest/all/pattern/random",
    )
    args = parser.parse_args()

    # Default values
    args.pretrained_embeddings, args.base_embedding_dim = get_set_embeddings_details(
        args
    )
    args.d_model = args.base_embedding_dim
    args.d_ff = args.base_embedding_dim * 4  # FeedForward dimension in bert

    print("Args ", str(args))
    main(args)
