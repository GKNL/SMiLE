import json
import os
import pickle
import shutil
import time
import itertools

import numpy as np
import torch
import torch.nn
import torch.optim as optim

from src.generate_relation_neg_samples import generate_relation_positive_samples, select_relation_negative_samples_random
from src.smile_model.smile_model import SMILE
from src.smile_model.process_walks_gcn import (
    Processing_GCN_Walks,
    get_normalized_masked_ids,
)
from src.utils.data_utils import (
    create_batches,
    split_data,
)
from src.utils.utils import EarlyStopping, load_pickle, show_progress

from src.generate_schema_neg_samples import get_target_nodes_from_subgraphs, get_target_nodes_embedding, \
    generate_schema_instances, select_schema_negative_samples, combine_pos_neg_samples, \
    store_batch_tar_instances_to_queue

from src.contrast import Contrast
from src.contrast_relation import Contrast_relation


def setup_pretraining_input(args, attr_graph, context_gen, data_path):
    """

    :param args:
    :param attr_graph:
    :param context_gen:
    :param data_path:
    :return:
    """
    # split train/validation/test dataset
    tasks = ["train", "validate", "test"]
    split_file = {
        task: os.path.join(data_path, args.data_name + "_pretrain_" + task + ".txt")
        for task in tasks
    }
    if os.path.exists(split_file["train"]):
        print("\n load existing /validate/test data ...")
        for task in tasks:
            fin = open(split_file[task], "r")
            walk_data = []
            for line in fin:
                line = json.loads(line)
                walk_data.append(line)
            fin.close()
            if task == "train":
                walk_train = walk_data
            elif task == "validate":
                walk_validate = walk_data
            else:
                walk_test = walk_data
    else:
        # generate walks(context subgraphs)
        print("\n Generating subgraphs for pre-training ...")

        all_walks = context_gen.get_pretrain_subgraphs(
            data_path,
            args.data_name,
            args.beam_width,
            args.max_length,
            args.walk_type,  # walk type bfs/dfs
        )
        print("\n split data to train/validate/test and save the files ...")

        walk_train, walk_validate, walk_test = split_data(
            all_walks, data_path, args.data_name
        )
    print(len(walk_train), len(walk_validate), len(walk_test))

    # create batches
    num_batches = {}
    for task in tasks:
        if task == "train":
            walk_data = walk_train
        elif task == "validate":
            walk_data = walk_validate
        else:
            walk_data = walk_test
        cnt = create_batches(
            walk_data, data_path, args.data_name, task, args.batch_size
        )
        num_batches[task] = cnt
    print("number of batches for pre-training: ", num_batches)
    return num_batches


def run_pretraining(
    args, attr_graph, no_batches, pretrained_node_embedding, schema_graph
):
    earlystopping = EarlyStopping(patience=3, delta=0.001)
    # data_path = args.data_path +'compgcn_output/' + args.data_name + '/'
    data_path = args.data_path + "/" + args.data_name + "/"
    out_dir = args.outdir + "/"
    try:
        shutil.rmtree(out_dir)
        os.mkdir(out_dir)
    except:
        os.mkdir(out_dir)

    relations = attr_graph.relation_to_id
    #no_relations = attr_graph.get_number_of_relations()
    nodeid2rowid = attr_graph.get_nodeid2rowid()
    #no_nodes = attr_graph.get_number_of_nodes()
    walk_processor = Processing_GCN_Walks(
        nodeid2rowid, relations, args.n_pred, args.max_length, args.max_pred
    )

    print("\n processing walks in minibaches before running model:")
    batch_input_file = os.path.join(data_path, args.data_name + "_batch_input.pickled")
    if os.path.exists(batch_input_file):
        print("loading saved files ...")
        batch_input = load_pickle(batch_input_file)
    else:
        batch_input = {}
        tasks = ["train", "validate", "test"]
        for task in tasks:
            print(task)
            batch_input[task] = {}
            for batch_id in range(no_batches[task]):
                (
                    subgraphs_list,
                    all_nodes,
                    masked_nodes,
                    masked_postion,
                ) = walk_processor.process_minibatch(
                    data_path, args.data_name, task, batch_id
                )

                batch_input[task][batch_id] = [
                    subgraphs_list,
                    all_nodes,
                    masked_nodes,
                    masked_postion,
                ]
        pickle.dump(batch_input, open(batch_input_file, "wb"))

    if args.is_pre_trained:
        pretrained_node_embedding_tensor = pretrained_node_embedding
    else:
        pretrained_node_embedding_tensor = None

    smile = SMILE(
        args.n_layers,  # number of encoder layers in bert
        args.d_model,  # embedding size in bert
        args.d_k,
        args.d_v,
        args.d_ff,
        args.n_heads,
        attr_graph,
        pretrained_node_embedding_tensor,
        args.is_pre_trained,
        args.base_embedding_dim,
        args.max_length,
        args.num_gcn_layers,
        args.node_edge_composition_func,
        args.gcn_option,  # preprocess bert input once or alternate gcn and bert, preprocess|alternate|no_gcn
        args.get_bert_encoder_embeddings,  # indicate if need to get node vectors from BERT encoder output
        # ent2id,
        # rel2id,
    )

    if args.use_schema:
        contrast_schema = Contrast(hidden_dim=args.base_embedding_dim, tau=0.8, lam=0.8
                                   ).cuda()  # lam=0.4
    # contrast_relation = Contrast_relation(hidden_dim=args.base_embedding_dim, tau=0.8).cuda()

    # if torch.cuda.is_available():
    #    smile =  torch.nn.DataParallel(smile, device_ids=[0,1])

    if args.use_schema:
        optimizer = optim.Adam(itertools.chain(smile.parameters(), contrast_schema.parameters()), args.lr)

    node_embeddings = dict()

    loss_dev_final = []
    print("\n Begin Training")
    for epoch in range(args.n_epochs):
        loss_arr = []
        if args.use_schema:
            loss_inter_arr = []
            loss_intra_arr = []
        loss_dev_min = 1e6

        if args.use_schema:
            # {entity type : {batch_id : [schema emb list]} }
            previous_samples_emb_queue = {}
            # {entity type id : {batch_id : [schema id list]} }
            previous_samples_schema_id_queue = {}

        print("\nEpoch: {}".format(epoch))
        start_time = time.time()
        for batch_id in range(no_batches["train"]):
            (
                subgraphs_list,
                all_nodes,
                masked_nodes,
                masked_postion,
            ) = walk_processor.process_minibatch(
                data_path, args.data_name, "train", batch_id
            )

            node_emb, concat_node_emb, out_rel_emb = smile(
                    subgraphs_list, all_nodes  # , masked_postion, masked_nodes
                )

            batch_target_nodes_id = get_target_nodes_from_subgraphs(subgraphs_list)
            # 得到batch_target_nodes的embedding
            batch_target_nodes_emb = get_target_nodes_embedding(node_emb)

            if args.use_schema:
                # print("\n Sampling schema instances(Positive samples)...")
                batch_sch_instances_emb_list, batch_schema_id_list = generate_schema_instances(schema_graph,
                                                                                               batch_target_nodes_id,
                                                                                               subgraphs_list,
                                                                                               node_emb,
                                                                                               args)

                # print("\n Selecting intra & inter negative samples...")
                intra_schema_neg_samples_emb, inter_schema_neg_samples_emb = \
                    select_schema_negative_samples(batch_target_nodes_id,
                                                         batch_sch_instances_emb_list,
                                                         batch_schema_id_list,
                                                         previous_samples_emb_queue,
                                                         previous_samples_schema_id_queue,
                                                         args)

                # if batch_id % 8 == 0:
                #     previous_samples_emb_queue = {}
                #     previous_samples_schema_id_queue = {}

                target_emb_inter, inter_samples_emb, inter_positive_mask, \
                target_emb_intra, intra_samples_emb, intra_positive_mask = combine_pos_neg_samples(intra_schema_neg_samples_emb,
                                                                                                   inter_schema_neg_samples_emb,
                                                                                                   batch_target_nodes_emb,
                                                                                                   batch_sch_instances_emb_list)

                loss_schema, loss_inter_schema, loss_intra_schema = contrast_schema(target_emb_inter,
                                                                                    target_emb_intra,
                                                                                    inter_samples_emb,
                                                                                    intra_samples_emb,
                                                                                    inter_positive_mask,
                                                                                    intra_positive_mask)

                loss = loss_schema

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_arr.append(loss.data.cpu().numpy().tolist())

            if args.use_schema:
                loss_inter_arr.append(loss_inter_schema.data.cpu().numpy().tolist())
                loss_intra_arr.append(loss_intra_schema.data.cpu().numpy().tolist())

                # previous_samples_emb_queue, previous_samples_schema_id_queue = \
                #     store_batch_tar_instances_to_queue(batch_sch_instances_emb_list,
                #                                        batch_schema_id_list,
                #                                        previous_samples_emb_queue,
                #                                        previous_samples_schema_id_queue,
                #                                        batch_id,
                #                                        store_recent=1)

            ccnt = batch_id % args.checkpoint
            if batch_id > 0 and batch_id % args.checkpoint == 0:
                ccnt = args.checkpoint
            message = "Loss: {}, AvgLoss: {}{}".format(
                np.around(loss.data.cpu().numpy().tolist(), 4),
                np.around(np.average(loss_arr).tolist(), 4),
                " " * 10,
            )
            show_progress(ccnt, min(args.checkpoint, no_batches["train"]), message)


            if (
                batch_id % args.checkpoint == 0 and batch_id > 0
            ) or batch_id == no_batches["train"] - 1:
                if args.use_schema:
                    print(
                        "\nBatch: {}, Loss: {}, AvgLoss: {}, Avg_Inter_Loss: {}, Avg_Intra_Loss: {}".format(
                            batch_id,
                            np.around(loss.data.cpu().numpy().tolist(), 4),
                            np.around(np.average(loss_arr).tolist(), 4),
                            np.around(np.average(loss_inter_arr).tolist(), 4),
                            np.around(np.average(loss_intra_arr).tolist(), 4)
                        )
                    )
                else:
                    print(
                        "\nBatch: {}, Loss: {}, AvgLoss: {}".format(
                            batch_id,
                            np.around(loss.data.cpu().numpy().tolist(), 4),
                            np.around(np.average(loss_arr).tolist(), 4)
                        )
                    )

                # validation
                smile.eval()
                # print("\n Begin validation...")

                with torch.no_grad():
                    loss_dev_arr = []
                    val_previous_samples_emb_queue = {}
                    val_previous_samples_schema_id_queue = {}
                    for batch_dev_id in range(no_batches["validate"]):
                        (
                            subgraphs_list,
                            all_nodes,
                            masked_nodes,
                            masked_postion,
                        ) = batch_input["validate"][batch_dev_id]

                        node_emb, concat_node_emb, out_rel_emb = smile(
                            subgraphs_list, all_nodes  # , masked_postion, masked_nodes
                        )

                        batch_target_nodes_id = get_target_nodes_from_subgraphs(subgraphs_list)

                        batch_target_nodes_emb = get_target_nodes_embedding(node_emb)


                        if args.use_schema:
                            # print("\n Sampling schema instances(Positive samples)...")
                            batch_sch_instances_emb_list, batch_schema_id_list = generate_schema_instances(schema_graph,
                                                                                                           batch_target_nodes_id,
                                                                                                           subgraphs_list,
                                                                                                           node_emb,
                                                                                                           args)

                            # print("\n Selecting intra & inter negative samples...")
                            intra_schema_neg_samples_emb, inter_schema_neg_samples_emb = \
                                select_schema_negative_samples(batch_target_nodes_id,
                                                               batch_sch_instances_emb_list,
                                                               batch_schema_id_list,
                                                               val_previous_samples_emb_queue,
                                                               val_previous_samples_schema_id_queue,
                                                               args)
                            # if batch_dev_id % 3 == 0:
                            #     val_previous_samples_emb_queue = {}
                            #     val_previous_samples_schema_id_queue = {}

                            target_emb_inter, inter_samples_emb, inter_positive_mask, \
                            target_emb_intra, intra_samples_emb, intra_positive_mask = combine_pos_neg_samples(
                                intra_schema_neg_samples_emb,
                                inter_schema_neg_samples_emb,
                                batch_target_nodes_emb,
                                batch_sch_instances_emb_list)

                            loss_schema_val, loss_inter_schema, loss_intra_schema = contrast_schema(target_emb_inter,
                                                                                  target_emb_intra,
                                                                                  inter_samples_emb,
                                                                                  intra_samples_emb,
                                                                                  inter_positive_mask,
                                                                                  intra_positive_mask)
                            loss_val = loss_schema_val

                            val_previous_samples_emb_queue, val_previous_samples_schema_id_queue = \
                                store_batch_tar_instances_to_queue(batch_sch_instances_emb_list,
                                                                   batch_schema_id_list,
                                                                   val_previous_samples_emb_queue,
                                                                   val_previous_samples_schema_id_queue,
                                                                   batch_id,
                                                                   store_recent = 2)


                        loss_dev_arr.append(loss_val.data.cpu().numpy().tolist())
                    loss_dev_avg = np.average(loss_dev_arr)

                print(
                    "Validation  -->  MinLoss: {}, CurLoss: {}".format(
                        np.around(loss_dev_min, 4), np.around(loss_dev_avg, 4)
                    )
                )
                if loss_dev_avg < loss_dev_min:
                    loss_dev_min = loss_dev_avg
                    fmodel = open(
                        os.path.join(out_dir, "SMILE_GCL_" + str(epoch) + ".model"), "wb"
                    )
                    torch.save(smile.state_dict(), fmodel)
                    fmodel.close()

                smile.train()
                # if(earlystopping.check_early_stopping(loss_dev_avg) == True):
                #    print("Loss not decreasing over 3 epochs exiting")
                #    epoch = args.n_epochs + 1

        loss_dev_final.append(loss_dev_min)
        print("MinLoss: ", np.around(loss_dev_min, 4))
        end_time = time.time()
        print("epoch time: (s)", (end_time - start_time))

    best_epoch = np.argsort(loss_dev_final)[0]
    print("\nBest Epoch: {}".format(best_epoch))
    fbest = open(os.path.join(out_dir, "best_epoch_id.txt"), "w")
    fbest.write(str(best_epoch) + "\n")
    fbest.close()
    np.save(f"{args.outdir}/pretraining_loss.npy", loss_dev_final)

    return


def get_str_subgraph(subgraph):
    str_subgraph = ""
    for edge in subgraph:
        str_subgraph += str(edge[0]) + "_" + str(edge[1]) + "_" + str(edge[2]) + ";"
    return str_subgraph
