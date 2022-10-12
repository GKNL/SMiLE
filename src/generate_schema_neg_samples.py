import random

import torch


def get_target_nodes_from_subgraphs(subgraphs_list):
    """

    :param subgraphs_list:
    :return:
    """
    target_nodes = []
    for i, subgraph in enumerate(subgraphs_list):
        tmp_triple = subgraph[0]
        tmp_target_node = tmp_triple[0]
        target_nodes.append(tmp_target_node)
    return target_nodes


def get_target_nodes_embedding(node_embeddings):
    """

    :param node_embeddings:
    :return:
    """
    index_tensor = torch.tensor([0]).cuda()
    filtered_emb = node_embeddings.index_select(1, index_tensor)
    target_emb = torch.squeeze(filtered_emb, 1)
    return target_emb


def check_valid_paths_by_random_tail_type(path_list, schema_list, entityId2typeIds_map, target_type_id):
    """

    :param path_list:
    :param schema_list:
    :param entityId2typeIds_map:
    :return:
    """
    mask_list = []
    corresponding_schema = []
    for i, sub_path in enumerate(path_list):  # path: (h, t, r)
        head_type_id = target_type_id
        tail_node_id = sub_path[1]
        rel_type_id = sub_path[2]
        if tail_node_id not in entityId2typeIds_map:
            continue
        tail_type_ids = entityId2typeIds_map[tail_node_id]

        random_index = random.randint(0, len(tail_type_ids) - 1)
        random_tail_type_id = tail_type_ids[random_index]

        sub_path_type = [head_type_id, rel_type_id, random_tail_type_id]

        if sub_path_type in schema_list:
            mask_list.append(i + 1)
            corresponding_schema.append(sub_path_type)
    mask_tensor = torch.tensor(mask_list).cuda()
    return mask_tensor, corresponding_schema


def check_valid_paths_by_every_tail_type(path_list, schema_list, entityId2typeIds_map, target_type_id):
    """

    :param path_list:
    :param schema_list:
    :param entityId2typeIds_map:
    :return:
    """
    mask_list = []
    corresponding_schema = []
    for i, sub_path in enumerate(path_list):  # path: (h, t, r)
        head_type_id = target_type_id
        tail_node_id = sub_path[1]
        rel_type_id = sub_path[2]
        if tail_node_id not in entityId2typeIds_map:
            continue
        tail_type_ids = entityId2typeIds_map[tail_node_id]
        for k in range(len(tail_type_ids)):
            random_tail_type_id = tail_type_ids[k]
            sub_path_type = [head_type_id, rel_type_id, random_tail_type_id]
            if sub_path_type in schema_list:
                mask_list.append(i + 1)
                corresponding_schema.append(sub_path_type)
                break
    mask_tensor = torch.tensor(mask_list).cuda()
    return mask_tensor, corresponding_schema


def generate_schema_instances(schema_graph, tar_nodes_id, subgraphs_list, subgraphs_emb, args):
    """

    :param schema_graph:
    :param tar_nodes_id:
    :param subgraphs_list:
    :param subgraphs_emb:
    :return:
    """
    entityId2typeIds_map = schema_graph.entityId2typeIds
    schema_dict = schema_graph.schema_dict

    batch_sch_instances_emb_list = []
    batch_schema_id_list = []

    for i, target_id in enumerate(tar_nodes_id):
        paths_list = subgraphs_list[i]
        tmp_subgraph_emb_tensor = subgraphs_emb[i]
        if target_id not in entityId2typeIds_map:
            target_node_type_ids = []
        else:
            target_node_type_ids = entityId2typeIds_map[target_id]

        target_sch_instances_emb_tensor = torch.tensor([]).cuda()
        target_schema_id_list = []
        for j, target_type_id in enumerate(target_node_type_ids):
            valid_index_tensor = torch.tensor([]).cuda()
            if target_type_id in schema_dict:
                r_t_list = schema_dict[target_type_id]
                hrt_list = [[target_type_id, r, t] for (r, t) in r_t_list]
                # valid_index_mask = check_valid_paths(paths_list, hrt_list)
                # tmp_sch_instances = [sub_list[2] for sub_list in intersection_list]
                valid_index_tensor, corresponding_schema = check_valid_paths_by_every_tail_type(paths_list, hrt_list,
                                                                                                entityId2typeIds_map,
                                                                                                target_type_id)
            if valid_index_tensor.shape[0] > target_sch_instances_emb_tensor.shape[0]:

                target_sch_instances_emb_tensor = torch.index_select(tmp_subgraph_emb_tensor, 0, valid_index_tensor)
                target_schema_id_list = corresponding_schema

        batch_schema_id_list.append(target_schema_id_list)

        if len(target_schema_id_list) == 0:
            target_sch_instances_emb_tensor = torch.zeros((1, args.base_embedding_dim)).cuda()
        batch_sch_instances_emb_list.append(target_sch_instances_emb_tensor)
        # if i == 0:
        #     batch_sch_instances_emb = target_sch_instances_emb_tensor.unsqueeze(0)
        # else:
        #     batch_sch_instances_emb = torch.cat((batch_sch_instances_emb, target_sch_instances_emb_tensor.unsqueeze(0)), 0)

    return batch_sch_instances_emb_list, batch_schema_id_list


def select_schema_negative_samples(batch_target_nodes_id,
                                   batch_sch_instances_emb_list, batch_schema_id_list,
                                   previous_samples_emb_queue, previous_samples_schema_id_queue, args):

    intra_schema_neg_samples_emb = []
    inter_schema_neg_samples_emb = []

    for i, target_id in enumerate(batch_target_nodes_id):

        tmp_intra_samples = torch.tensor([]).cuda()
        tmp_inter_samples = torch.tensor([]).cuda()

        tar_schema = batch_schema_id_list[i]
        if len(tar_schema) == 0:
            intra_schema_neg_samples_emb.append(tmp_intra_samples)
            inter_schema_neg_samples_emb.append(tmp_inter_samples)
            continue
        target_type_id = tar_schema[0][0]

        for j, oppo_target_id in enumerate(batch_target_nodes_id):
            if len(batch_schema_id_list[j]) == 0:
                continue
            oppo_target_type_id = batch_schema_id_list[j][0][0]
            oppo_instances_emb = batch_sch_instances_emb_list[j]
            if (j != i) and (target_type_id == oppo_target_type_id):
                if sorted(batch_schema_id_list[j]) == sorted(tar_schema):  # intra schema
                    tmp_intra_samples = torch.cat((tmp_intra_samples, oppo_instances_emb), 0)
                elif sorted(batch_schema_id_list[j]) != sorted(tar_schema):  # inter schema
                    tmp_inter_samples = torch.cat((tmp_inter_samples, oppo_instances_emb), 0)

        previous_target_type_id = [key for key in previous_samples_schema_id_queue]
        if len(previous_samples_schema_id_queue) != 0 and target_type_id in previous_target_type_id:
            # corres_pre_schema_list = previous_samples_schema_id_queue[target_type_id]
            # corres_pre_neg_emb_list = previous_samples_emb_queue[target_type_id]

            pre_batch_schema = [x for x in previous_samples_schema_id_queue[target_type_id].values()]
            corres_pre_schema_list = sum(pre_batch_schema, [])
            pre_neg_emb = [x for x in previous_samples_emb_queue[target_type_id].values()]
            corres_pre_neg_emb_list = sum(pre_neg_emb, [])

            for k, tmp_pre_sch in enumerate(corres_pre_schema_list):
                # tmp_pre_instances_emb = torch.from_numpy(corres_pre_neg_emb_list[k]).cuda()
                tmp_pre_instances_emb = corres_pre_neg_emb_list[k]
                if sorted(tar_schema) == sorted(tmp_pre_sch):  # intra schema
                    tmp_intra_samples = torch.cat((tmp_intra_samples, tmp_pre_instances_emb), 0)
                # elif sorted(tar_schema) != sorted(tmp_pre_sch):  # inter schema
                #     tmp_inter_samples = torch.cat((tmp_inter_samples, tmp_pre_instances_emb), 0)

        max_neg_samples_num = args.max_neg_samples_num
        intra_sample_num = tmp_intra_samples.shape[0]
        inter_sample_num = tmp_inter_samples.shape[0]

        if intra_sample_num > max_neg_samples_num:
            intra_mask_index = torch.LongTensor(random.sample(range(intra_sample_num), max_neg_samples_num)).cuda()
            tmp_intra_samples = torch.index_select(tmp_intra_samples, 0, intra_mask_index)
        if inter_sample_num > max_neg_samples_num:
            inter_mask_index = torch.LongTensor(random.sample(range(inter_sample_num), max_neg_samples_num)).cuda()
            tmp_inter_samples = torch.index_select(tmp_inter_samples, 0, inter_mask_index)

        intra_schema_neg_samples_emb.append(tmp_intra_samples)
        inter_schema_neg_samples_emb.append(tmp_inter_samples)

    return intra_schema_neg_samples_emb, inter_schema_neg_samples_emb

def combine_pos_neg_samples(intra_schema_neg_samples_emb, inter_schema_neg_samples_emb,
                            batch_target_nodes_emb, batch_sch_instances_emb_list):
    """

    :param intra_schema_neg_samples_emb:
    :param inter_schema_neg_samples_emb:
    :param batch_target_nodes_emb:
    :param batch_sch_instances_emb_list:
    :return:
    """
    inter_samples_emb = []
    inter_positive_mask = []
    remain_target_emb_inter = []
    intra_samples_emb = []
    intra_positive_mask = []
    remain_target_emb_intra = []
    for i in range(len(batch_sch_instances_emb_list)):
        tmp_positive_sample_emb = batch_sch_instances_emb_list[i]
        pos_num = tmp_positive_sample_emb.shape[0]

        # Inter negative/positive samples
        tmp_inter_neg_sample_emb = inter_schema_neg_samples_emb[i]
        if min(tmp_inter_neg_sample_emb.shape) != 0:

            remain_target_emb_inter.append(batch_target_nodes_emb[i])

            inter_samples_emb.append(torch.cat((tmp_inter_neg_sample_emb, tmp_positive_sample_emb), 0))

            tmp_inter_positive_mask = torch.zeros(1, tmp_inter_neg_sample_emb.shape[0], dtype=torch.float64).cuda()
            tmp_inter_positive_mask = torch.cat((tmp_inter_positive_mask, torch.ones(1, pos_num, dtype=torch.float64).cuda()), 1)
            inter_positive_mask.append(tmp_inter_positive_mask)

        # Intra negative/positive samples
        tmp_intra_neg_sample_emb = intra_schema_neg_samples_emb[i]
        if min(tmp_intra_neg_sample_emb.shape) != 0:
            remain_target_emb_intra.append(batch_target_nodes_emb[i])
            intra_samples_emb.append(torch.cat((tmp_intra_neg_sample_emb, tmp_positive_sample_emb), 0))
            tmp_intra_positive_mask = torch.zeros(1, tmp_intra_neg_sample_emb.shape[0], dtype=torch.float64).cuda()
            tmp_intra_positive_mask = torch.cat((tmp_intra_positive_mask, torch.ones(1, pos_num, dtype=torch.float64).cuda()), 1)
            intra_positive_mask.append(tmp_intra_positive_mask)

    return remain_target_emb_inter, inter_samples_emb, inter_positive_mask,\
           remain_target_emb_intra, intra_samples_emb, intra_positive_mask

def store_batch_tar_instances_to_queue(batch_sch_instances_emb_list, batch_schema_id_list,
                                       previous_samples_emb_queue, previous_samples_schema_id_queue,
                                       batch_id, store_recent=10):
    """

    :param batch_sch_instances_emb_list:
    :param batch_schema_id_list:
    :param previous_samples_emb_queue:
    :param previous_samples_schema_id_queue:
    :param batch_id:
    :param store_recent:
    :return:
    """
    for i, tmp_schema in enumerate(batch_schema_id_list):
        # tmp_sch_instances_emb = batch_sch_instances_emb_list[i].clone().detach().cpu().numpy()
        tmp_sch_instances_emb = batch_sch_instances_emb_list[i].detach()
        if len(tmp_schema) == 0:
            continue

        target_type_id = tmp_schema[0][0]
        if target_type_id not in previous_samples_schema_id_queue:
            # previous_samples_schema_id_queue[target_type_id] = [tmp_schema]
            # previous_samples_emb_queue[target_type_id] = [tmp_sch_instances_emb]

            previous_samples_schema_id_queue[target_type_id] = {}
            previous_samples_schema_id_queue[target_type_id][batch_id] = [tmp_schema]
            previous_samples_emb_queue[target_type_id] = {}
            previous_samples_emb_queue[target_type_id][batch_id] = [tmp_sch_instances_emb]
        else:
            # previous_samples_schema_id_queue[target_type_id].append(tmp_schema)
            # previous_samples_emb_queue[target_type_id].append(tmp_sch_instances_emb)

            if batch_id not in previous_samples_schema_id_queue[target_type_id]:
                previous_samples_schema_id_queue[target_type_id][batch_id] = [tmp_schema]
                previous_samples_emb_queue[target_type_id][batch_id] = [tmp_sch_instances_emb]
            else:
                previous_samples_schema_id_queue[target_type_id][batch_id].append(tmp_schema)
                previous_samples_emb_queue[target_type_id][batch_id].append(tmp_sch_instances_emb)

    batch_to_remove = batch_id - store_recent
    for type_id, tmp_batch_emb_dict in previous_samples_emb_queue.items():
        tmp_batch_schema_id_dict = previous_samples_schema_id_queue[type_id]

        if batch_to_remove in tmp_batch_emb_dict:
            del tmp_batch_emb_dict[batch_to_remove]
            del tmp_batch_schema_id_dict[batch_to_remove]

    return previous_samples_emb_queue, previous_samples_schema_id_queue
