import random

import torch

def generate_relation_positive_samples(tar_nodes_id, subgraphs_list, subgraphs_emb):

    pos_samples_id_list = []
    pos_samples_emb_list = []
    for i, target_id in enumerate(tar_nodes_id):
        paths_list = subgraphs_list[i]
        tmp_subgraph_emb_tensor = subgraphs_emb[i]
        tmp_pos_samples_id = []
        tmp_pos_emb_mask = []
        for j, path in enumerate(paths_list):
            head_id = path[0]
            tail_id = path[1]
            if head_id == target_id:
                tmp_pos_samples_id.append(tail_id)
                tmp_pos_emb_mask.append(j + 1)

        pos_samples_id_list.append(tmp_pos_samples_id)

        tmp_mask_tensor = torch.tensor(tmp_pos_emb_mask).cuda()
        tmp_pos_embs = torch.index_select(tmp_subgraph_emb_tensor, 0, tmp_mask_tensor)
        pos_samples_emb_list.append(tmp_pos_embs)

    return pos_samples_id_list, pos_samples_emb_list

def build_entity_type_to_ids_map(schema_graph, all_nodes, subgraphs_emb):
    """

    :param schema_graph:
    :param all_nodes:
    :return:
    """
    entityId2typeIds_map = schema_graph.entityId2typeIds
    # all_nodes_wo_target = [i[1:] for i in all_nodes]
    typeId2entityIds_map = {}
    typeId2entityEmbs_map = {}

    for i, nodes in enumerate(all_nodes):
        for j, node_id in enumerate(nodes):
            if str(node_id) not in entityId2typeIds_map:
                continue
            tmp_type_list = entityId2typeIds_map[str(node_id)]
            tmp_node_emb = subgraphs_emb[i][j]
            for k, type_id in enumerate(tmp_type_list):
                if type_id not in typeId2entityIds_map:
                    typeId2entityIds_map[type_id] = [node_id]
                    typeId2entityEmbs_map[type_id] = [tmp_node_emb]
                else:
                    if node_id not in typeId2entityIds_map[type_id]:
                        typeId2entityIds_map[type_id].append(node_id)
                        typeId2entityEmbs_map[type_id].append(tmp_node_emb)

    return typeId2entityIds_map, typeId2entityEmbs_map

def find_neg_of_pos_total(tmp_tar_id, tmp_pos_id_list, tmp_type_list, tmp_pos_id, tmp_pos_emb, typeId2entityIds_map, typeId2entityEmbs_map):
    """
    :param tmp_tar_id:
    :param tmp_pos_id_list:
    :param tmp_type_list:
    :param typeId2entityIds_map:
    :param typeId2entityEmbs_map:
    :return:
    """
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    tar_neg_ids = [tmp_pos_id]
    tar_neg_embs = tmp_pos_emb.unsqueeze(0)
    for i in range(len(tmp_type_list)):

        tmp_pos_type = tmp_type_list[i]
        tmp_neg_ids = typeId2entityIds_map[tmp_pos_type]
        tmp_neg_embs = typeId2entityEmbs_map[tmp_pos_type]
        res_neg_ids = []
        res_neg_embs = torch.tensor([]).cuda()
        for j in range(len(tmp_neg_ids)):

            if len(res_neg_ids) >= 256:
                res_neg_ids = res_neg_ids[0:256]
                res_neg_embs = res_neg_embs[0:256]
                break

            tmp_neg_id = str(tmp_neg_ids[j])
            tmp_neg_emb = tmp_neg_embs[j].unsqueeze(0)

            if cos(tmp_neg_emb, tmp_pos_emb) >= 0.7:  # 0.99
                continue
            if (tmp_neg_id != tmp_tar_id) and (tmp_neg_id not in tmp_pos_id_list):
                res_neg_ids.append(tmp_neg_id)
                res_neg_embs = torch.cat((res_neg_embs, tmp_neg_emb), 0)
        tar_neg_ids.extend(res_neg_ids)
        # tar_neg_embs.append(res_neg_embs)
        tar_neg_embs = torch.cat((tar_neg_embs, res_neg_embs), 0)

    if len(tar_neg_ids) >= 257:
        tar_neg_ids = tar_neg_ids[0:257]
        tar_neg_embs = tar_neg_embs[0:257]

    return tar_neg_ids, tar_neg_embs

def find_neg_of_pos_random(tmp_tar_id, tmp_pos_id_list, tmp_type_list, tmp_pos_id, tmp_pos_emb, typeId2entityIds_map, typeId2entityEmbs_map):
    """

    :param tmp_tar_id:
    :param tmp_pos_id_list:
    :param tmp_type_list:
    :param typeId2entityIds_map:
    :param typeId2entityEmbs_map:
    :return:
    """
    cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

    tar_neg_ids = [tmp_pos_id]
    tar_neg_embs = tmp_pos_emb.unsqueeze(0)

    i = random.randint(0, len(tmp_type_list) - 1)
    tmp_pos_type = tmp_type_list[i]

    if tmp_pos_type not in typeId2entityIds_map:
        return [], tar_neg_embs
    tmp_neg_ids = typeId2entityIds_map[tmp_pos_type]
    tmp_neg_embs = typeId2entityEmbs_map[tmp_pos_type]
    res_neg_ids = []
    res_neg_embs = torch.tensor([]).cuda()

    for j in range(len(tmp_neg_ids)):

        if len(res_neg_ids) >= 256:
            res_neg_ids = res_neg_ids[0:256]
            res_neg_embs = res_neg_embs[0:256]
            break

        tmp_neg_id = str(tmp_neg_ids[j])
        tmp_neg_emb = tmp_neg_embs[j].unsqueeze(0)

        if cos(tmp_neg_emb, tmp_pos_emb) >= 0.99:
            continue
        if (tmp_neg_id != tmp_tar_id) and (tmp_neg_id not in tmp_pos_id_list):
            res_neg_ids.append(tmp_neg_id)
            res_neg_embs = torch.cat((res_neg_embs, tmp_neg_emb), 0)
    tar_neg_ids.extend(res_neg_ids)
    # tar_neg_embs.append(res_neg_embs)
    tar_neg_embs = torch.cat((tar_neg_embs, res_neg_embs), 0)

    if len(tar_neg_ids) >= 257:
        tar_neg_ids = tar_neg_ids[0:257]
        tar_neg_embs = tar_neg_embs[0:257]

    return tar_neg_ids, tar_neg_embs


def select_relation_negative_samples_random(schema_graph, all_nodes, subgraphs_emb, batch_target_nodes_id, pos_samples_id_list, pos_samples_emb_list):
    """

    :param all_nodes:
    :param pos_samples_id_list: size为(batch_size, N)
    :return:
    """
    entityId2typeIds_map = schema_graph.entityId2typeIds
    typeId2entityIds_map, typeId2entityEmbs_map = build_entity_type_to_ids_map(schema_graph, all_nodes, subgraphs_emb)
    pos_neg_samples_id_list = []
    pos_neg_samples_emb_list = []
    tar_emb_relation_list = []

    for i, tmp_pos_id_list in enumerate(pos_samples_id_list):
        tmp_pos_emb_list = pos_samples_emb_list[i]
        tmp_tar_id = batch_target_nodes_id[i]
        tmp_tar_emb = subgraphs_emb[i][0]

        j = random.randint(0, len(tmp_pos_id_list)-1)
        tmp_pos_id = tmp_pos_id_list[j]
        tmp_pos_emb = tmp_pos_emb_list[j]
        if tmp_pos_id not in entityId2typeIds_map:
            continue
        tmp_type_list = entityId2typeIds_map[tmp_pos_id]
        if len(tmp_type_list) > 4:
            tmp_type_list = tmp_type_list[0:4]

        tmp_pos_neg_ids, tmp_pos_neg_embs = find_neg_of_pos_random(tmp_tar_id, tmp_pos_id_list, tmp_type_list, tmp_pos_id, tmp_pos_emb,
                                                    typeId2entityIds_map, typeId2entityEmbs_map)
        if len(tmp_pos_neg_ids) == 0:
            continue

        pos_neg_samples_id_list.append(tmp_pos_neg_ids)
        pos_neg_samples_emb_list.append(tmp_pos_neg_embs)
        tar_emb_relation_list.append(tmp_tar_emb)

    return tar_emb_relation_list, pos_neg_samples_id_list, pos_neg_samples_emb_list