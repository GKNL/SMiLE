import argparse

import torch
import operator
from src.utils.utils import get_name_id_map_from_txt


def build_entity2types_dictionaries(data_path, entity_name2id, type_name2id):

    entityId2typeIds = {}
    typeId2entityIds = {}

    entity2type_file = open(data_path + "/entity2types_ttv.txt", "r")

    for line in entity2type_file:
        splitted_line = line.strip().split("\t")
        entity_name = splitted_line[0]
        entity_type = splitted_line[1]
        entity_id = entity_name2id[entity_name]
        type_id = type_name2id[entity_type]

        # if entity_name not in entityName2entityTypes:
        #     entityName2entityTypes[entity_name] = []
        # if entity_type not in entityName2entityTypes[entity_name]:
        #     entityName2entityTypes[entity_name].append(entity_type)
        # 
        # if entity_type not in entityType2entityNames:
        #     entityType2entityNames[entity_type] = []
        # if entity_name not in entityType2entityNames[entity_type]:
        #     entityType2entityNames[entity_type].append(entity_name)

        if entity_id not in entityId2typeIds:
            entityId2typeIds[entity_id] = []
        if entity_type not in entityId2typeIds[entity_id]:
            entityId2typeIds[entity_id].append(type_id)

        if type_id not in typeId2entityIds:
            typeId2entityIds[type_id] = []
        if entity_id not in typeId2entityIds[type_id]:
            typeId2entityIds[type_id].append(entity_id)

    entity2type_file.close()

    return entityId2typeIds, typeId2entityIds


def build_type2id_v2(data_path):

    type2id = {}
    id2type = {}
    type_counter = 0
    with open(data_path + "/entity2types_ttv.txt") as entity2type_file:
        for line in entity2type_file:
            splitted_line = line.strip().split("\t")
            entity_type = splitted_line[1]

            if entity_type not in type2id:
                type2id[entity_type] = str(type_counter)
                id2type[str(type_counter)] = entity_type
                type_counter += 1
    entity2type_file.close()

    type2id["UNK"] = len(type2id)
    id2type[len(type2id)] = "UNK"

    return type2id, id2type


def build_type2relationType2frequency(data_path):

    type_relation_type_file = open(data_path + "/type2relation2type_ttv.txt", "r")

    type2relationType2frequency = {}
    for line in type_relation_type_file:
        splitted_line = line.strip().split("\t")
        head_type = splitted_line[0]
        relation = splitted_line[1]
        tail_type = splitted_line[2]

        relationType = (relation, tail_type)

        if head_type not in type2relationType2frequency:
            type2relationType2frequency[head_type] = {}

        if relationType not in type2relationType2frequency[head_type]:
            type2relationType2frequency[head_type][relationType] = 1
        else:
            type2relationType2frequency[head_type][relationType] += 1

    type_relation_type_file.close()

    return type2relationType2frequency


def build_schema_matrix_and_dict(relation2id, topNfilters, type2relationType2frequency, type2id):
    # tailType_relation_headType_tensor = torch.zeros((len(type2id), len(relation2id), len(type2id)),
    #                                                 requires_grad=False).to(device)

    list_of_type_relation_type = []
    for h_type in type2relationType2frequency:
        for relation_type in type2relationType2frequency[h_type]:
            relation = relation_type[0]
            t_type = relation_type[1]
            list_of_type_relation_type.append((h_type, relation, t_type))


    list_of_filtered_type_relation_type = []
    for h_type in type2relationType2frequency:
        if topNfilters <= 0:
            sorted_relation_tailType = sorted(type2relationType2frequency[h_type].items(), key=operator.itemgetter(1),
                                              reverse=True)
            for list_idx, relationType_frequency in reversed(list(enumerate(sorted_relation_tailType))):
                freq = relationType_frequency[1]
                if freq <= (topNfilters * -1):
                    del sorted_relation_tailType[list_idx]

        for relationType_frequency in sorted_relation_tailType:
            relation = relationType_frequency[0][0]
            t_type = relationType_frequency[0][1]
            list_of_filtered_type_relation_type.append((h_type, relation, t_type))
    list_of_type_relation_type = list_of_filtered_type_relation_type

    # tTypeId_relTypeId_hTypeId_list = []
    tTypeId_relTypeId_hTypeId_dict = {}
    for trt in list_of_type_relation_type:
        head_type = trt[0]
        relation = trt[1]
        tail_type = trt[2]


        head_type_id = type2id[head_type]
        relation_id = relation2id[relation]
        tail_type_id = type2id[tail_type]


        if head_type_id not in tTypeId_relTypeId_hTypeId_dict:
            tTypeId_relTypeId_hTypeId_dict[head_type_id] = []

        if (relation_id, tail_type_id) not in tTypeId_relTypeId_hTypeId_dict[head_type_id]:
            tTypeId_relTypeId_hTypeId_dict[head_type_id].append((relation_id, tail_type_id))

    return tTypeId_relTypeId_hTypeId_dict


def get_tTypeId_relTypeId_hTypeId_dict(data_path, relation2id, typeName2id, topNfilters):

    # Eg: type2relationType2frequency: {h_type:{(r,t_type):1,...},...}
    type2relationType2frequency = build_type2relationType2frequency(data_path)

    tTypeId_relTypeId_hTypeId_dict = build_schema_matrix_and_dict(relation2id,
                                                                  topNfilters,
                                                                  type2relationType2frequency,
                                                                  typeName2id)

    return tTypeId_relTypeId_hTypeId_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="/SMiLE/data/FB15k", help='Dataset path')
    parser.add_argument("--topNfilters", default=-700, type=int, help="Typed-triples frequency threshold")
    args = parser.parse_args()

    data_path = args.data_path
    topNfilters = args.topNfilters
    rel_name2id_map = get_name_id_map_from_txt(f"{data_path}/relationname2id.txt")

    typeName_to_id, typeId_to_name = build_type2id_v2(data_path)
    schema_dict = get_tTypeId_relTypeId_hTypeId_dict(data_path,
                                                     rel_name2id_map,typeName_to_id,
                                                     topNfilters)
    output_file = open(data_path + "/schema_ttv{}.txt".format(str(topNfilters)), "w")

    for head_type_id, rt_list in schema_dict.items():
        for rt_tuple in rt_list:
            relation_type_id = rt_tuple[0]
            tail_type_id = rt_tuple[1]
            line = str(head_type_id) + '\t' + str(relation_type_id) + '\t' + str(tail_type_id) + '\n'
            output_file.write(line)
    output_file.close()
