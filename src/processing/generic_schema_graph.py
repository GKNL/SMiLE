import random

import os
import pandas as pd
from src.processing.attributed_graph import AttributedGraph
from src.build_schema import get_tTypeId_relTypeId_hTypeId_dict, build_entity2types_dictionaries, build_type2id_v2


class GenericSchemaGraph(AttributedGraph):
    def __init__(self, main_dir, rel_name2id_map, ent_name2id_map, topNfilters=-300, attributes_file="", sample_training_set=False):
        """

        :param main_dir:
        :param rel_name2id_map:
        :param topNfilters:
        :param attributes_file:
        :param sample_training_set:

        Assumes derived classes will create a networkx graph object along with
        following
        1) self.unique_relations = set()
        2) self.node_attr_dfs = dict()
        3) self.node_types = dict()
        4) self.G  = nx.Graph()
        """
        super().__init__()
        self.main_dir = main_dir
        self.create_normalized_node_id_map()

        self.entity_to_id = ent_name2id_map
        self.relation_to_id = rel_name2id_map
        self.id_to_relation = {}
        self.id_to_entity = {}
        for tmpkey in self.relation_to_id:
            self.id_to_relation[self.relation_to_id[tmpkey]] = tmpkey
        for tmpkey in self.entity_to_id:
            self.id_to_entity[self.entity_to_id[tmpkey]] = tmpkey
        self.typeName_to_id, _ = build_type2id_v2(self.main_dir)
        self.entityId2typeIds, self.typeId2entityIds = build_entity2types_dictionaries(main_dir,
                                                                                       self.entity_to_id,
                                                                                       self.typeName_to_id)

        # build schema
        self.schema_dict = self.build_schema_info(topNfilters)

        self.load_schema_graph()

        # get product attributes
        self.node_attr = dict()
        self.node_attr_list = []

        if attributes_file == "":
            for node_id in self.G.nodes():
                self.node_attr[node_id] = dict()
                self.node_attr[node_id]["schema_attr"] = "True"
            self.node_attr_list = ["schema_attr"]
        else:
            raise NotImplementedError

        self.node_attr_dfs = {
            "schema_type": pd.DataFrame.from_dict(self.node_attr, orient="index")
        }


    def build_schema_info(self, topNfilters):
        schema_file_dir = self.main_dir + '/schema_ttv{}.txt'.format(str(topNfilters))
        if os.path.exists(schema_file_dir):
            print("\n load existing schema data from txt({}) ...".format(str(topNfilters)))
            schema_dict = {}
            fin = open(schema_file_dir, "r")
            for line in fin:
                trt = line.strip().split('\t')
                head_type_id = trt[0]
                relation_type_id = trt[1]
                tail_type_id = trt[2]

                if head_type_id not in schema_dict:
                    schema_dict[head_type_id] = []

                if (relation_type_id, tail_type_id) not in schema_dict[head_type_id]:
                    schema_dict[head_type_id].append((relation_type_id, tail_type_id))

            fin.close()
        else:
            print("\n generating schema data dictionary({})...".format(str(topNfilters)))
            schema_dict = get_tTypeId_relTypeId_hTypeId_dict(self.main_dir,
                                                             self.relation_to_id,
                                                             self.typeName_to_id,
                                                             topNfilters)
        return schema_dict


    def load_schema_graph(self):
        for head_type_id, rt_list in self.schema_dict.items():
            for rt_tuple in rt_list:
                relation_type_id = rt_tuple[0]
                tail_type_id = rt_tuple[1]
                self.G.add_edge(head_type_id, tail_type_id, label=relation_type_id)
                self.unique_relations.add(relation_type_id)
                self.node_types[head_type_id] = "schema_type"
                self.node_types[tail_type_id] = "schema_type"

        return

    def get_continuous_cols(self):
        return {"schema_type": None}

    def get_wide_cols(self):
        return {"schema_type": self.node_attr_list}
