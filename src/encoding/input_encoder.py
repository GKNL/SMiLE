import torch


class NodeEncoder(torch.nn.Module):
    def __init__(
        self,
        base_embedding_dim,
        num_nodes,
        pretrained_node_embedding_tensor,
        is_pre_trained,
    ):

        super().__init__()
        self.pretrained_node_embedding_tensor = pretrained_node_embedding_tensor
        self.base_embedding_dim = base_embedding_dim

        if not is_pre_trained:
            print("No pretrained embedding, so initial by torch.nn.Embedding \n")
            self.base_embedding_layer = torch.nn.Embedding(
                num_nodes, base_embedding_dim
            ).cuda()
            self.base_embedding_layer.weight.data.uniform_(-1, 1)
        else:
            # print("NodeEncoder: Node2vec embedding size",
            #     pretrained_node_embedding_tensor.size())

            self.base_embedding_layer = torch.nn.Embedding.from_pretrained(
                pretrained_node_embedding_tensor
            ).cuda()

    def forward(self, node_id):
        node_id = torch.LongTensor([node_id]).cuda()
        x_base = self.base_embedding_layer(node_id)  # Get embeddings for index 'node_id'

        return x_base
