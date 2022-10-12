import numpy as np
import torch
import torch.nn as nn

class Contrast_relation(nn.Module):
    def __init__(self, hidden_dim, tau=0.8):
        super(Contrast_relation, self).__init__()
        self.mapping = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.Dropout(0.2)
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        for model in self.mapping:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    # def loss_relation_total(self, target_emb, pos_neg_samples_emb):
    #     res_loss = torch.tensor([0]).cuda()
    #     if len(target_emb) == 0:
    #         return res_loss
    #     for i, tar_pos_neg_samples_tensor_list in enumerate(pos_neg_samples_emb):
    #         tmp_target_emb_mapping = self.mapping(target_emb[i]).unsqueeze(0)
    #         for j, tar_pos_neg_samples_tensor in enumerate(tar_pos_neg_samples_tensor_list):
    #             tmp_samples_emb_mapping = self.mapping(tar_pos_neg_samples_tensor)
    #             matrix_sim_between_tar_and_sample = self.sim(tmp_target_emb_mapping, tmp_samples_emb_mapping)
    #             matrix_sim_softmax = matrix_sim_between_tar_and_sample / (torch.sum(matrix_sim_between_tar_and_sample, dim=1).view(-1, 1) + 1e-8)
    #             tmp_loss = -torch.log(matrix_sim_softmax[0])
    #             res_loss = torch.add(res_loss, tmp_loss)
    #     mean_loss = res_loss / len(target_emb)
    #     return mean_loss

    def loss_relation(self, target_emb, pos_neg_samples_emb):
        res_loss = torch.tensor([0]).cuda()
        if len(target_emb) == 0:
            return res_loss
        for i, tar_pos_neg_sample_tensor in enumerate(pos_neg_samples_emb):
            mask_list = np.zeros(tar_pos_neg_sample_tensor.shape[0])
            mask_list[0] = 1
            mask_tensor = torch.tensor(mask_list).cuda()

            tmp_target_emb_mapping = self.mapping(target_emb[i]).unsqueeze(0)
            tmp_samples_emb_mapping = self.mapping(tar_pos_neg_sample_tensor)
            matrix_sim_between_tar_and_sample = self.sim(tmp_target_emb_mapping, tmp_samples_emb_mapping)
            matrix_sim_softmax = matrix_sim_between_tar_and_sample / (torch.sum(matrix_sim_between_tar_and_sample, dim=1).view(-1, 1) + 1e-8)
            tmp_loss = -torch.log(matrix_sim_softmax.mul(mask_tensor).sum(dim=-1))
            res_loss = torch.add(res_loss, tmp_loss)
        mean_loss = res_loss / len(target_emb)
        return mean_loss

    def forward(self, target_emb, pos_neg_samples_emb):
        loss_relation = self.loss_relation(target_emb, pos_neg_samples_emb)

        return loss_relation
