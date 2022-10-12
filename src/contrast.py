import torch
import torch.nn as nn

class Contrast(nn.Module):
    def __init__(self, hidden_dim, tau=0.8, lam=0.5):
        super(Contrast, self).__init__()
        self.inter_mapping = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.Dropout(0.2)
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.intra_mapping = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.tau = tau
        self.lam = lam
        for model in self.inter_mapping:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)
        for model in self.intra_mapping:
            if isinstance(model, nn.Linear):
                nn.init.xavier_normal_(model.weight, gain=1.414)

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def loss_inter_schema(self, target_emb, inter_samples_emb, inter_positive_mask):
        res_loss = torch.tensor([0]).cuda()
        if len(target_emb) == 0:
            return res_loss
        for i, inter_sample_tensor in enumerate(inter_samples_emb):
            tmp_target_emb_mapping = self.inter_mapping(target_emb[i]).unsqueeze(0)
            tmp_samples_emb_mapping = self.inter_mapping(inter_sample_tensor)
            matrix_sim_between_tar_and_sample = self.sim(tmp_target_emb_mapping, tmp_samples_emb_mapping)
            matrix_sim_softmax = matrix_sim_between_tar_and_sample / (torch.sum(matrix_sim_between_tar_and_sample, dim=1).view(-1, 1) + 1e-8)
            tmp_loss = -torch.log(matrix_sim_softmax.mul(inter_positive_mask[i]).sum(dim=-1))
            res_loss = torch.add(res_loss, tmp_loss)
        mean_loss = res_loss / len(target_emb)
        return mean_loss

    def loss_intra_schema(self, target_emb, intra_samples_emb, intra_positive_mask):
        res_loss = torch.tensor([0]).cuda()
        if len(target_emb) == 0:
            return res_loss
        for i, intra_sample_tensor in enumerate(intra_samples_emb):
            tmp_target_emb_mapping = self.intra_mapping(target_emb[i]).unsqueeze(0)
            tmp_samples_emb_mapping = self.intra_mapping(intra_sample_tensor)
            matrix_sim_between_tar_and_sample = self.sim(tmp_target_emb_mapping, tmp_samples_emb_mapping)
            matrix_sim_softmax = matrix_sim_between_tar_and_sample / (
                        torch.sum(matrix_sim_between_tar_and_sample, dim=1).view(-1, 1) + 1e-8)
            tmp_loss = -torch.log(matrix_sim_softmax.mul(intra_positive_mask[i]).sum(dim=-1))
            res_loss = torch.add(res_loss, tmp_loss)
        mean_loss = res_loss / len(target_emb)
        return mean_loss

    def forward(self, target_emb_inter, target_emb_intra, inter_samples_emb, intra_samples_emb, inter_positive_mask, intra_positive_mask):
        loss_inter_schema = self.loss_inter_schema(target_emb_inter, inter_samples_emb, inter_positive_mask)
        loss_intra_schema = self.loss_intra_schema(target_emb_intra, intra_samples_emb, intra_positive_mask)

        total_loss = self.lam * loss_inter_schema + (1 - self.lam) * loss_intra_schema

        return total_loss, loss_inter_schema, loss_intra_schema
