import torch
import torch.nn as nn

class ReconstructionLoss(nn.Module):
    def __init__(self, velocity_loss_weight, kl_loss_weight, mse_loss_weight=0.5):
        super(ReconstructionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.velocity_loss_weight = velocity_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.mse_loss_weight = mse_loss_weight
    
    def forward(self, pred_data_1, pred_data_2, target_1, target_2, mean_1, log_var_1, mean_2, log_var_2, mean_duet, log_var_duet, is_simplified_model, out_1, out_2, dancer1_data, dancer2_data, frames=1):
        if is_simplified_model:
            mse_1 = self.mse_loss(out_1, dancer1_data)
            mse_2 = self.mse_loss(out_2, dancer2_data)
            return self.mse_loss_weight * (mse_1 + mse_2)

        else:
            mse_1 = self.mse_loss(pred_data_1, target_1)
            mse_2 = self.mse_loss(pred_data_2, target_2)

            # video continuity loss
            # [batch, seq_len, 29, 3]
            velocity_1 = pred_data_1[:, 1:, :, :] - pred_data_1[:, :-1, :, :]
            velocity_1_diff = velocity_1[:, frames:, :, :] - velocity_1[:, :-frames, :, :]
            velocity_loss_1 = torch.mean(torch.norm(velocity_1_diff, p=2, dim=[2, 3]))

            velocity_2 = pred_data_2[:, 1:, :, :] - pred_data_2[:, :-1, :, :]
            velocity_2_diff = velocity_2[:, frames:, :, :] - velocity_2[:, :-frames, :, :]
            velocity_loss_2 = torch.mean(torch.norm(velocity_2_diff, p=2, dim=[2, 3]))

            # KL divergence loss
            return self.mse_loss_weight * mse_1 + self.mse_loss_weight * mse_2 + self.velocity_loss_weight * velocity_loss_1 + self.velocity_loss_weight * velocity_loss_2 + self.kl_loss_weight * (
                - 0.5 * torch.mean(torch.sum(1 + log_var_1 - mean_1 ** 2 - torch.exp(log_var_1), dim=-1), dim=0)
                - 0.5 * torch.mean(torch.sum(1 + log_var_2 - mean_2 ** 2 - torch.exp(log_var_2), dim=-1), dim=0)
                - 0.5 * torch.mean(torch.sum(1 + log_var_duet - mean_duet ** 2 - torch.exp(log_var_duet), dim=-1), dim=0)
            )
