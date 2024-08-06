import torch
import torch.nn as nn

class ReconstructionLoss(nn.Module):
    def __init__(self):
        super(ReconstructionLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    
    def forward(self, out_1, out_2, target_1, target_2, mean_1, log_var_1, mean_2, log_var_2, mean_duet, log_var_duet):
        # out_1: predict dancer 2, out_2: predict dancer 1
        mse_1 = self.mse_loss(out_1, target_1)
        mse_2 = self.mse_loss(out_2, target_2)

        # video continuity loss
        velocity_1 = out_1[:, 1:, :, :] - out_1[:, :-1, :, :]
        velocity_1_diff = velocity_1[:, 1:, :, :] - velocity_1[:, :-1, :, :]
        velocity_loss_1 = torch.mean(torch.norm(velocity_1_diff, p=2, dim=[2, 3]))

        velocity_2 = out_2[:, 1:, :, :] - out_2[:, :-1, :, :]
        velocity_2_diff = velocity_2[:, 1:, :, :] - velocity_2[:, :-1, :, :]
        velocity_loss_2 = torch.mean(torch.norm(velocity_2_diff, p=2, dim=[2, 3]))

        total_variation_loss = torch.sum(torch.abs(out_1[:, 1:] - out_1[:, :-1])) + torch.sum(torch.abs(out_2[:, 1:] - out_2[:, :-1]))

        # KL divergence loss
        return 0.5 * mse_1 + 0.5 * mse_2 + 0.1 * velocity_loss_1 + 0.1 * velocity_loss_2 + 0.01 * total_variation_loss + 0.0001 * (
            - 0.5 * torch.mean(torch.sum(1 + log_var_1 - mean_1 ** 2 - torch.exp(log_var_1), dim=-1), dim=0)
            - 0.5 * torch.mean(torch.sum(1 + log_var_2 - mean_2 ** 2 - torch.exp(log_var_2), dim=-1), dim=0)
            - 0.5 * torch.mean(torch.sum(1 + log_var_duet - mean_duet ** 2 - torch.exp(log_var_duet), dim=-1), dim=0)
        )
