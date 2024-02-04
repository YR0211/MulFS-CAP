import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
from model import model

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

def Loss_intensity(vis, ir, image_fused):
    assert (vis.size() == ir.size() == image_fused.size())
    ir_li = F.l1_loss(image_fused, ir)
    vis_li = F.l1_loss(image_fused, vis)
    li = ir_li + vis_li
    return li

class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, img1, img2, image_fused=None):
        if image_fused == None:
            image_1_Y = img1[:, :1, :, :]
            image_2_Y = img2[:, :1, :, :]
            gradient_1 = self.sobelconv(image_1_Y)
            gradient_2 = self.sobelconv(image_2_Y)
            Loss_gradient = F.l1_loss(gradient_1, gradient_2)
            return Loss_gradient
        else:
            image_1_Y = img1[:, :1, :, :]
            image_2_Y = img2[:, :1, :, :]
            image_fused_Y = image_fused[:, :1, :, :]
            gradient_1 = self.sobelconv(image_1_Y)
            gradient_2 = self.sobelconv(image_2_Y)
            gradient_fused = self.sobelconv(image_fused_Y)
            gradient_joint = torch.max(gradient_1, gradient_2)
            Loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
            return Loss_gradient


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class CorrelationCoefficient(nn.Module):
    def __init__(self):
        super(CorrelationCoefficient, self).__init__()

    def c_CC(self, A, B):
        A_mean = torch.mean(A, dim=[2, 3], keepdim=True)
        B_mean = torch.mean(B, dim=[2, 3], keepdim=True)
        A_sub_mean = A - A_mean
        B_sub_mean = B - B_mean
        sim = torch.sum(torch.mul(A_sub_mean, B_sub_mean))
        A_sdev = torch.sqrt(torch.sum(torch.pow(A_sub_mean, 2)))
        B_sdev = torch.sqrt(torch.sum(torch.pow(B_sub_mean, 2)))
        out = sim / (A_sdev * B_sdev)
        return out

    def forward(self, A, B, Fusion=None):
        if Fusion is None:
            CC = self.c_CC(A, B)
        else:
            r_1 = self.c_CC(A, Fusion)
            r_2 = self.c_CC(B, Fusion)
            CC = (r_1 + r_2) / 2
        return CC


class L_correspondence(nn.Module):
    def __init__(self, height=256, weight=256):
        super(L_correspondence, self).__init__()
        self.height = height
        self.weight = weight

    def forward(self, correspondence_matrixs, index_r):
        size = correspondence_matrixs.size()
        device = correspondence_matrixs.device
        small_window_size = int(math.sqrt(size[2]))
        large_window_size = int(math.sqrt(size[3]))
        batch_size = size[1]
        win_num = size[0]
        index = index_r

        base_index = torch.arange(0, self.height * self.weight, device=device).reshape(self.height, self.weight)
        unfold_win = nn.Unfold(kernel_size=(small_window_size, small_window_size), stride=small_window_size)
        base_index = base_index.repeat(1, 1, 1, 1).to(dtype=torch.float32)
        sw_absolute_base_index = unfold_win(base_index)
        lw_absolute_base_index = model.df_window_partition(base_index, large_window_size, small_window_size,
                                                           is_bewindow=False)
        sw_win_ralative_base_index = torch.arange(0, small_window_size * small_window_size, device=device)

        loss_correspondence_matrix = torch.zeros(win_num, batch_size, device=device)
        loss_correspondence_matrix_1 = torch.zeros(win_num, batch_size, device=device)

        for i in range(batch_size):
            for j in range(win_num):
                lw_win_absolute_base_index = lw_absolute_base_index[0, :, j]
                sw_win_absolute_base_index = sw_absolute_base_index[0, :, j]
                indices = (lw_win_absolute_base_index.unsqueeze(dim=1) == index[i, 1, :]).nonzero(as_tuple=True)
                corresponding_lw_absolute_indices = lw_win_absolute_base_index[indices[0]]
                corresponding_allimgae_absolute_indices = index[i, 0, :][indices[1]]
                corresponding_lw_relative_indices = indices[0]
                insw_indices = (sw_win_absolute_base_index.unsqueeze(
                    dim=1) == corresponding_allimgae_absolute_indices).nonzero(as_tuple=True)
                insw_corresponding_sw_absolute_index = sw_win_absolute_base_index[insw_indices[0]]
                insw_corresponding_sw_relative_index = sw_win_ralative_base_index[insw_indices[0]]
                insw_corresponding_lw_absolute_index = corresponding_lw_absolute_indices[insw_indices[1]]
                insw_corresponding_lw_relative_index = corresponding_lw_relative_indices[insw_indices[1]]

                zero_mask = torch.logical_or(insw_corresponding_sw_absolute_index != 0,
                                             insw_corresponding_lw_absolute_index != 0).nonzero()
                nozeropair_insw_corresponding_sw_relative_index = insw_corresponding_sw_relative_index[zero_mask]
                nozeropair_insw_corresponding_lw_relative_index = insw_corresponding_lw_relative_index[zero_mask]

                corresponding_win_index = torch.cat([nozeropair_insw_corresponding_sw_relative_index.permute(1, 0),
                                                     nozeropair_insw_corresponding_lw_relative_index.permute(1, 0)],
                                                    dim=0)
                corresponding_win_matrix = torch.sparse_coo_tensor(corresponding_win_index,
                                                                   torch.ones(corresponding_win_index.size()[1],
                                                                              device=device),
                                                                   (small_window_size * small_window_size,
                                                                    large_window_size * large_window_size))
                assert (torch.sum(torch.abs(corresponding_win_matrix.to_dense())) != 0)

                predict_correspondence_matrix = correspondence_matrixs[j, i, :, :]
                c_num = nozeropair_insw_corresponding_sw_relative_index.size()[0]
                predict_correspondence_matrix_1 = torch.clamp(predict_correspondence_matrix, 1e-6, 1 - 1e-6)
                l_cm = (-1 / c_num) * torch.sum(
                    torch.mul(torch.log(predict_correspondence_matrix_1), corresponding_win_matrix.to_dense()))

                loss_correspondence_matrix[j, i] = l_cm

                l_c = F.l1_loss(predict_correspondence_matrix, corresponding_win_matrix.to_dense())
                loss_correspondence_matrix_1[j, i] = l_c

        loss_correspondence_matrix = torch.mean(loss_correspondence_matrix)
        loss_correspondence_matrix_1 = torch.mean(loss_correspondence_matrix_1)

        return loss_correspondence_matrix, loss_correspondence_matrix_1
