import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import kornia
from kornia.filters.kernels import get_gaussian_kernel2d


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.E = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.E(x)
        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.D_0 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
        )
        self.D = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(3, 3), stride=1),
        )

    def forward(self, x):
        out_f = self.D_0(x)
        out = self.D(out_f)
        return out, out_f


class Enhance(nn.Module):
    def __init__(self):
        super(Enhance, self).__init__()
        self.E = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.E(x)
        return out


class base(nn.Module):
    def __init__(self):
        super(base, self).__init__()
        self.B = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.B(x)
        return out


class FusionMoudle(nn.Module):
    def __init__(self):
        super(FusionMoudle, self).__init__()
        self.D = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(3, 3), stride=1),
        )

    def forward(self, vis, ir):
        x = vis + ir
        out = self.D(x)
        return out


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        device = x.device
        x = x + self.pe[:, :x.size(1)].to(device)
        return self.dropout(x)


class AffineTransform(nn.Module):
    """
    Add random affine transforms to a tensor image.
    Most functions are obtained from Kornia, difference:
    - gain the disp grid
    - no p and same_on_batch
    """

    def __init__(self, degrees=0, translate=0.1, return_warp=False):
        super(AffineTransform, self).__init__()
        self.trs = kornia.augmentation.RandomAffine(degrees, (translate, translate), return_transform=True, p=1)
        self.return_warp = return_warp

    def forward(self, input):
        # image shape
        device = input.device
        batch_size, _, height, weight = input.shape
        # affine transform
        warped, affine_param = self.trs(input)  # [batch_size, 3, 3]

        T = torch.FloatTensor([[2. / weight, 0, -1],
                               [0, 2. / height, -1],
                               [0, 0, 1]]).repeat(batch_size, 1, 1).to(device)
        theta = torch.inverse(torch.bmm(torch.bmm(T, affine_param), torch.inverse(T)))

        base = kornia.utils.create_meshgrid(height, weight, device=device).to(input.dtype)
        grid = F.affine_grid(theta[:, :2, :], size=input.size(), align_corners=False)  # [batch_size, height, weight, 2]

        disp = grid - base

        if self.return_warp:
            warped_grid_sample = F.grid_sample(input, grid)
            return warped_grid_sample, disp
        else:
            return disp


class ElasticTransform(nn.Module):
    """
    Add random elastic transforms to a tensor image.
    Most functions are obtained from Kornia, difference:
    - gain the disp grid
    - no p and same_on_batch
    """

    def __init__(self, kernel_size=63, sigma=32, align_corners=False, mode="bilinear", return_warp=False):
        super(ElasticTransform, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.align_corners = align_corners
        self.mode = mode
        self.return_warp = return_warp

    def forward(self, input):
        # generate noise
        batch_size, _, height, weight = input.shape
        device = input.device
        noise = torch.rand(batch_size, 2, height, weight, device=device) * 2 - 1
        # elastic transform
        if self.return_warp:
            warped, disp = self.elastic_transform2d(input, noise)
            return warped, disp
        else:
            disp = self.elastic_transform2d(input, noise)
            return disp

    def elastic_transform2d(self, image: torch.Tensor, noise: torch.Tensor):
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input image is not torch.Tensor. Got {type(image)}")

        if not isinstance(noise, torch.Tensor):
            raise TypeError(f"Input noise is not torch.Tensor. Got {type(noise)}")

        if not len(image.shape) == 4:
            raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {image.shape}")

        if not len(noise.shape) == 4 or noise.shape[1] != 2:
            raise ValueError(f"Invalid noise shape, we expect Bx2xHxW. Got: {noise.shape}")

        # unpack hyper parameters
        device = image.device

        # Get Gaussian kernel for 'y' and 'x' displacement
        kernel_x: torch.Tensor = get_gaussian_kernel2d((self.kernel_size, self.kernel_size), (self.sigma, self.sigma))[
            None]
        kernel_y: torch.Tensor = get_gaussian_kernel2d((self.kernel_size, self.kernel_size), (self.sigma, self.sigma))[
            None]

        # Convolve over a random displacement matrix and scale them with 'alpha'
        disp_x: torch.Tensor = noise[:, :1].to(device)
        disp_y: torch.Tensor = noise[:, 1:].to(device)

        disp_x = kornia.filters.filter2d(disp_x, kernel=kernel_y, border_type="constant")
        disp_y = kornia.filters.filter2d(disp_y, kernel=kernel_x, border_type="constant")

        # stack and normalize displacement
        disp = torch.cat([disp_x, disp_y], dim=1).permute(0, 2, 3, 1)

        if self.return_warp:
            # Warp image based on displacement matrix
            b, c, h, w = image.shape
            base = kornia.utils.create_meshgrid(h, w, device=image.device).to(image.dtype)
            grid = (base + disp).clamp(-1, 1)
            warped = F.grid_sample(image, grid, align_corners=self.align_corners, mode=self.mode)
            return warped, disp
        else:
            return disp


class ImageTransform(nn.Module):
    def __init__(self, ET_kernel_size=101, ET_kernel_sigma=16, AT_translate=0.01):
        super(ImageTransform, self).__init__()
        self.affine = AffineTransform(translate=AT_translate)
        self.elastic = ElasticTransform(kernel_size=ET_kernel_size, sigma=ET_kernel_sigma)

    def generate_grid(self, input):
        device = input.device
        batch_size, _, height, weight = input.size()
        # affine transform
        affine_disp = self.affine(input)  # warped, warped_grid_sample, disp
        # elastic transform
        elastic_disp = self.elastic(input)
        # make grid
        base = kornia.utils.create_meshgrid(height, weight).to(dtype=input.dtype).repeat(batch_size, 1, 1, 1).to(device)
        disp = affine_disp + elastic_disp
        grid = base + disp
        return grid

    def make_transform_matrix(self, grid):
        device = grid.device
        batch_size, height, weight, _ = grid.size()
        grid_s = torch.zeros_like(grid)
        grid_s[:, :, :, 0] = ((grid[:, :, :, 0] / 2) + 0.5) * (weight - 1)
        grid_s[:, :, :, 1] = ((grid[:, :, :, 1] / 2) + 0.5) * (height - 1)
        grid_s = torch.round(grid_s).to(dtype=torch.int64)
        base_s = kornia.utils.create_meshgrid(height, weight, normalized_coordinates=False).to(dtype=grid.dtype).repeat(
            batch_size, 1, 1, 1).to(device)
        x_index = base_s.reshape([batch_size, height * weight, 2]).to(dtype=torch.int64)
        y_index = grid_s.reshape([batch_size, height * weight, 2]).to(dtype=torch.int64)
        x_index_o = x_index[:, :, 0] + x_index[:, :, 1] * height
        y_index_o = y_index[:, :, 0] + y_index[:, :, 1] * height

        mask_x_min = (y_index[:, :, 0] > -1).to(torch.int64).unsqueeze(dim=-1).repeat(1, 1, 2)
        mask_y_min = (y_index[:, :, 1] > -1).to(torch.int64).unsqueeze(dim=-1).repeat(1, 1, 2)
        mask_x_max = (y_index[:, :, 0] < weight).to(torch.int64).unsqueeze(dim=-1).repeat(1, 1, 2)
        mask_y_max = (y_index[:, :, 1] < height).to(torch.int64).unsqueeze(dim=-1).repeat(1, 1, 2)
        mask = torch.mul(torch.mul(mask_x_min, mask_y_min), torch.mul(mask_x_max, mask_y_max))
        x_index_o = torch.mul(x_index_o, mask[:, :, 0])
        y_index_o = torch.mul(y_index_o, mask[:, :, 0])
        filler = mask[:, :, 0].to(dtype=torch.float32)

        index = torch.cat([x_index_o.unsqueeze(dim=1), y_index_o.unsqueeze(dim=1)], dim=1)
        index_r = torch.cat([y_index_o.unsqueeze(dim=1), x_index_o.unsqueeze(dim=1)], dim=1)

        return index, index_r, filler

    def forward(self, image_1, image_2):
        assert (image_1.size() == image_2.size())

        # generate grid that affine and elastic
        grid = self.generate_grid(image_1)

        # make tranform matrix
        index, index_r, filler = self.make_transform_matrix(grid)

        image_1_warp = F.grid_sample(image_1, grid, align_corners=False, mode='bilinear')
        image_2_warp = F.grid_sample(image_2, grid, align_corners=False, mode='bilinear')

        return image_1_warp, image_2_warp, index, index_r, filler


def window_partition(x, window_size, stride):
    # x: [batch_size, channels, height, weight]
    batch_size, channel, height, weight = x.size()
    unfold_win = nn.Unfold(kernel_size=(window_size, window_size), stride=stride)
    x_windows = unfold_win(x)  # [batch_size, p_w * p_w * channel, patch_nums]
    x_out_windows = x_windows.reshape(batch_size, channel, window_size, window_size, x_windows.size()[2]).permute(4,
                                                                                                                  0,
                                                                                                                  1,
                                                                                                                  2,
                                                                                                                  3)
    return x_out_windows


class resume(nn.Module):
    def __init__(self, height, weight, window_size, stride, channel):
        super(resume, self).__init__()
        self.channel = channel
        self.window_size = window_size
        self.flod_win = nn.Fold(output_size=(height, weight), kernel_size=(window_size, window_size), stride=stride)

    def forward(self, x_windows):
        size = x_windows.size()
        x_out = x_windows.permute(1, 2, 3, 4, 0)
        x_out = x_out.reshape(size[1], self.channel * self.window_size * self.window_size, size[0])
        r_out = self.flod_win(x_out)
        return r_out


def feature_reorganization(similaritys, x):
    """
    :param similaritys: [windows_num, batch_size, small_window_size * small_window_size, large_window_size * large_window_size]
    :param x: [batch_size, channel, height, weight]
    :return:
    """
    device = similaritys.device
    windows_num, batch_size, sw_size_pow2, lw_size_pow2 = similaritys.size()
    sw_size = int(math.sqrt(sw_size_pow2))
    lw_size = int(math.sqrt(lw_size_pow2))
    _, channel, height, weight = x.size()

    fold = nn.Fold(output_size=(sw_size, sw_size), kernel_size=(1, 1), stride=1)
    unflod_win = nn.Unfold(kernel_size=(1, 1), stride=1)
    resume_sw = resume(height, weight, sw_size, sw_size, channel)

    x_windows = df_window_partition(x, lw_size, sw_size)

    sample_windows = torch.zeros(windows_num, batch_size, channel, sw_size, sw_size, device=device)

    for i in range(windows_num):
        for j in range(batch_size):
            x_window = x_windows[i, j, :, :, :].unsqueeze(dim=0)
            x_patchs = unflod_win(x_window).permute(0, 2, 1)
            similarity = similaritys[i, j]
            sample_patch = torch.bmm(similarity.unsqueeze(dim=0), x_patchs).permute(0, 2, 1)
            sample_window = fold(sample_patch)
            sample_windows[i, j, :, :, :] = sample_window.squeeze(dim=0)

    sample = resume_sw(sample_windows)

    return sample


def df_window_partition(x, large_window_size, small_window_size, is_bewindow=True):
    """
    :param is_bewindow:
    :param small_window_size:
    :param large_window_size:
    :param x: [batch_size, channel, height, weight]
    :return:
    """
    batch_size, channel, height, weight = x.size()
    padding_num = int((large_window_size - small_window_size) / 2)
    center_unfold = nn.Unfold(kernel_size=(large_window_size, large_window_size), stride=small_window_size)
    x_center_w = center_unfold(F.pad(x, pad=[padding_num, padding_num, padding_num, padding_num]))

    corner_unfold = nn.Unfold(kernel_size=(large_window_size, large_window_size),
                              stride=(height - large_window_size, weight - large_window_size))
    x_corner_w = corner_unfold(x)
    top_bottom_unfold = nn.Unfold(kernel_size=(large_window_size, large_window_size),
                                  stride=(height - large_window_size, small_window_size))

    x_top_bottom_w = top_bottom_unfold(F.pad(x, pad=[padding_num, padding_num, 0, 0]))

    left_right_unfold = nn.Unfold(kernel_size=(large_window_size, large_window_size),
                                  stride=(small_window_size, weight - large_window_size))
    x_left_right_w = left_right_unfold(F.pad(x, pad=[0, 0, padding_num, padding_num]))

    weight_block_num = int(weight / small_window_size)
    height_block_num = int(height / small_window_size)

    m = torch.ones((1, 1, height_block_num, weight_block_num))
    m_unfold = nn.Unfold(kernel_size=(2, 2), stride=1)
    m_fold = nn.Fold(output_size=(height_block_num, weight_block_num), kernel_size=(2, 2), stride=1)
    mask = m_fold(m_unfold(m))

    mask[:, :, 0, :] = 3
    mask[:, :, height_block_num-1, :] = 3
    mask[:, :, height_block_num-1, weight_block_num-1] = 1
    mask[:, :, 0, weight_block_num-1] = 1
    mask[:, :, height_block_num-1, 0] = 1
    mask[:, :, 0, 0] = 1

    windows = torch.zeros_like(x_center_w)

    lr_index = 2
    tb_index = 1
    corner_index = 0
    for i in range(height_block_num):
        for j in range(weight_block_num):
            index = i * weight_block_num + j
            c = mask[0, 0, i, j]
            if c == 4:  # center
                windows[:, :, index] = x_center_w[:, :, index]
            elif c == 2:  # left and right
                windows[:, :, index] = x_left_right_w[:, :, lr_index]
                lr_index += 1
            elif c == 3:  # top and bottom
                if tb_index == weight_block_num-1:
                    tb_index += 2
                windows[:, :, index] = x_top_bottom_w[:, :, tb_index]
                tb_index += 1
            elif c == 1:  # corner
                windows[:, :, index] = x_corner_w[:, :, corner_index]
                corner_index += 1

    if is_bewindow:
        out_windows = windows.reshape(batch_size, channel, large_window_size, large_window_size,
                                      windows.size()[2]).permute(
            4,
            0,
            1,
            2,
            3)
    else:
        out_windows = windows
    return out_windows


class MHCSAB(nn.Module):
    def __init__(self):
        super(MHCSAB, self).__init__()
        self.LargeScaleEncoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(4),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(4),
            nn.PReLU(),
        )
        self.SmallScaleEncoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(4),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(4),
            nn.PReLU(),
        )
        self.mapping_l2s = nn.Sequential(
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, 4)
        )
        self.mapping_s2l = nn.Sequential(
            nn.Linear(4, 16),
            nn.GELU(),
            nn.Linear(16, 64)
        )
        self.Decoder = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(8),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(16),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(16),
            nn.PReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=1),
            nn.InstanceNorm2d(8),
            nn.PReLU(),
        )
        self.largesize = 4
        self.smallsize = 1
        self.dropout = 0.1
        self.channel = 4
        self.SA_large = nn.MultiheadAttention(self.largesize * self.largesize * self.channel, 1, self.dropout)
        self.SA_small = nn.MultiheadAttention(self.smallsize * self.smallsize * self.channel, 1, self.dropout)
        self.CA_large = nn.MultiheadAttention(self.largesize * self.largesize * self.channel, 1, self.dropout)
        self.CA_small = nn.MultiheadAttention(self.smallsize * self.smallsize * self.channel, 1, self.dropout)

    def self_attention(self, input_s, MHA):
        """
        :param MHA:
        :param input_s: [batch_size, patch_nums, patch_size * patch_size * channel]
        :return:
        """
        embeding_dim = input_s.size()[2]
        if MHA.training:
            PE = PositionalEncoding(embeding_dim, self.dropout).train()
        else:
            PE = PositionalEncoding(embeding_dim, self.dropout).eval()
        input_pe = PE(input_s)

        q = input_pe
        k = input_pe
        v = input_s
        a = MHA(q, k, v)[0]
        enhance = a + input_s
        return enhance

    def cross_attention(self, query, key_value, MHA):
        """
        :param MHA:
        :param query:
        :param key_value:
        :return:
        """
        embeding_dim = query.size()[2]
        if MHA.training:
            PE = PositionalEncoding(embeding_dim, self.dropout).train()
        else:
            PE = PositionalEncoding(embeding_dim, self.dropout).eval()

        q_pe = PE(query)
        kv_pe = PE(key_value)

        q = q_pe
        k = kv_pe
        v = key_value
        a = MHA(q, k, v)[0]
        enhance = a + query
        return enhance

    def forward(self, input):
        window_size = input.size()[3]
        flod_win_l = nn.Fold(output_size=(window_size, window_size), kernel_size=(self.largesize, self.largesize),
                             stride=self.largesize)
        flod_win_s = nn.Fold(output_size=(window_size, window_size), kernel_size=(self.smallsize, self.smallsize),
                             stride=self.smallsize)
        unflod_win_l = nn.Unfold(kernel_size=(self.largesize, self.largesize), stride=self.largesize)
        unflod_win_s = nn.Unfold(kernel_size=(self.smallsize, self.smallsize), stride=self.smallsize)

        large_scale_f = self.LargeScaleEncoder(input)
        small_scale_f = self.SmallScaleEncoder(input)

        large_scale_f_w = unflod_win_l(large_scale_f).permute(2, 0, 1)
        small_scale_f_w = unflod_win_s(small_scale_f).permute(2, 0, 1)

        large_scale_f_w_s = self.self_attention(large_scale_f_w, self.SA_large)
        small_scale_f_w_s = self.self_attention(small_scale_f_w, self.SA_small)

        l_size = large_scale_f_w_s.size()
        s_size = small_scale_f_w_s.size()

        large_scale_f_w_s_map2s = self.mapping_l2s(large_scale_f_w_s.reshape(l_size[0] * l_size[1], l_size[2])).reshape(
            l_size[0], l_size[1], s_size[2])
        small_scale_f_w_s_map2l = self.mapping_s2l(small_scale_f_w_s.reshape(s_size[0] * s_size[1], s_size[2])).reshape(
            s_size[0], s_size[1], l_size[2])

        large_scale_f_w_s_c = self.cross_attention(large_scale_f_w_s, small_scale_f_w_s_map2l, self.CA_large)
        small_scale_f_w_s_c = self.cross_attention(small_scale_f_w_s, large_scale_f_w_s_map2s, self.CA_small)

        large_scale_f_s_c = flod_win_l(large_scale_f_w_s_c.permute(1, 2, 0))
        small_scale_f_s_c = flod_win_s(small_scale_f_w_s_c.permute(1, 2, 0))

        enhance_f = torch.cat([large_scale_f_s_c, small_scale_f_s_c], dim=1)
        enhance_f = self.Decoder(enhance_f)

        return enhance_f


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.unflod_win_1 = nn.Unfold(kernel_size=(1, 1), stride=1)

    def c_similarity(self, s, r):
        B, Nt, E = s.shape
        s = s / math.sqrt(E)
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn = torch.bmm(s, r.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        return attn

    def forward(self, fixed_window, moving_window):
        fixed_patch = self.unflod_win_1(fixed_window).permute(0, 2, 1)
        moving_patch = self.unflod_win_1(moving_window).permute(0, 2, 1)
        similarity = self.c_similarity(fixed_patch, moving_patch)
        return similarity


def CMAP(fixed_windows, moving_windows, vis_MHCSA, ir_MHCSA, is_vis_fixed):
    assert (fixed_windows.size()[0] == fixed_windows.size()[0])
    att = Attention()
    device = fixed_windows.device
    window_nums, batch_size, _, window_size_f, _ = fixed_windows.size()
    window_size_m = moving_windows.size()[3]
    similaritys = torch.zeros(
        (window_nums, batch_size, int(window_size_f * window_size_f), int(window_size_m * window_size_m)),
        device=device)
    for i in range(window_nums):
        fixed_window = fixed_windows[i, :, :, :, :]  # [batch_size, channel, w_size, w_size]
        moving_window = moving_windows[i, :, :, :, :]  # [batch_size, channel, w_size, w_size]
        if is_vis_fixed:
            fixed_enhance = vis_MHCSA(fixed_window)
            moving_enhance = ir_MHCSA(moving_window)
        else:
            fixed_enhance = ir_MHCSA(fixed_window)
            moving_enhance = vis_MHCSA(moving_window)
        similarity = att(fixed_enhance, moving_enhance)
        similaritys[i, :, :, :] = similarity  # window_nums, batch_size, window_size * window_size
    return similaritys


class DictionaryRepresentationModule(nn.Module):
    def __init__(self):
        super(DictionaryRepresentationModule, self).__init__()
        element_size = 4
        self.element_size = element_size
        channel = 8
        l_n = 16
        c_n = 16
        self.Dictionary = nn.Parameter(
            torch.FloatTensor(l_n * c_n, 1, element_size * element_size * channel).to(torch.device("cuda:0")),
            requires_grad=True)
        nn.init.uniform_(self.Dictionary, 0, 1)
        self.unflod_win = nn.Unfold(kernel_size=(element_size, element_size), stride=element_size)
        self.CA = nn.MultiheadAttention(embed_dim=element_size * element_size * channel, num_heads=1, dropout=0)
        self.flod_win_1 = nn.Fold(output_size=(l_n * element_size, c_n * element_size),
                                  kernel_size=(element_size, element_size), stride=element_size)

    def forward(self, x):
        size = x.size()
        flod_win = nn.Fold(output_size=(size[2], size[3]), kernel_size=(self.element_size, self.element_size),
                           stride=self.element_size)
        D = self.Dictionary.repeat(1, size[0], 1)
        x_w = self.unflod_win(x).permute(2, 0, 1)

        q = x_w
        k = D
        v = D
        a = self.CA(q, k, v)[0]

        representation = flod_win(a.permute(1, 2, 0))
        visible_D = self.flod_win_1(self.Dictionary.permute(1, 2, 0))

        return representation, visible_D