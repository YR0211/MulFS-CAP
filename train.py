import os
import time
from pathlib import Path

import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from PIL import Image
from utils.utils import save_img
from tqdm import tqdm

import args
from loss import loss as Loss
from model import model
from utils import utils

model_name = "MulFS-CAP"

device_id = "0"

os.environ['CUDA_LAUNCH_BLOCKING'] = device_id

device = torch.device("cuda:" + device_id if torch.cuda.is_available() else "cpu")

now = int(time.time())
timeArr = time.localtime(now)
nowTime = time.strftime("%Y%m%d_%H-%M-%S", timeArr)
save_model_dir = args.args.train_save_model_dir + "/" + nowTime + "_" + model_name + "_model"
save_img_dir = args.args.train_save_img_dir + "/" + nowTime + "_" + model_name + "_img"
utils.check_dir(save_model_dir)
utils.check_dir(save_img_dir)

def adjust_learning_rate(optimizer, epoch_count):
    lr = args.args.LR + 0.5 * (args.args.LR_target - args.args.LR) * (
            1 + math.cos((epoch_count - args.args.Warm_epoch) / (args.args.Epoch - args.args.Warm_epoch) * math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def warmup_learning_rate(optimizer, epoch_count):
    lr = epoch_count * ((args.args.LR_target - args.args.LR) / args.args.Warm_epoch) + args.args.LR
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class TrainDataset(data.Dataset):
    def __init__(self, vis_dir, ir_dir, transform):
        super(TrainDataset, self).__init__()
        self.vis_dir = vis_dir
        self.ir_dir = ir_dir

        self.vis_path, self.vis_paths = self.find_file(self.vis_dir)
        self.ir_path, self.ir_paths = self.find_file(self.ir_dir)

        assert (len(self.vis_path) == len(self.ir_path))

        self.transform = transform

    def find_file(self, dir):
        path = os.listdir(dir)
        if os.path.isdir(os.path.join(dir, path[0])):
            paths = []
            for dir_name in os.listdir(dir):
                for file_name in os.listdir(os.path.join(dir, dir_name)):
                    paths.append(os.path.join(dir, file_name, file_name))
        else:
            paths = list(Path(dir).glob('*'))
        return path, paths

    def read_image(self, path):
        img = Image.open(str(path)).convert('L')
        img = self.transform(img)
        return img

    def __getitem__(self, index):
        vis_path = self.vis_paths[index]
        ir_path = self.ir_paths[index]

        vis_img = self.read_image(vis_path)
        ir_img = self.read_image(ir_path)

        return vis_img, ir_img

    def __len__(self):
        return len(self.vis_path)


tf = torchvision.transforms.Compose([
    torchvision.transforms.Resize([args.args.img_size, args.args.img_size]),
    torchvision.transforms.ToTensor()  # (0, 255) -> (0, 1)
])

dataset = TrainDataset(args.args.vis_train_dir, args.args.ir_train_dir, tf)

data_iter = data.DataLoader(
    dataset=dataset,
    shuffle=True,
    batch_size=args.args.batch_size,
    num_workers=4
)

iter_num = int(dataset.__len__() / args.args.batch_size)
save_image_iter = int(iter_num / args.args.save_image_num)

Lgrad = Loss.L_Grad().to(device)
CC = Loss.CorrelationCoefficient().to(device)
Lcorrespondence = Loss.L_correspondence()

with torch.no_grad():
    base = model.base()
    vis_MFE = model.FeatureExtractor()
    ir_MFE = model.FeatureExtractor()
    fusion_decoder = model.Decoder()
    PAFE = model.FeatureExtractor()
    decoder = model.Decoder()
    MN_vis = model.Enhance()
    MN_ir = model.Enhance()
    VISDP = model.DictionaryRepresentationModule()
    IRDP = model.DictionaryRepresentationModule()
    ImageDeformation = model.ImageTransform()
    MHCSA_vis = model.MHCSAB()
    MHCSA_ir = model.MHCSAB()
    fusion_module = model.FusionMoudle()

base.train()
vis_MFE.train()
ir_MFE.train()
fusion_decoder.train()
PAFE.train()
decoder.train()
VISDP.train()
IRDP.train()
MN_vis.train()
MN_ir.train()
MHCSA_vis.train()
MHCSA_ir.train()
fusion_module.train()

base.to(device)
vis_MFE.to(device)
ir_MFE.to(device)
fusion_decoder.to(device)
PAFE.to(device)
decoder.to(device)
VISDP.to(device)
IRDP.to(device)
MN_vis.to(device)
MN_ir.to(device)
MHCSA_vis.to(device)
MHCSA_ir.to(device)
fusion_module.to(device)

optimizer_FE = torch.optim.Adam([{'params': base.parameters()},
                                 {'params': vis_MFE.parameters()}, {'params': ir_MFE.parameters()},
                                 {'params': fusion_decoder.parameters()},
                                 {'params': PAFE.parameters()}, {'params': decoder.parameters()},
                                 {'params': MN_vis.parameters()}, {'params': MN_ir.parameters()}],
                                lr=0.0002)
optimizer_VISDP = torch.optim.Adam(VISDP.parameters(), lr=0.0008)
optimizer_IRDP = torch.optim.Adam(IRDP.parameters(), lr=0.0008)
optimizer_MHCSAvis = torch.optim.Adam(MHCSA_vis.parameters(), lr=args.args.LR)
optimizer_MHCSAir = torch.optim.Adam(MHCSA_ir.parameters(), lr=args.args.LR)
optimizer_FusionModule = torch.optim.Adam(fusion_module.parameters(), lr=0.0002)


def train(epoch):
    epoch_loss_VISDP = []
    epoch_loss_IRDP = []
    epoch_loss_same = []
    epoch_loss_correspondence_matrix = []
    epoch_loss_correspondence_predict = []

    for step, x in enumerate(data_iter):
        vis = x[0].to(device)  # vis
        ir = x[1].to(device)  # ir

        with torch.no_grad():
            vis_d, ir_d, _, index_r, _ = ImageDeformation(vis, ir)

        vis_1 = base(vis)
        vis_d_1 = base(vis_d)
        ir_1 = base(ir)
        ir_d_1 = base(ir_d)

        vis_fe = vis_MFE(vis_1)
        ir_fe = ir_MFE(ir_1)
        simple_fusion_f_1 = vis_fe + ir_fe
        fusion_image_1, fusion_f_1 = fusion_decoder(simple_fusion_f_1)
        vis_d_fe = vis_MFE(vis_d_1)
        ir_d_fe = ir_MFE(ir_d_1)
        simple_fusion_d_f_1 = vis_d_fe + ir_d_fe
        fusion_d_image_1, fusion_d_f_1 = fusion_decoder(simple_fusion_d_f_1)

        vis_f = PAFE(vis_1)
        ir_f = PAFE(ir_1)
        simple_fusion_f = vis_f + ir_f
        fusion_image, fusion_f = decoder(simple_fusion_f)
        vis_d_f = PAFE(vis_d_1)
        ir_d_f = PAFE(ir_d_1)
        simple_fusion_d_f = vis_d_f + ir_d_f
        fusion_d_image, fusion_d_f = decoder(simple_fusion_d_f)

        vis_e_f = MN_vis(vis_f)
        ir_e_f = MN_ir(ir_f)
        vis_d_e_f = MN_vis(vis_d_f)
        ir_d_e_f = MN_ir(ir_d_f)

        VISDP_vis_f, _ = VISDP(vis_e_f)
        IRDP_ir_f, _ = IRDP(ir_e_f)
        VISDP_vis_d_f, _ = VISDP(vis_d_e_f)
        IRDP_ir_d_f, _ = IRDP(ir_d_e_f)

        fixed_DP = VISDP_vis_f
        moving_DP = IRDP_ir_d_f

        moving_DP_lw = model.df_window_partition(moving_DP, args.args.large_w_size, args.args.small_w_size)
        fixed_DP_sw = model.window_partition(fixed_DP, args.args.small_w_size, args.args.small_w_size)

        correspondence_matrixs = model.CMAP(fixed_DP_sw, moving_DP_lw, MHCSA_vis, MHCSA_ir,
                                                     True)

        ir_d_f_sample = model.feature_reorganization(correspondence_matrixs, ir_d_fe)
        fusion_image_sample = fusion_module(vis_fe, ir_d_f_sample)

        # calculate loss
        loss_fusion = Lgrad(vis, ir, fusion_image) + Loss.Loss_intensity(vis, ir, fusion_image) + \
                      Lgrad(vis_d, ir_d, fusion_d_image) + Loss.Loss_intensity(vis_d, ir_d, fusion_d_image)
        loss_fusion_1 = Lgrad(vis, ir, fusion_image_1) + Loss.Loss_intensity(vis, ir, fusion_image_1) + \
                        Lgrad(vis_d, ir_d, fusion_d_image_1) + Loss.Loss_intensity(vis_d, ir_d, fusion_d_image_1)
        loss_0 = loss_fusion
        loss_VISDP = - CC(VISDP_vis_f, fusion_f.detach()) - CC(VISDP_vis_d_f, fusion_d_f.detach())
        loss_IRDP = - CC(IRDP_ir_f, fusion_f.detach()) - CC(IRDP_ir_d_f, fusion_d_f.detach())
        loss_same = F.mse_loss(VISDP_vis_f, IRDP_ir_f) + F.mse_loss(VISDP_vis_d_f, IRDP_ir_d_f)
        loss_1 = 2 * (loss_VISDP + loss_IRDP + loss_same)
        loss_2 = Lgrad(vis, ir, fusion_image_sample) + Loss.Loss_intensity(vis, ir, fusion_image_sample)
        loss_correspondence_matrix, loss_correspondence_matrix_1 = Lcorrespondence(
            correspondence_matrixs, index_r)
        loss_3 = 4 * (loss_correspondence_matrix + loss_correspondence_matrix_1)
        loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_fusion_1

        # optimizer network
        optimizer_VISDP.zero_grad()
        optimizer_IRDP.zero_grad()
        optimizer_MHCSAvis.zero_grad()
        optimizer_MHCSAir.zero_grad()
        optimizer_FusionModule.zero_grad()
        optimizer_FE.zero_grad()
        loss.backward()
        optimizer_FE.step()
        optimizer_VISDP.step()
        optimizer_IRDP.step()
        optimizer_MHCSAvis.step()
        optimizer_MHCSAir.step()
        optimizer_FusionModule.step()

        epoch_loss_VISDP.append(loss_VISDP.item())
        epoch_loss_IRDP.append(loss_IRDP.item())
        epoch_loss_same.append(loss_same.item())
        epoch_loss_correspondence_matrix.append(loss_correspondence_matrix.item())
        epoch_loss_correspondence_predict.append(loss_correspondence_matrix_1.item())

        if step % save_image_iter == 0:
            epoch_step_name = str(epoch) + "epoch" + str(step) + "step"
            if epoch % 2 == 0:
                output_name = save_img_dir + "/" + epoch_step_name + ".jpg"
                out = torch.cat([vis, ir_d, fusion_image_1, fusion_image_sample, fusion_d_image_1], dim=2)
                save_img(out, output_name)

        if ((epoch + 1) == args.args.Epoch and (step + 1) % iter_num == 0) or (
                epoch % args.args.save_model_num == 0 and (step + 1) % iter_num == 0):
            module_name = "base"
            save_dir = '{:s}/epoch{:d}_iter{:d}_{:s}.pth'.format(save_model_dir, epoch, step + 1,
                                                                 module_name)
            utils.save_state_dir(base, save_dir)

            module_name = "vis_MFE"
            save_dir = '{:s}/epoch{:d}_iter{:d}_{:s}.pth'.format(save_model_dir, epoch, step + 1,
                                                                 module_name)
            utils.save_state_dir(vis_MFE, save_dir)

            module_name = "ir_MFE"
            save_dir = '{:s}/epoch{:d}_iter{:d}_{:s}.pth'.format(save_model_dir, epoch, step + 1,
                                                                 module_name)
            utils.save_state_dir(ir_MFE, save_dir)

            module_name = "PAFE"
            save_dir = '{:s}/epoch{:d}_iter{:d}_{:s}.pth'.format(save_model_dir, epoch, step + 1,
                                                                 module_name)
            utils.save_state_dir(PAFE, save_dir)

            module_name = "MN_vis"
            save_dir = '{:s}/epoch{:d}_iter{:d}_{:s}.pth'.format(save_model_dir, epoch, step + 1,
                                                                 module_name)
            utils.save_state_dir(MN_vis, save_dir)

            module_name = "MN_ir"
            save_dir = '{:s}/epoch{:d}_iter{:d}_{:s}.pth'.format(save_model_dir, epoch, step + 1,
                                                                 module_name)
            utils.save_state_dir(MN_ir, save_dir)

            module_name = "VISDP"
            save_dir = '{:s}/epoch{:d}_iter{:d}_{:s}.pth'.format(save_model_dir, epoch, step + 1,
                                                                 module_name)
            utils.save_state_dir(VISDP, save_dir)

            module_name = "IRDP"
            save_dir = '{:s}/epoch{:d}_iter{:d}_{:s}.pth'.format(save_model_dir, epoch, step + 1,
                                                                 module_name)
            utils.save_state_dir(IRDP, save_dir)

            module_name = "MHCSA_vis"
            save_dir = '{:s}/epoch{:d}_iter{:d}_{:s}.pth'.format(save_model_dir, epoch, step + 1,
                                                                 module_name)
            utils.save_state_dir(MHCSA_vis, save_dir)

            module_name = "MHCSA_ir"
            save_dir = '{:s}/epoch{:d}_iter{:d}_{:s}.pth'.format(save_model_dir, epoch, step + 1,
                                                                 module_name)
            utils.save_state_dir(MHCSA_ir, save_dir)

            module_name = "fusion_module"
            save_dir = '{:s}/epoch{:d}_iter{:d}_{:s}.pth'.format(save_model_dir, epoch, step + 1,
                                                                 module_name)
            utils.save_state_dir(fusion_module, save_dir)

    epoch_loss_correspondence_matrix_mean = np.mean(epoch_loss_correspondence_matrix)
    epoch_loss_correspondence_predict_mean = np.mean(epoch_loss_correspondence_predict)
    epoch_loss_VISDP_mean = np.mean(epoch_loss_VISDP)
    epoch_loss_IRDP_mean = np.mean(epoch_loss_IRDP)
    epoch_loss_same_mean = np.mean(epoch_loss_same)

    print()
    print(" -epoch " + str(epoch))
    print(" -loss_cm " + str(epoch_loss_correspondence_matrix_mean) + " -loss_cp " + str(
        epoch_loss_correspondence_predict_mean))
    print(" -loss_VISDP " + str(epoch_loss_VISDP_mean) + " -loss_IRDP " + str(
        epoch_loss_IRDP_mean))
    print(" -loss_same " + str(epoch_loss_same_mean))


for epoch in tqdm(range(args.args.Epoch)):
    if epoch < args.args.Warm_epoch:
        warmup_learning_rate(optimizer_MHCSAvis, epoch)
        warmup_learning_rate(optimizer_MHCSAir, epoch)
    else:
        adjust_learning_rate(optimizer_MHCSAvis, epoch)
        adjust_learning_rate(optimizer_MHCSAir, epoch)

    train(epoch)
