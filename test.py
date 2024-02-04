import os
from pathlib import Path

import torch
import torch.utils.data as data
import torchvision
from PIL import Image
from tqdm import tqdm

import model.model as model
import utils.utils as utils
import args

device_id = "0"

os.environ['CUDA_LAUNCH_BLOCKING'] = device_id

device = torch.device("cuda:" + device_id if torch.cuda.is_available() else "cpu")


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

        return vis_img, ir_img, str(vis_path)

    def __len__(self):
        return len(self.vis_path)


tf_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize([args.args.img_size, args.args.img_size]),
    torchvision.transforms.ToTensor()  # (0, 255) -> (0, 1)
])

vis_test_dir = r"..."
ir_test_dir = r"..."

save_dir = "./results"
save_ird_dir = save_dir + "/ird"
save_fusion_dir = save_dir + "/fusion"

utils.check_dir(save_dir)
utils.check_dir(save_ird_dir)
utils.check_dir(save_fusion_dir)

test_dataset = TrainDataset(vis_test_dir, ir_test_dir, tf_test)

test_data_iter = data.DataLoader(
    dataset=test_dataset,
    shuffle=False,
    batch_size=1,
    num_workers=0
)

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
    ImageDeformation = model.ImageTransform().cpu()
    MHCSA_vis = model.MHCSAB()
    MHCSA_ir = model.MHCSAB()
    fusion_module = model.FusionMoudle()

pretrain_dir = r"..."
pretrain_dir_base = os.path.join(pretrain_dir, "base" + ".pth")
pretrain_dir_vis_MFE = os.path.join(pretrain_dir, "vis_MFE" + ".pth")
pretrain_dir_ir_MFE = os.path.join(pretrain_dir, "ir_MFE" + ".pth")
pretrain_dir_PAFE = os.path.join(pretrain_dir, "PAFE" + ".pth")
pretrain_dir_VISDP = os.path.join(pretrain_dir, "VISDP" + ".pth")
pretrain_dir_IRDP = os.path.join(pretrain_dir, "IRDP" + ".pth")
pretrain_dir_MN_vis = os.path.join(pretrain_dir, "MN_vis" + ".pth")
pretrain_dir_MN_ir = os.path.join(pretrain_dir, "MN_ir" + ".pth")
pretrain_dir_MHCSA_vis = os.path.join(pretrain_dir, "MHCSA_vis" + ".pth")
pretrain_dir_MHCSA_ir = os.path.join(pretrain_dir, "MHCSA_ir" + ".pth")
pretrain_dir_fusion_module = os.path.join(pretrain_dir, "fusion_module" + ".pth")

utils.load_state_dir(base, pretrain_dir_base, device)
utils.load_state_dir(vis_MFE, pretrain_dir_vis_MFE, device)
utils.load_state_dir(ir_MFE, pretrain_dir_ir_MFE, device)
utils.load_state_dir(PAFE, pretrain_dir_PAFE, device)
utils.load_state_dir(VISDP, pretrain_dir_VISDP, device)
utils.load_state_dir(IRDP, pretrain_dir_IRDP, device)
utils.load_state_dir(MN_vis, pretrain_dir_MN_vis, device)
utils.load_state_dir(MN_ir, pretrain_dir_MN_ir, device)
utils.load_state_dir(MHCSA_vis, pretrain_dir_MHCSA_vis, device)
utils.load_state_dir(MHCSA_ir, pretrain_dir_MHCSA_ir, device)
utils.load_state_dir(fusion_module, pretrain_dir_fusion_module, device)

base.eval()
vis_MFE.eval()
ir_MFE.eval()
PAFE.eval()
VISDP.eval()
IRDP.eval()
MN_vis.eval()
MN_ir.eval()
MHCSA_vis.eval()
MHCSA_ir.eval()
fusion_module.eval()

for x in tqdm(test_data_iter):
    vis = x[0].to(device)  # vis
    ir = x[1].to(device)  # ir
    dir = x[2]

    with torch.no_grad():
        _, ir_d, _, _, _ = ImageDeformation(vis, ir)

        vis_1 = base(vis)
        ir_d_1 = base(ir_d)
        vis_fe = vis_MFE(vis_1)
        ir_d_fe = ir_MFE(ir_d_1)
        vis_f = PAFE(vis_1)
        ir_d_f = PAFE(ir_d_1)

        vis_e_f = MN_vis(vis_f)
        ir_d_e_f = MN_ir(ir_d_f)
        VISDP_vis_f, _ = VISDP(vis_e_f)
        IRDP_ir_d_f, _ = IRDP(ir_d_e_f)

        fixed_DP = VISDP_vis_f
        moving_DP = IRDP_ir_d_f
        moving_DP_lw = model.df_window_partition(moving_DP, args.args.large_w_size, args.args.small_w_size)
        fixed_DP_sw = model.window_partition(fixed_DP, args.args.small_w_size, args.args.small_w_size)
        correspondence_matrixs = model.CMAP(fixed_DP_sw, moving_DP_lw, MHCSA_vis, MHCSA_ir,
                                                     True)

        ir_d_f_sample = model.feature_reorganization(correspondence_matrixs, ir_d_fe)
        fusion_image_sample = fusion_module(vis_fe, ir_d_f_sample)

        file_name = dir[0].split("\\")[-1].split('.')[0]
        output_name = save_dir + "/" + file_name + ".png"
        out = torch.cat([vis, ir_d, fusion_image_sample], dim=2)
        utils.save_img(out, output_name)

        output_name = save_ird_dir + "/" + file_name + ".png"
        utils.save_img(ir_d, output_name)

        output_name = save_fusion_dir + "/" + file_name + ".png"
        out = fusion_image_sample
        utils.save_img(out, output_name)
