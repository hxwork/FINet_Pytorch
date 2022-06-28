import logging
import numpy as np
import torch
import torch.nn as nn
import transforms3d.euler as t3d
from common import quaternion, se3

from model.module import *


class FINet(nn.Module):

    def __init__(self, params):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        self.params = params
        self.num_iter = params.titer
        self.enc_config = params.enc_config
        self.reg_config = params.reg_config
        if self.enc_config["share_weights"]:
            self.encoder = FINetEncoder(self.enc_config)
        else:
            self.encoder = nn.ModuleList([FINetEncoder(self.enc_config) for _ in range(self.num_iter)])
        if self.enc_config["gfi"]:
            self.fusion = nn.ModuleList([FINetFusion(self.enc_config) for _ in range(self.num_iter)])
        self.regression = nn.ModuleList([FINetRegression(self.reg_config) for _ in range(self.num_iter)])

    def forward(self, data):
        endpoints = {}

        xyz_src = data["points_src"][:, :, :3]
        xyz_ref = data["points_ref"][:, :, :3]
        transform_gt = data["transform_gt"]
        pose_gt = data["pose_gt"]

        # init endpoints
        all_center = []
        all_R_feats = []
        all_t_feats = []
        all_dropout_R_feats = []
        all_dropout_t_feats = []
        all_transform_pair = []
        all_pose_pair = []
        all_xyz_src_t = [xyz_src]

        # init params
        B = xyz_src.size()[0]
        init_quat = t3d.euler2quat(0., 0., 0., "sxyz")
        init_quat = torch.from_numpy(init_quat).expand(B, 4)
        init_translate = torch.from_numpy(np.array([[0., 0., 0.]])).expand(B, 3)
        pose_pred = torch.cat((init_quat, init_translate), dim=1).float().cuda()  # B, 7
        transform_pred = quaternion.torch_quat2mat(pose_pred)

        # rename
        xyz_src_iter = xyz_src

        for i in range(self.num_iter):
            # encoder
            if self.enc_config["share_weights"]:
                encoder = self.encoder
            else:
                encoder = self.encoder[i]
            enc_input = torch.cat((xyz_src_iter.transpose(1, 2).detach(), xyz_ref.transpose(1, 2)), dim=0)  # 2B, C, N
            enc_feats = encoder(enc_input)
            src_enc_feats = [feat[:B, ...] for feat in enc_feats]
            ref_enc_feats = [feat[B:, ...] for feat in enc_feats]
            enc_src_R_feat = src_enc_feats[0]  # B, C
            enc_src_t_feat = src_enc_feats[1]  # B, C
            enc_ref_R_feat = ref_enc_feats[0]  # B, C
            enc_ref_t_feat = ref_enc_feats[1]  # B, C

            # fusion
            if self.enc_config["gfi"]:
                src_R_cat_feat = torch.cat((enc_src_R_feat, enc_ref_R_feat), dim=-1)  # B, 2C
                ref_R_cat_feat = torch.cat((enc_ref_R_feat, enc_src_R_feat), dim=-1)  # B, 2C
                src_t_cat_feat = torch.cat((enc_src_t_feat, enc_ref_t_feat), dim=-1)  # B, 2C
                ref_t_cat_feat = torch.cat((enc_ref_t_feat, enc_src_t_feat), dim=-1)  # B, 2C
                fusion_R_input = torch.cat((src_R_cat_feat, ref_R_cat_feat), dim=0)  # 2B, C
                fusion_t_input = torch.cat((src_t_cat_feat, ref_t_cat_feat), dim=0)  # 2B, C
                fusion_feats = self.fusion[i](fusion_R_input, fusion_t_input)
                src_fusion_feats = [feat[:B, ...] for feat in fusion_feats]
                ref_fusion_feats = [feat[B:, ...] for feat in fusion_feats]
                src_R_feat = src_fusion_feats[0]  # B, C
                src_t_feat = src_fusion_feats[1]  # B, C
                ref_R_feat = ref_fusion_feats[0]  # B, C
                ref_t_feat = ref_fusion_feats[1]  # B, C
            else:
                src_R_feat = enc_src_R_feat  # B, C
                src_t_feat = enc_src_t_feat  # B, C
                ref_R_feat = enc_ref_R_feat  # B, C
                ref_t_feat = enc_ref_t_feat  # B, C

            # R feats
            if self.reg_config["R_feats"] == "tr-tr":
                if self.reg_config["detach"]:
                    R_feats = torch.cat((src_t_feat.detach(), src_R_feat, ref_t_feat.detach(), ref_R_feat), dim=-1)  # B, 4C
                else:
                    R_feats = torch.cat((src_t_feat, src_R_feat, ref_t_feat, ref_R_feat), dim=-1)  # B, 4C

            elif self.reg_config["R_feats"] == "tr-r":
                if self.reg_config["detach"]:
                    R_feats = torch.cat((src_R_feat, src_t_feat.detach(), ref_R_feat), dim=-1)  # B, 3C
                else:
                    R_feats = torch.cat((src_R_feat, src_t_feat, ref_R_feat), dim=-1)  # B, 3C

            elif self.reg_config["R_feats"] == "r-r":
                R_feats = torch.cat((src_R_feat, ref_R_feat), dim=-1)  # B, 2C

            else:
                raise ValueError("Unknown R_feats order {}".format(self.reg_config["R_feats"]))

            # t feats
            if self.reg_config["t_feats"] == "tr-t":
                if self.reg_config["detach"]:
                    src_t_feats = torch.cat((src_t_feat, src_R_feat.detach(), ref_t_feat), dim=-1)  # B, 3C
                    ref_t_feats = torch.cat((ref_t_feat, ref_R_feat.detach(), src_t_feat), dim=-1)  # B, 3C
                else:
                    src_t_feats = torch.cat((src_t_feat, src_R_feat, ref_t_feat), dim=-1)  # B, 3C
                    ref_t_feats = torch.cat((ref_t_feat, ref_R_feat, src_t_feat), dim=-1)  # B, 3C

            elif self.reg_config["t_feats"] == "t-t":
                src_t_feats = torch.cat((src_t_feat, ref_t_feat), dim=-1)  # B, 2C
                ref_t_feats = torch.cat((ref_t_feat, src_t_feat), dim=-1)  # B, 2C

            else:
                raise ValueError("Unknown t_feats order {}".format(self.reg_config["t_feats"]))

            # regression
            if self.reg_config["reg_center"]:
                t_feats = torch.cat((src_t_feats, ref_t_feats), dim=0)  # 2B, 3C or 2B, 2C
                pred_quat, pred_center = self.regression[i](R_feats, t_feats)
                src_pred_center, ref_pred_center = torch.chunk(pred_center, 2, dim=0)
                pred_translate = ref_pred_center - src_pred_center
                pose_pred_iter = torch.cat((pred_quat, pred_translate), dim=-1)  # B, 7
            else:
                t_feats = src_t_feats  # B, 3C or B, 2C
                pred_quat, pred_translate = self.regression[i](R_feats, t_feats)
                pose_pred_iter = torch.cat((pred_quat, pred_translate), dim=-1)  # B, 7

            # extract features for compute loss on R and t features
            xyz_src_rotated = quaternion.torch_quat_rotate(xyz_src_iter.detach(), pose_pred_iter.detach())  # B, N, 3
            xyz_src_translated = xyz_src_iter.detach() + pose_pred_iter.detach()[:, 4:].unsqueeze(1)  # B, N, 3

            rotated_enc_input = torch.cat((xyz_src_rotated.transpose(1, 2).detach(), xyz_ref.transpose(1, 2)), dim=0)  # 2B, C, N
            rotated_enc_feats = encoder(rotated_enc_input)
            rotated_src_enc_feats = [feat[:B, ...] for feat in rotated_enc_feats]
            rotated_enc_src_R_feat = rotated_src_enc_feats[0]  # B, C
            rotated_enc_src_t_feat = rotated_src_enc_feats[1]  # B, C

            translated_enc_input = torch.cat((xyz_src_translated.transpose(1, 2).detach(), xyz_ref.transpose(1, 2)), dim=0)  # 2B, C, N
            translated_enc_feats = encoder(translated_enc_input)
            translated_src_enc_feats = [feat[:B, ...] for feat in translated_enc_feats]
            translated_enc_src_R_feat = translated_src_enc_feats[0]  # B, C
            translated_enc_src_t_feat = translated_src_enc_feats[1]  # B, C

            # dropout
            if self.enc_config["dropout"]:
                dropout_src_R_feat = src_enc_feats[8]  # B, C
                dropout_src_t_feat = src_enc_feats[9]  # B, C
                dropout_ref_R_feat = ref_enc_feats[8]  # B, C
                dropout_ref_t_feat = ref_enc_feats[9]  # B, C

            # do transform
            xyz_src_iter = quaternion.torch_quat_transform(pose_pred_iter, xyz_src_iter.detach())
            pose_pred = quaternion.torch_transform_pose(pose_pred.detach(), pose_pred_iter)
            transform_pred = quaternion.torch_quat2mat(pose_pred)

            # add endpoints at each iteration
            all_R_feats.append([enc_src_R_feat, rotated_enc_src_R_feat, translated_enc_src_R_feat])
            all_t_feats.append([enc_src_t_feat, rotated_enc_src_t_feat, translated_enc_src_t_feat])
            if self.enc_config["dropout"]:
                all_dropout_R_feats.append([dropout_src_R_feat, enc_src_R_feat, dropout_ref_R_feat, enc_ref_R_feat])
                all_dropout_t_feats.append([dropout_src_t_feat, enc_src_t_feat, dropout_ref_t_feat, enc_ref_t_feat])
            all_transform_pair.append([transform_gt, transform_pred])
            all_pose_pair.append([pose_gt, pose_pred])
            all_xyz_src_t.append(xyz_src_iter)

        # add endpoints finally
        endpoints["all_center"] = all_center
        endpoints["all_R_feats"] = all_R_feats
        endpoints["all_t_feats"] = all_t_feats
        endpoints["all_dropout_R_feats"] = all_dropout_R_feats
        endpoints["all_dropout_t_feats"] = all_dropout_t_feats
        endpoints["all_transform_pair"] = all_transform_pair
        endpoints["all_pose_pair"] = all_pose_pair
        endpoints["transform_pair"] = [transform_gt, transform_pred]
        endpoints["pose_pair"] = [pose_gt, pose_pred]
        endpoints["all_xyz_src_t"] = all_xyz_src_t

        return endpoints


def fetch_net(params):
    if params.net_type == "finet":
        net = FINet(params)

    else:
        raise NotImplementedError

    return net
