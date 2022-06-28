import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================================================


class FINetEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config["lrelu"]:
            self.relu = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

        # R
        self.R_block1 = nn.Sequential(nn.Conv1d(3, 64, 1, bias=False), nn.BatchNorm1d(64), self.relu)
        self.R_block2 = nn.Sequential(nn.Conv1d(64, 64, 1, bias=False), nn.BatchNorm1d(64), self.relu)
        if self.config["pfi"][0] != "none":
            self.R_block3 = nn.Sequential(nn.Conv1d(128, 128, 1, bias=False), nn.BatchNorm1d(128), self.relu)
        else:
            self.R_block3 = nn.Sequential(nn.Conv1d(64, 128, 1, bias=False), nn.BatchNorm1d(128), self.relu)
        self.R_block4 = nn.Sequential(nn.Conv1d(128, 256, 1, bias=False), nn.BatchNorm1d(256), self.relu)
        if self.config["pfi"][1] != "none":
            self.R_block5 = nn.Sequential(nn.Conv1d(512, 512, 1, bias=False), nn.BatchNorm1d(512), self.relu)
        else:
            self.R_block5 = nn.Sequential(nn.Conv1d(256, 512, 1, bias=False), nn.BatchNorm1d(512), self.relu)

        # t
        self.t_block1 = nn.Sequential(nn.Conv1d(3, 64, 1, bias=False), nn.BatchNorm1d(64), self.relu)
        self.t_block2 = nn.Sequential(nn.Conv1d(64, 64, 1, bias=False), nn.BatchNorm1d(64), self.relu)
        if self.config["pfi"][0] != "none":
            self.t_block3 = nn.Sequential(nn.Conv1d(128, 128, 1, bias=False), nn.BatchNorm1d(128), self.relu)
        else:
            self.t_block3 = nn.Sequential(nn.Conv1d(64, 128, 1, bias=False), nn.BatchNorm1d(128), self.relu)
        self.t_block4 = nn.Sequential(nn.Conv1d(128, 256, 1, bias=False), nn.BatchNorm1d(256), self.relu)
        if self.config["pfi"][1] != "none":
            self.t_block5 = nn.Sequential(nn.Conv1d(512, 512, 1, bias=False), nn.BatchNorm1d(512), self.relu)
        else:
            self.t_block5 = nn.Sequential(nn.Conv1d(256, 512, 1, bias=False), nn.BatchNorm1d(512), self.relu)

    def forward(self, x, mask=None):
        B, C, N = x.size()
        if self.config["dropout"]:
            if self.training:
                rand_mask = torch.rand((B, 1, N), device=x.device) > self.config["dropout_ratio"]
            else:
                rand_mask = 1

        # R stage1
        R_feat_output1 = self.R_block1(x)
        if mask is not None:
            R_feat_output1 = R_feat_output1 * mask
        R_feat_output2 = self.R_block2(R_feat_output1)
        if mask is not None:
            R_feat_output2 = R_feat_output2 * mask
        R_feat_glob2 = torch.max(R_feat_output2, dim=-1, keepdim=True)[0]

        # t stage1
        t_feat_output1 = self.t_block1(x)
        if mask is not None:
            t_feat_output1 = t_feat_output1 * mask
        t_feat_output2 = self.t_block2(t_feat_output1)
        if mask is not None:
            t_feat_output2 = t_feat_output2 * mask
        t_feat_glob2 = torch.max(t_feat_output2, dim=-1, keepdim=True)[0]

        # exchange1
        if self.config["pfi"][0] == "cross":
            src_R_feat_glob2, ref_R_feat_glob2 = torch.chunk(R_feat_glob2, 2, dim=0)
            src_t_feat_glob2, ref_t_feat_glob2 = torch.chunk(t_feat_glob2, 2, dim=0)
            exchange_R_feat = torch.cat((ref_R_feat_glob2.repeat(1, 1, N), src_R_feat_glob2.repeat(1, 1, N)), dim=0)
            exchange_t_feat = torch.cat((ref_t_feat_glob2.repeat(1, 1, N), src_t_feat_glob2.repeat(1, 1, N)), dim=0)
            if self.config["detach"]:
                exchange_R_feat = torch.cat((R_feat_output2, exchange_R_feat.detach()), dim=1)
                exchange_t_feat = torch.cat((t_feat_output2, exchange_t_feat.detach()), dim=1)
            else:
                exchange_R_feat = torch.cat((R_feat_output2, exchange_R_feat), dim=1)
                exchange_t_feat = torch.cat((t_feat_output2, exchange_t_feat), dim=1)
        elif self.config["pfi"][0] == "self":
            if self.config["detach"]:
                exchange_R_feat = torch.cat((R_feat_output2, t_feat_glob2.repeat(1, 1, N).detach()), dim=1)
                exchange_t_feat = torch.cat((t_feat_output2, R_feat_glob2.repeat(1, 1, N).detach()), dim=1)
            else:
                exchange_R_feat = torch.cat((R_feat_output2, t_feat_glob2.repeat(1, 1, N)), dim=1)
                exchange_t_feat = torch.cat((t_feat_output2, R_feat_glob2.repeat(1, 1, N)), dim=1)
        else:
            exchange_R_feat = R_feat_output2
            exchange_t_feat = t_feat_output2

        # R stage2
        R_feat_output3 = self.R_block3(exchange_R_feat)
        if mask is not None:
            R_feat_output3 = R_feat_output3 * mask
        R_feat_output4 = self.R_block4(R_feat_output3)
        if mask is not None:
            R_feat_output4 = R_feat_output4 * mask
        R_feat_glob4 = torch.max(R_feat_output4, dim=-1, keepdim=True)[0]

        # t stage2
        t_feat_output3 = self.t_block3(exchange_t_feat)
        if mask is not None:
            t_feat_output3 = t_feat_output3 * mask
        t_feat_output4 = self.t_block4(t_feat_output3)
        if mask is not None:
            t_feat_output4 = t_feat_output4 * mask
        t_feat_glob4 = torch.max(t_feat_output4, dim=-1, keepdim=True)[0]

        # exchange2
        if self.config["pfi"][1] == "cross":
            src_R_feat_glob4, ref_R_feat_glob4 = torch.chunk(R_feat_glob4, 2, dim=0)
            src_t_feat_glob4, ref_t_feat_glob4 = torch.chunk(t_feat_glob4, 2, dim=0)
            exchange_R_feat = torch.cat((ref_R_feat_glob4.repeat(1, 1, N), src_R_feat_glob4.repeat(1, 1, N)), dim=0)
            exchange_t_feat = torch.cat((ref_t_feat_glob4.repeat(1, 1, N), src_t_feat_glob4.repeat(1, 1, N)), dim=0)
            if self.config["detach"]:
                exchange_R_feat = torch.cat((R_feat_output4, exchange_R_feat.detach()), dim=1)
                exchange_t_feat = torch.cat((t_feat_output4, exchange_t_feat.detach()), dim=1)
            else:
                exchange_R_feat = torch.cat((R_feat_output4, exchange_R_feat), dim=1)
                exchange_t_feat = torch.cat((t_feat_output4, exchange_t_feat), dim=1)
        elif self.config["pfi"][1] == "self":
            if self.config["detach"]:
                exchange_R_feat = torch.cat((R_feat_output4, t_feat_glob4.repeat(1, 1, N).detach()), dim=1)
                exchange_t_feat = torch.cat((t_feat_output4, R_feat_glob4.repeat(1, 1, N).detach()), dim=1)
            else:
                exchange_R_feat = torch.cat((R_feat_output4, t_feat_glob4.repeat(1, 1, N)), dim=1)
                exchange_t_feat = torch.cat((t_feat_output4, R_feat_glob4.repeat(1, 1, N)), dim=1)
        else:
            exchange_R_feat = R_feat_output4
            exchange_t_feat = t_feat_output4

        # R stage3
        R_feat_output5 = self.R_block5(exchange_R_feat)
        if mask is not None:
            R_feat_output5 = R_feat_output5 * mask

        # t stage3
        t_feat_output5 = self.t_block5(exchange_t_feat)
        if mask is not None:
            t_feat_output5 = t_feat_output5 * mask

        # final
        R_final_feat_output = torch.cat((R_feat_output1, R_feat_output2, R_feat_output3, R_feat_output4, R_feat_output5), dim=1)
        t_final_feat_output = torch.cat((t_feat_output1, t_feat_output2, t_feat_output3, t_feat_output4, t_feat_output5), dim=1)

        R_final_glob_feat, R_final_glob_idx = torch.max(R_final_feat_output, dim=-1, keepdim=False)
        t_final_glob_feat, t_final_glob_idx = torch.max(t_final_feat_output, dim=-1, keepdim=False)
        R_point_feat_value = R_final_feat_output.norm(dim=1)
        t_point_feat_value = t_final_feat_output.norm(dim=1)

        if self.config["dropout"]:
            R_final_feat_dropout = R_final_feat_output * rand_mask
            R_final_feat_dropout = torch.max(R_final_feat_dropout, dim=-1, keepdim=False)[0]

            t_final_feat_dropout = t_final_feat_output * rand_mask
            t_final_feat_dropout = torch.max(t_final_feat_dropout, dim=-1, keepdim=False)[0]

            return [
                R_final_glob_feat, t_final_glob_feat, R_final_glob_idx, t_final_glob_idx, R_point_feat_value, t_point_feat_value,
                R_final_feat_output, t_final_feat_output, R_final_feat_dropout, t_final_feat_dropout
            ]
        else:
            return [
                R_final_glob_feat, t_final_glob_feat, R_final_glob_idx, t_final_glob_idx, R_point_feat_value, t_point_feat_value,
                R_final_feat_output, t_final_feat_output
            ]


class FINetFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config["lrelu"]:
            self.relu = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        # R
        self.R_block1 = nn.Sequential(nn.Linear(2048, 2048), nn.BatchNorm1d(2048), self.relu)
        self.R_block2 = nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), self.relu)
        self.R_block3 = nn.Sequential(nn.Linear(1024, 1024), nn.BatchNorm1d(1024), self.relu)

        # t
        self.t_block1 = nn.Sequential(nn.Linear(2048, 2048), nn.BatchNorm1d(2048), self.relu)
        self.t_block2 = nn.Sequential(nn.Linear(2048, 1024), nn.BatchNorm1d(1024), self.relu)
        self.t_block3 = nn.Sequential(nn.Linear(1024, 1024), nn.BatchNorm1d(1024), self.relu)

    def forward(self, R_feat, t_feat):
        # R
        fuse_R_feat = self.R_block1(R_feat)
        fuse_R_feat = self.R_block2(fuse_R_feat)
        fuse_R_feat = self.R_block3(fuse_R_feat)
        # t
        fuse_t_feat = self.t_block1(t_feat)
        fuse_t_feat = self.t_block2(fuse_t_feat)
        fuse_t_feat = self.t_block3(fuse_t_feat)

        return [fuse_R_feat, fuse_t_feat]


class FINetRegression(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if self.config["R_feats"] in ["tr-tr"]:
            R_in_channel = 4096
        elif self.config["R_feats"] in ["tr-r"]:
            R_in_channel = 3072
        elif self.config["R_feats"] in ["r-r"]:
            R_in_channel = 2048
        else:
            raise ValueError("Unknown R_feats order {}".format(self.config["R_feats"]))

        if self.config["t_feats"] in ["tr-t"]:
            t_in_channel = 3072
        elif self.config["t_feats"] in ["t-t"]:
            t_in_channel = 2048
        else:
            raise ValueError("Unknown t_feats order {}".format(self.config["t_feats"]))

        if self.config["lrelu"]:
            self.relu = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)

        self.R_net = nn.Sequential(
            # block 1
            nn.Linear(R_in_channel, 2048),
            nn.BatchNorm1d(2048),
            self.relu,
            # block 2
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            self.relu,
            # block 3
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            self.relu,
            # block 4
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.relu,
            # final fc
            nn.Linear(256, 4),
        )

        self.t_net = nn.Sequential(
            # block 1
            nn.Linear(t_in_channel, 2048),
            nn.BatchNorm1d(2048),
            self.relu,
            # block 2
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            self.relu,
            # block 3
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            self.relu,
            # block 4
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.relu,
            # final fc
            nn.Linear(256, 3),
        )

    def forward(self, R_feat, t_feat):

        pred_quat = self.R_net(R_feat)
        pred_quat = F.normalize(pred_quat, dim=1)
        pred_translate = self.t_net(t_feat)

        return [pred_quat, pred_translate]
