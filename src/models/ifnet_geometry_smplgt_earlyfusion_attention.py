import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = torch.nn.Conv3d(
            in_channels, out_channels, 3, padding=1, padding_mode='replicate')
        self.attention = torch.nn.Conv3d(
            in_channels, out_channels, kernel_size=3, padding=1, padding_mode='replicate', bias=False)
        with torch.no_grad():
            self.attention.weight.copy_(
                torch.zeros_like(self.attention.weight))

    def forward(self, x):
        features = self.conv(x)
        attention_mask = torch.sigmoid(self.attention(x))
        return features * attention_mask


class IFNetGeometrySMPLGT_EarlyFusion_Attention(nn.Module):
    def __init__(self, hidden_dim=256):
        super(IFNetGeometrySMPLGT_EarlyFusion_Attention, self).__init__()

        self.conv_in_partial = nn.Conv3d(
            1, 16, 3, padding=1, padding_mode='replicate')  # out: 256 ->m.p. 128

        self.conv_in_smpl = nn.Conv3d(
            1, 4, 3, padding=1, padding_mode='replicate')  # out: 256 ->m.p. 128

        self.SA = SelfAttention(4, 4)
        self.conv_0_fusion = nn.Conv3d(16+4, 32, 3, padding=1,
                                       padding_mode='replicate')  # out: 128
        self.conv_0_1_fusion = nn.Conv3d(
            32, 32, 3, padding=1, padding_mode='replicate')  # out: 128 ->m.p. 64

        self.conv_0 = nn.Conv3d(32, 32, 3, padding=1,
                                padding_mode='replicate')  # out: 128
        self.conv_0_1 = nn.Conv3d(
            32, 32, 3, padding=1, padding_mode='replicate')  # out: 128 ->m.p. 64

        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1,
                                padding_mode='replicate')  # out: 64
        self.conv_1_1 = nn.Conv3d(
            64, 64, 3, padding=1, padding_mode='replicate')  # out: 64 -> mp 32

        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1,
                                padding_mode='replicate')  # out: 32
        self.conv_2_1 = nn.Conv3d(
            128, 128, 3, padding=1, padding_mode='replicate')  # out: 32 -> mp 16
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1,
                                padding_mode='replicate')  # out: 16
        self.conv_3_1 = nn.Conv3d(
            128, 128, 3, padding=1, padding_mode='replicate')  # out: 16 -> mp 8
        self.conv_4 = nn.Conv3d(128, 128, 3, padding=1,
                                padding_mode='replicate')  # out: 8
        self.conv_4_1 = nn.Conv3d(
            128, 128, 3, padding=1, padding_mode='replicate')  # out: 8

        feature_size = (1 + 32 + 32 + 64 + 128 + 128 + 128) * 7 + 3
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim * 2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool3d(2)

        self.partial_conv_in_bn = nn.InstanceNorm3d(16)
        self.smpl_conv_in_bn = nn.InstanceNorm3d(4)

        self.conv0_1_bn_fusion = nn.InstanceNorm3d(32)
        self.conv0_1_bn = nn.InstanceNorm3d(32)
        self.conv1_1_bn = nn.InstanceNorm3d(64)
        self.conv2_1_bn = nn.InstanceNorm3d(128)
        self.conv3_1_bn = nn.InstanceNorm3d(128)
        self.conv4_1_bn = nn.InstanceNorm3d(128)

        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        for x in range(3):
            for y in [-1, 1]:
                input = [0, 0, 0]
                input[x] = y * displacment
                displacments.append(input)

        self.displacments = displacments

    def forward(self, batch):
        p = batch.get('grid_coords')
        x = batch.get('inputs')
        x_smpl = batch.get('smpl_inputs')
        # x_smpl = batch.get('inputs')
        x = x.unsqueeze(1)
        x_smpl = x_smpl.unsqueeze(1)
        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        displacments = torch.Tensor(self.displacments).to(
            dtype=p.dtype, device=p.device)
        p = torch.cat([p + d for d in displacments], dim=2)

        # partial inputs feature extraction
        feature_0_partial = F.grid_sample(x, p, padding_mode='border')
        net_partial = self.actvn(self.conv_in_partial(x))
        net_partial = self.partial_conv_in_bn(net_partial)
        net_partial = self.maxpool(net_partial)  # out 64

        # smpl inputs feature extraction
        feature_0_smpl = F.grid_sample(x_smpl, p, padding_mode='border')
        net_smpl = self.actvn(self.conv_in_smpl(x_smpl))
        net_smpl = self.smpl_conv_in_bn(net_smpl)
        net_smpl = self.maxpool(net_smpl)  # out 64
        net_smpl = self.SA(net_smpl)

        # Feature fusion
        net = self.actvn(self.conv_0_fusion(
            torch.concat([net_partial, net_smpl], dim=1)))
        net = self.actvn(self.conv_0_1_fusion(net))
        net = self.conv0_1_bn_fusion(net)
        feature_1_fused = F.grid_sample(net, p, padding_mode='border')
        # net = self.maxpool(net)  # out 64

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feature_2 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)  # out 32

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_3 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)  # out 16

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_4 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)  # out 8

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_5 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)  # out 4

        net = self.actvn(self.conv_4(net))
        net = self.actvn(self.conv_4_1(net))
        net = self.conv4_1_bn(net)
        feature_6 = F.grid_sample(net, p, padding_mode='border')    # out 2

        # here every channel corresponse to one feature.

        features = torch.cat((feature_0_partial, feature_1_fused, feature_2, feature_3, feature_4, feature_5, feature_6),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features,
                                 (shape[0], shape[1] * shape[3], shape[4]))  # (B, featues_per_sample, samples_num)
        # (B, featue_size, samples_num)
        features = torch.cat((features, p_features), dim=1)

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.fc_out(net)
        out = net.squeeze(1)

        return out
