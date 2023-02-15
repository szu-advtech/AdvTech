import torch
from torch import nn


def block1(ics, ocs):
    return nn.Sequential(nn.BatchNorm2d(ics), nn.ReLU(True), nn.Conv2d(ics, ocs, kernel_size=3, padding=1))


def block2(ics, ocs):
    return nn.Sequential(nn.BatchNorm2d(ics), nn.ReLU(True), nn.Conv2d(ics, ocs, kernel_size=1), )


class CA(nn.Module):
    def __init__(self, ics):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.conv_block = nn.Sequential(
            nn.Conv2d(ics, int(ics // 4), kernel_size=1, stride=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(int(ics // 4), ics, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        out1 = self.conv_block(self.avg_pool(x))
        out2 = self.conv_block(self.max_pool(x))

        final_out = torch.add(out1, out2)
        final_out = torch.sigmoid(final_out)
        final_out = torch.mul(x, final_out)
        final_out = torch.add(x, final_out)
        return final_out


class SA(nn.Module):
    def __init__(self):
        super(SA, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out1 = torch.mean(x, dim=1, keepdim=True)  # 平均池化
        out2, _ = torch.max(x, dim=1, keepdim=True)  # 最大池化
        final_out = torch.cat([out1, out2], 1)
        final_out = self.conv(final_out)
        final_out = torch.sigmoid(final_out)
        final_out = torch.mul(x, final_out)
        final_out = torch.add(x, final_out)
        return final_out


class TSB(nn.Module):
    def __init__(self):
        super(TSB, self).__init__()

        h_net = []
        p_net = []
        for i in range(2):  # 每个流2个卷积块
            if i > 0:
                h_ics = 64 + i * 16
                p_ics = 64 + i * 16
            else:
                h_ics = 64
                p_ics = 64
            h_net.append(block1(h_ics, 16))
            p_net.append(block1(p_ics, 16))
        self.h_net = nn.ModuleList(h_net)
        self.p_net = nn.ModuleList(p_net)

        self.h_b1 = block1(96, 64)
        self.p_b1 = block1(96, 64)

    def forward(self, input1, input2):
        h_out1, p_out1 = self.h_net[0](input1), self.p_net[0](input2)

        h_out2, p_out2 = self.h_net[1](torch.cat([input1, h_out1], 1)), self.p_net[1](
            torch.cat([input2, p_out1], 1))

        h_final_out = self.h_b1(torch.cat([input1, h_out1, h_out2], 1))
        p_final_out = self.p_b1(torch.cat([input2, p_out1, p_out2], 1))

        h_final_out = torch.add(input1, h_final_out)
        p_final_out = torch.add(input2, p_final_out)
        return h_final_out, p_final_out


class TTSN(nn.Module):

    def __init__(self, v_n, n_n, TSB_num1, TSB_num2):
        super(TTSN, self).__init__()

        self.v_n = v_n
        self.TSB_num1 = TSB_num1
        self.TSB_num2 = TSB_num2
        # 可SFEN
        self.v_sfen_h = nn.Sequential(
            block2(v_n, 64),
            block2(64, 64)
        )

        self.v_sfen_p = nn.Sequential(
            block1(1, 64),
            block1(64, 64)
        )

        # 近SFEN
        self.n_sfen_h = nn.Sequential(
            block2(n_n, 64),
            block2(64, 64)
        )

        self.n_sfen_p = nn.Sequential(
            block1(1, 64),
            block1(64, 64)
        )

        # 可DFEN
        net = []
        for _ in range(TSB_num1):
            net.append(TSB())
        self.v_dfen = nn.ModuleList(net)

        # 近DFEN
        net = []
        for _ in range(TSB_num2):
            net.append(TSB())
        self.n_dfen = nn.ModuleList(net)

        # FFN
        self.v_ffn = block1(TSB_num1 * 64 * 2, TSB_num1 * 64 * 2)
        self.n_ffn = block1(TSB_num2 * 64 * 2, TSB_num2 * 64 * 2)

        # DRN
        self.drn1 = block2(TSB_num2 * 64 * 2 + TSB_num1 * 64 * 2, 128)
        self.drn2 = block2(128, 128)
        self.drn3 = nn.Conv2d(128, v_n + n_n, kernel_size=1, stride=1)

    def forward(self, input1, input2):

        input1_vl = input1[:, :self.v_n, :, :]
        input1_ni = input1[:, self.v_n:, :, :]

        # 可
        out_vl = self.v_sfen_h(input1_vl)
        out_pan_vl = self.v_sfen_p(input2)

        out_list_vl = []
        out_list_pan_vl = []
        for tsb in self.v_dfen:
            out_vl, out_pan_vl = tsb(out_vl, out_pan_vl)
            out_list_vl.append(out_vl)
            out_list_pan_vl.append(out_pan_vl)

        out_vl = out_list_vl[0]
        out_pan_vl = out_list_pan_vl[0]
        for i in range(self.TSB_num1 - 1):
            out_vl = torch.cat([out_vl, out_list_vl[i + 1]], 1)
            out_pan_vl = torch.cat([out_pan_vl, out_list_pan_vl[i + 1]], 1)

        out1 = torch.cat([out_vl, out_pan_vl], 1)
        out1 = self.v_ffn(out1)

        # 近
        out_ni = self.n_sfen_h(input1_ni)
        out_pan_ni = self.n_sfen_p(input2)

        out_list_ni = []
        out_list_pan_ni = []
        for tsb in self.n_dfen:
            out_ni, out_pan_ni = tsb(out_ni, out_pan_ni)
            out_list_ni.append(out_ni)
            out_list_pan_ni.append(out_pan_ni)

        out_ni = out_list_ni[0]
        out_pan_ni = out_list_pan_ni[0]
        for i in range(self.TSB_num2 - 1):
            out_ni = torch.cat([out_ni, out_list_ni[i + 1]], 1)
            out_pan_ni = torch.cat([out_pan_ni, out_list_pan_ni[i + 1]], 1)

        out2 = torch.cat([out_ni, out_pan_ni], 1)
        out2 = self.n_ffn(out2)

        out = torch.cat([out1, out2], 1)
        out = self.drn1(out)
        out = self.drn2(out)
        out = self.drn3(out)
        out = torch.add(out, input1)

        return out


class RHDB(nn.Module):
    def __init__(self):
        super(RHDB, self).__init__()

        h_net = []
        p_net = []
        for i in range(2):
            if i > 0:
                h_ics = 64 + i * 16 + 64 + (i - 1) * 16
                p_ics = 64 + i * 16 + 64 + (i - 1) * 16
            else:
                h_ics = 64
                p_ics = 64
            h_net.append(block1(h_ics, 16))
            p_net.append(block1(p_ics, 16))
        self.h_net = nn.ModuleList(h_net)
        self.p_net = nn.ModuleList(p_net)

        ca = []
        sa = []
        for _ in range(2):
            ca.append(CA(64))
            sa.append(SA())
        self.ca1 = nn.ModuleList(ca)
        self.sa1 = nn.ModuleList(sa)

        ca = []
        sa = []
        for _ in range(2 - 1):
            ca.append(CA(16))
            sa.append(SA())
        self.ca2 = nn.ModuleList(ca)
        self.sa2 = nn.ModuleList(sa)

        self.h_b1 = block1(176, 64)
        self.p_b1 = block1(176, 64)

    def forward(self, input1, input2):

        ca1, sa1 = [], []
        for i in range(2):
            ca1.append(self.ca1[i](input1))
            sa1.append(self.sa1[i](input2))

        h_out1, p_out1 = self.h_net[0](input1), self.p_net[0](input2)
        ca2 = self.ca2[0](h_out1)
        sa2 = self.sa2[0](p_out1)

        h_out2, p_out2 = self.h_net[1](torch.cat([input1, h_out1, sa1[0]], 1)), self.p_net[1](
            torch.cat([input2, p_out1, ca1[0]], 1))

        h_final_out = self.h_b1(torch.cat([input1, h_out1, h_out2, sa1[1], sa2], 1))
        p_final_out = self.p_b1(torch.cat([input2, p_out1, p_out2, ca1[1], ca2], 1))

        h_final_out = torch.add(input1, h_final_out)
        p_final_out = torch.add(input2, p_final_out)
        return h_final_out, p_final_out


class RHDN(nn.Module):

    def __init__(self, v_n, n_n, RHDB_num1, RHDB_num2):
        super(RHDN, self).__init__()

        self.v_n = v_n
        self.RHDB_num1 = RHDB_num1
        self.RHDB_num2 = RHDB_num2
        # 可SFEN
        self.v_sfen_h = nn.Sequential(
            block2(v_n, 64),
            block2(64, 64)
        )

        self.v_sfen_p = nn.Sequential(
            block1(1, 64),
            block1(64, 64)
        )

        # 近SFEN
        self.n_sfen_h = nn.Sequential(
            block2(n_n, 64),
            block2(64, 64)
        )

        self.n_sfen_p = nn.Sequential(
            block1(1, 64),
            block1(64, 64)
        )

        # 可DFEN
        net = []
        for _ in range(RHDB_num1):
            net.append(RHDB())
        self.v_dfen = nn.ModuleList(net)

        # 近DFEN
        net = []
        for _ in range(RHDB_num2):
            net.append(RHDB())
        self.n_dfen = nn.ModuleList(net)

        # FFN
        self.v_ffn = block1(RHDB_num1 * 64 * 2, RHDB_num1 * 64 * 2)
        self.n_ffn = block1(RHDB_num2 * 64 * 2, RHDB_num2 * 64 * 2)

        # DRN
        self.drn1 = block2(RHDB_num2 * 64 * 2 + RHDB_num1 * 64 * 2, 128)
        self.drn2 = block2(128, 128)
        self.drn3 = nn.Conv2d(128, v_n + n_n, kernel_size=1, stride=1)

    def forward(self, input1, input2):

        input1_vl = input1[:, :self.v_n, :, :]
        input1_ni = input1[:, self.v_n:, :, :]

        # 可
        out_vl = self.v_sfen_h(input1_vl)
        out_pan_vl = self.v_sfen_p(input2)

        out_list_vl = []
        out_list_pan_vl = []
        for rhdb in self.v_dfen:
            out_vl, out_pan_vl = rhdb(out_vl, out_pan_vl)
            out_list_vl.append(out_vl)
            out_list_pan_vl.append(out_pan_vl)

        out_vl = out_list_vl[0]
        out_pan_vl = out_list_pan_vl[0]
        for i in range(self.RHDB_num1 - 1):
            out_vl = torch.cat([out_vl, out_list_vl[i + 1]], 1)
            out_pan_vl = torch.cat([out_pan_vl, out_list_pan_vl[i + 1]], 1)

        out1 = torch.cat([out_vl, out_pan_vl], 1)
        out1 = self.v_ffn(out1)

        # 近
        out_ni = self.n_sfen_h(input1_ni)
        out_pan_ni = self.n_sfen_p(input2)

        out_list_ni = []
        out_list_pan_ni = []
        for rhdb in self.n_dfen:
            out_ni, out_pan_ni = rhdb(out_ni, out_pan_ni)
            out_list_ni.append(out_ni)
            out_list_pan_ni.append(out_pan_ni)

        out_ni = out_list_ni[0]
        out_pan_ni = out_list_pan_ni[0]
        for i in range(self.RHDB_num2 - 1):
            out_ni = torch.cat([out_ni, out_list_ni[i + 1]], 1)
            out_pan_ni = torch.cat([out_pan_ni, out_list_pan_ni[i + 1]], 1)

        out2 = torch.cat([out_ni, out_pan_ni], 1)
        out2 = self.n_ffn(out2)

        out = torch.cat([out1, out2], 1)
        out = self.drn1(out)
        out = self.drn2(out)
        out = self.drn3(out)
        out = torch.add(out, input1)

        return out


class Baseline(nn.Module):

    def __init__(self, in_c, RHDB_num):
        super(Baseline, self).__init__()

        self.ch = in_c
        self.RHDB_num = RHDB_num
        # SFEN
        self.SFEN = nn.Sequential(
            block2(in_c, 64),
            block2(64, 64)
        )
        self.PAN_SFEN = nn.Sequential(
            block1(1, 64),
            block1(64, 64)
        )
        # DFEN
        DFEN = []
        for _ in range(RHDB_num):
            DFEN.append(RHDB())
        self.DFEN = nn.ModuleList(DFEN)

        # FFN
        self.FFN = block1(RHDB_num * 64 * 2, RHDB_num * 64 * 2)

        # DRN
        self.drn1 = block2(RHDB_num * 64 * 2, 128)
        self.drn2 = block2(128, 128)
        self.drn3 = nn.Conv2d(128, in_c, kernel_size=1, stride=1)

    def forward(self, input1, input2):

        out = self.SFEN(input1)
        out_pan = self.PAN_SFEN(input2)

        out_list = []
        out_list_pan = []
        for rhdb in self.DFEN:
            out, out_pan = rhdb(out, out_pan)
            out_list.append(out)
            out_list_pan.append(out_pan)

        # 拼接
        out = out_list[0]
        out_pan = out_list_pan[0]
        for i in range(self.RHDB_num - 1):
            out = torch.cat([out, out_list[i + 1]], 1)
            out_pan = torch.cat([out_pan, out_list_pan[i + 1]], 1)

        out = torch.cat([out, out_pan], 1)
        out = self.FFN(out)

        out = self.drn1(out)
        out = self.drn2(out)
        out = self.drn3(out)
        out = torch.add(out, input1)

        return out


