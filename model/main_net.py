import torch
from torch import nn
import cv2
import torch.nn.functional as F
import torchvision
import numpy as np
# import torchmaxflow
from torch.linalg import lstsq
import basicblock as B

class SpatialAttentionModule(nn.Module):
    def __init__(self, n_feats):
        super(SpatialAttentionModule, self).__init__()
        self.att1 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        att_map = torch.sigmoid(self.att2(self.relu(self.att1(x))))
        return att_map

class SAM(nn.Module): # Spatial Attention Module
    def __init__(self, bias=False):
        super(SAM, self).__init__()
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=self.bias)

    def forward(self, x):
        max = torch.max(x,1)[0].unsqueeze(1)
        avg = torch.mean(x,1).unsqueeze(1)
        concat = torch.cat((max,avg), dim=1)
        output = self.conv(concat)
        # output = F.sigmoid(output) * x 
        return F.sigmoid(output) 

class PALayer(nn.Module): # Pixel Attention Layer
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return y

class simam_module(torch.nn.Module): # Simple Attention Module
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return self.activaton(y)

class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()

        channel = 32
        self.conv20 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.conv21 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv20(x))  # Bsx32x64x64
        rs1 = self.conv21(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs
    
class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB_smilecr(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers):
        super(RDB_smilecr, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

        # In/Out conv
        self.in_conv = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=6, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.LFF(self.convs(x)) + x
        x = self.out_conv(x)
        return x

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

        # In/Out conv
        self.in_conv = nn.Conv2d(in_channels=13, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=13, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.LFF(self.convs(x)) + x
        x = self.out_conv(x)
        return x

class RDB_mask(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers):
        super(RDB_mask, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

        # In/Out conv
        self.in_conv = nn.Conv2d(in_channels=13, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.LFF(self.convs(x)) + x
        x = self.out_conv(x)
        return x

class RDB_mask_2(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers):
        super(RDB_mask_2, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

        # In/Out conv
        self.in_conv = nn.Conv2d(in_channels=13, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.in_conv(x)
        x = self.LFF(self.convs(x)) + x
        x = self.out_conv(x)
        x = self.sigmoid(x)  # Apply sigmoid to output
        return x

class RDB_mask_2_smilecr(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers):
        super(RDB_mask_2_smilecr, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

        # In/Out conv
        self.in_conv = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.in_conv(x)
        x = self.LFF(self.convs(x)) + x
        x = self.out_conv(x)
        x = self.sigmoid(x)  # Apply sigmoid to output
        return x

class ProximalBlock(nn.Module):
    def __init__(self, out_channels):
        super(ProximalBlock, self).__init__()
        self.out_channels = out_channels
        self.proximal = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels=128, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        output = self.proximal(x)
        if self.out_channels == 1:
            output = self.sigmoid(output)
        return output

class RPCA_Block(nn.Module): #multi batchsize + check parameters + Slack variables on M
    def __init__(self):
        super(RPCA_Block, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)
        self.rho = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.eta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

        self.Proximal_P = RDB(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_mask_2(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, L3, L4, V, T, Y, W, X_hat): # Y is Y | W is F_att(Y)
        
        one_minus_M = self.ones - M
        one_minus_M_sq = torch.mul(one_minus_M, one_minus_M)
        Psi_k = one_minus_M * (Y - M * C) + \
                self.mu * one_minus_M_sq * Y + \
                self.gamma * X_hat + \
                self.alpha * G - \
                L1
        
        numerator = torch.mul(W, self.tensor_product(L, R)) + Psi_k
        denominator = (1+self.mu) * one_minus_M_sq + W + torch.abs(self.gamma) + torch.abs(self.alpha)
        X_k = torch.div(numerator, denominator)
        
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M

        X_minus_C = X_k - C
        X_minus_Y = X_k - Y

        # 1. Compute the quadratic weight tensor A_k
        A_k = torch.mul(X_minus_C, X_minus_C) + torch.abs(self.mu) * torch.mul(X_minus_Y, X_minus_Y) + torch.abs(self.beta) + torch.abs(self.eta) + torch.abs(self.rho)

        # 2. Compute the linear weight tensor Upsilon_k
        Upsilon_k = torch.mul(X_minus_C, X_minus_Y) + torch.abs(self.mu) * torch.mul(X_minus_Y, X_minus_Y) + \
                    torch.abs(self.beta) * H - L2 + torch.abs(self.rho) * V - L3 + torch.abs(self.eta) * (self.ones - T) - L4

        # 3. Update M_k using element-wise division
        X_input = torch.div(Upsilon_k, A_k + 1e-6)  # Add a small constant to avoid division by zero
        tau = self.lamb / (A_k + 1e-6)  # Compute the threshold for soft-thresholding

        # 4. Apply element-wise Soft-Thresholding: sign(x) * max(|x| - tau, 0)
        
        M_k = torch.sign(X_input) * nn.functional.relu(torch.abs(X_input) - tau)
        M_k = torch.clamp(M_k, min=0, max=1)  # Ensure M_k is in the range [0, 1]

        # Update S and T
        
        S_k = torch.clamp((L3 + torch.abs(self.rho) * M_k) / self.rho, min=0)  # Ensure S_k >= 0
        T_k = torch.clamp((torch.abs(self.eta) * (self.ones - M_k) - L4) / self.eta , min=0)  # Ensure T_k >= 0

        # Update C
        C_k = torch.div(Y - torch.mul(self.ones - M_k, X_k) , M_k + 1e-6)
        C_k = torch.clamp(C_k, min=0, max=1)  # Ensure C_k is in the range [0, 1]

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        L0_k = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        L3_k = L3 + torch.abs(self.rho) * (X_k - S_k)
        L4_k = L4 + torch.abs(self.eta) * (M_k + T_k - self.ones)

        # # check if X_k, M_k, C_k are nan
        # if torch.isnan(X_k).any() or torch.isnan(M_k).any() or torch.isnan(C_k).any():
        #     print("X_k is nan")
        #     print("M_k is nan")
        #     print("C_k is nan")
        #     a = 1/0
        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, L3_k, L4_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta), S_k, T_k

class SAR_ULRTC_CR(nn.Module): # multi-batchsize + check the parameters + abs parameters + Slack variables on M
    def __init__(self, N_iter):
        super(SAR_ULRTC_CR, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(13)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 13, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block())
        self.network = nn.ModuleList(blocks_list)

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI

        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Tau
        L4 = torch.zeros(X.size(), device=torch.device('cuda')) # zeta
        V = torch.zeros(X.size(), device=torch.device('cuda')) # V for M >=0
        T = torch.zeros(X.size(), device=torch.device('cuda')) # T for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 13, 10, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 10, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, L3, L4, lamb, gamma, mu, alpha, beta, V, T] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, L3, L4, V, T, RSI, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, L3, L4, cs_comp, lamb, gamma, mu, alpha, beta, V, T])

        return layers


class SAR_ULRTC_CR_SMILECR(nn.Module): # multi-batchsize + check the parameters + abs parameters + Slack variables on M
    def __init__(self, N_iter):
        super(SAR_ULRTC_CR_SMILECR, self).__init__()

        # Number of unrolled iterations
        self.N_iter = N_iter

        # Weight for RLS
        self.att_module = SpatialAttentionModule(6)
        # CS modules
        self.dup_conv = nn.Conv2d(3, 6, kernel_size=1, bias=False)
        # Unrolled network
        blocks_list = []
        for i in range(self.N_iter):
            blocks_list.append(RPCA_Block_smilecr())
        self.network = nn.ModuleList(blocks_list)
        # self.network = RPCA_Block_new_2_SAR_Trans_3()
        # self.loss_fn = nn.L1Loss()

    def forward(self, RSI, SAR, trans):

        # Observation
        X = RSI
        
        cs_comp = self.dup_conv(trans)  # Bsx13x256x256

        # Weight W for reweigted least-squares
        W = self.att_module(X)

        # Optimal variables

        L0 = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1 = torch.zeros(X.size(), device=torch.device('cuda')) # Gamma
        L2 = torch.zeros(X.size(), device=torch.device('cuda')) # Phi
        L3 = torch.zeros(X.size(), device=torch.device('cuda')) # Tau
        L4 = torch.zeros(X.size(), device=torch.device('cuda')) # zeta
        S = torch.zeros(X.size(), device=torch.device('cuda')) # U for M >=0
        T = torch.zeros(X.size(), device=torch.device('cuda')) # V for 1-M >=0

        M  = torch.zeros((X.shape[0],1,X.shape[2],X.shape[3]), device=torch.device('cuda')) # M
        C  = torch.zeros(X.size(), device=torch.device('cuda')) # C
        
        G  = torch.zeros(X.size(), device=torch.device('cuda')) # G for X
        H  = torch.zeros(X.size(), device=torch.device('cuda')) # H for M
        # Init L/R
        L = torch.ones((X.shape[0], 6, 4, X.shape[-1]), device=torch.device('cuda')) / 1e2
        R = torch.ones((X.shape[0], 4, X.shape[-2], X.shape[-1]), device=torch.device('cuda')) / 1e2

        # Main net
        layers = []
        for i in range(0, self.N_iter):
            [L, R, X, M, C, G, H, L0, L1, L2, L3, L4, lamb, gamma, mu, alpha, beta, S, T] = self.network[i](L, R, X, M, C, G, H, L0, L1, L2, L3, L4, S, T, RSI, W, cs_comp)
            layers.append([L, R, X, M, C, G, H, L0, L1, L2, L3, L4, cs_comp, lamb, gamma, mu, alpha, beta, S, T])

        return layers

class RPCA_Block_smilecr(nn.Module): #multi batchsize + check parameters + Slack variables on M
    def __init__(self):
        super(RPCA_Block_smilecr, self).__init__()

        self.lamb  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.gamma = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.mu    = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.alpha = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.beta  = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.ones = nn.Parameter(torch.ones(1), requires_grad=False)
        self.rho = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)
        self.eta = nn.Parameter(torch.ones(1)*0.001, requires_grad=True)

        self.Proximal_P = RDB_smilecr(growRate0=64, growRate=32, nConvLayers=8)
        self.Proximal_Q = RDB_mask_2_smilecr(growRate0=64, growRate=32, nConvLayers=6)

    def tensor_product(self, L, R): # vectorized version - multi batchsize
        # Apply FFT across all batches (along the last dimension)
        Lf = torch.fft.fft(L, n=L.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Rf = torch.fft.fft(R, n=R.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Reshape and permute for batch matrix multiplication
        # We need to handle each frequency component separately
        batch_size = L.shape[0]
        n_freq = L.shape[-1] // 2 + 1  # Number of frequency components for real FFT
        
        # Initialize result tensor
        Gf_shape = (batch_size, L.shape[1], R.shape[2], n_freq)
        Gf = torch.zeros(Gf_shape, dtype=torch.complex64, device=L.device)
        
        # For each frequency component, perform matrix multiplication
        # This can be done with batch_matmul but we need to reorder the dimensions first
        Lf_permuted = Lf.permute(0, 3, 1, 2)  # Shape: (B, 256, 13, 10)
        Rf_permuted = Rf.permute(0, 3, 1, 2)  # Shape: (B, 256, 10, 256)
        
        # Now we can use bmm for each frequency component
        for f in range(n_freq):
            Lf_f = Lf_permuted[:, f]  # Shape: (B, 13, 10)
            Rf_f = Rf_permuted[:, f]  # Shape: (B, 10, 256)
            Gf_f = torch.bmm(Lf_f, Rf_f)  # Shape: (B, 13, 256)
            Gf[:, :, :, f] = Gf_f
        
        # Permute back to expected order for irfft
        Gf = Gf.permute(0, 1, 2, 3)  # Shape: (B, 13, 256, 256//2+1)
        
        # Apply inverse FFT along the last dimension
        G_result = torch.fft.irfft(Gf, n=R.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        
        return G_result

    def decom_solution(self, L_k, R_k, X_k): # multi batchsize
        batch_size = X_k.shape[0]
        
        # Apply FFT across all batches (along the last dimension)
        X = torch.fft.fft(X_k, n=X_k.shape[-1], dim=3)  # Shape: (B, 13, 256, 256)
        L = torch.fft.fft(L_k, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        R = torch.fft.fft(R_k, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        # Initialize output tensors
        Li_list = []
        Ri_list = []
        
        # Process each batch separately
        for b in range(batch_size):
            # Permute to move the frequency dimension first
            X_b = X[b].permute(2, 0, 1)  # Shape: (256, 13, 256)
            L_b = L[b].permute(2, 0, 1)  # Shape: (256, 13, 10)
            R_b = R[b].permute(2, 0, 1)  # Shape: (256, 10, 256)
            
            # Calculate Li
            R_b_conj_T = torch.transpose(torch.conj(R_b), 1, 2)  # Shape: (256, 256, 10)
            R_R_term = torch.matmul(R_b, R_b_conj_T)  # Shape: (256, 10, 10)
            R_pinv = torch.linalg.pinv(R_R_term, rcond=1e-4)  # Shape: (256, 10, 10)
            X_R_term = torch.matmul(X_b, R_b_conj_T)  # Shape: (256, 13, 10)
            Li_b = torch.matmul(X_R_term, R_pinv).permute(1, 2, 0)  # Shape: (13, 10, 256)
            
            # Calculate Ri
            L_b_conj_T = torch.transpose(torch.conj(L_b), 1, 2)  # Shape: (256, 10, 13)
            L_L_term = torch.matmul(L_b_conj_T, L_b)  # Shape: (256, 10, 10)
            L_pinv = torch.linalg.pinv(L_L_term, rcond=1e-4)  # Shape: (256, 10, 10)
            L_pinv_L = torch.matmul(L_pinv, L_b_conj_T)  # Shape: (256, 10, 13)
            Ri_b = torch.matmul(L_pinv_L, X_b).permute(1, 2, 0)  # Shape: (10, 256, 256)
            
            # Append to list
            Li_list.append(Li_b)
            Ri_list.append(Ri_b)
        
        # Stack the results along the batch dimension
        Li_batched = torch.stack(Li_list, dim=0)  # Shape: (B, 13, 10, 256)
        Ri_batched = torch.stack(Ri_list, dim=0)  # Shape: (B, 10, 256, 256)
        
        # Apply inverse FFT along the last dimension
        Li_result = torch.fft.irfft(Li_batched, n=L_k.shape[-1], dim=3)  # Shape: (B, 13, 10, 256)
        Ri_result = torch.fft.irfft(Ri_batched, n=R_k.shape[-1], dim=3)  # Shape: (B, 10, 256, 256)
        
        return Li_result, Ri_result
    
    # def forward(self, L, R, C, E, T, P, Q, L1, L2, L3, Omega, W, Omega_C): 
    def forward(self, L, R, X, M, C, G, H, L0, L1, L2, L3, L4, V, T, Y, W, X_hat): # Y is Y | W is F_att(Y)
         
        one_minus_M = self.ones - M
        one_minus_M_sq = torch.mul(one_minus_M, one_minus_M)
        Psi_k = one_minus_M * (Y - M * C) + \
                self.mu * one_minus_M_sq * Y + \
                self.gamma * X_hat + \
                self.alpha * G - \
                L1
        
        numerator = torch.mul(W, self.tensor_product(L, R)) + Psi_k
        denominator = (1+self.mu) * one_minus_M_sq + W + torch.abs(self.gamma) + torch.abs(self.alpha)
        X_k = torch.div(numerator, denominator)

        # Update L and R
        L_k, R_k = self.decom_solution(L, R, X_k)

        # Update M

        X_minus_C = X_k - C
        X_minus_Y = X_k - Y

        # 1. Compute the quadratic weight tensor A_k
        A_k = torch.mul(X_minus_C, X_minus_C) + torch.abs(self.mu) * torch.mul(X_minus_Y, X_minus_Y) + torch.abs(self.beta) + torch.abs(self.eta) + torch.abs(self.rho)

        # 2. Compute the linear weight tensor Upsilon_k
        Upsilon_k = torch.mul(X_minus_C, X_minus_Y) + torch.abs(self.mu) * torch.mul(X_minus_Y, X_minus_Y) + \
                    torch.abs(self.beta) * H - L2 + torch.abs(self.rho) * V - L3 + torch.abs(self.eta) * (self.ones - T) - L4

        # 3. Update M_k using element-wise division
        X_input = torch.div(Upsilon_k, A_k + 1e-6)  # Add a small constant to avoid division by zero
        tau = self.lamb / (A_k + 1e-6)  # Compute the threshold for soft-thresholding

        # 4. Apply element-wise Soft-Thresholding: sign(x) * max(|x| - tau, 0)
        
        M_k = torch.sign(X_input) * nn.functional.relu(torch.abs(X_input) - tau)
        M_k = torch.clamp(M_k, min=0, max=1)  # Ensure M_k is in the range [0, 1]
      
        # Update S and T
        
        S_k = torch.clamp((L3 + torch.abs(self.rho) * M_k) / self.rho, min=0)  # Ensure S_k >= 0
        T_k = torch.clamp((torch.abs(self.eta) * (self.ones - M_k) - L4) / self.eta , min=0)  # Ensure T_k >= 0
      
        # Update C
        C_k = torch.div(Y - torch.mul(self.ones - M_k, X_k) , M_k + 1e-6)
        C_k = torch.clamp(C_k, min=0, max=1)  # Ensure C_k is in the range [0, 1]

        # Update P
        G_k = self.Proximal_P(X_k + L1 / (torch.abs(self.alpha) + 1e-6))

        # Update Q
        H_k = self.Proximal_Q(M_k + L2 / (torch.abs(self.beta) + 1e-6))

        # Update Lambda
        L0_k = torch.zeros(X.size(), device=torch.device('cuda')) # Lambda
        L1_k = L1 + torch.abs(self.alpha) * (X_k - G_k)
        L2_k = L2 + torch.abs(self.beta) * (M_k - H_k)
        L3_k = L3 + torch.abs(self.rho) * (X_k - S_k)
        L4_k = L4 + torch.abs(self.eta) * (M_k + T_k - self.ones)

        # # check if X_k, M_k, C_k are nan
        # if torch.isnan(X_k).any() or torch.isnan(M_k).any() or torch.isnan(C_k).any():
        #     print("X_k is nan")
        #     print("M_k is nan")
        #     print("C_k is nan")
        #     a = 1/0
        return L_k, R_k, X_k, M_k, C_k, G_k, H_k, L0_k, L1_k, L2_k, L3_k, L4_k, torch.abs(self.lamb), torch.abs(self.gamma), torch.abs(self.mu), torch.abs(self.alpha), torch.abs(self.beta), S_k, T_k
