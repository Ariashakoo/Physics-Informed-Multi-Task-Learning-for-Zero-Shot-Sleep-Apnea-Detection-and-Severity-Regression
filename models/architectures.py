import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        w = x.mean(dim=-1)
        w = self.fc(w).unsqueeze(-1)
        return x * w

class ResBlock1D_SE(nn.Module):
    def __init__(self, in_ch, out_ch, stride=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock1D(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return self.relu(out + res)

class TemporalAttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1)
        )
    def forward(self, x):
        x_t = x.permute(0, 2, 1)
        w = F.softmax(self.attn(x_t), dim=1)
        return (x_t * w).sum(dim=1)

class PIMTL_CPC(nn.Module):
    def __init__(self, cpc_dim=3, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        self.stage1 = ResBlock1D_SE(64, 128, stride=2)
        self.stage2 = ResBlock1D_SE(128, 256, stride=2)
        self.stage3 = ResBlock1D_SE(256, 256, stride=2)
        self.pool = TemporalAttentionPooling(256)
        self.dropout = nn.Dropout(dropout)
        
        self.apnea_head = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(128, 2)
        )
        self.ahi_head = nn.Sequential(
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(64, 1)
        )
        self.cpc_head = nn.Sequential(
            nn.Linear(256, 64), nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(64, cpc_dim)
        )

    def forward(self, ecg):
        x = self.stem(ecg)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        z = self.pool(x)
        z = self.dropout(z)
        return self.apnea_head(z), self.ahi_head(z), self.cpc_head(z)

class PIMTL_CPC_Loss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        if class_weights is not None:
            cw_tensor = class_weights.clone().detach() if isinstance(class_weights, torch.Tensor) else torch.tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', cw_tensor)
        else:
            self.class_weights = None

    def forward(self, apnea_logits, ahi_pred, cpc_pred, apnea_target, ahi_target, cpc_target):
        l_apnea = F.cross_entropy(apnea_logits, apnea_target.long(), weight=self.class_weights)
        l_ahi = F.smooth_l1_loss(ahi_pred.view(-1) / 100.0, ahi_target.view(-1) / 100.0)
        cpc_target_log = torch.log1p(torch.clamp(cpc_target, min=0.0))
        l_cpc = F.smooth_l1_loss(cpc_pred, cpc_target_log)
        return l_apnea + self.alpha * l_ahi + self.beta * l_cpc

class PreActSEBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, reduction=16):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.fc1 = nn.Linear(out_ch, max(out_ch // reduction, 8))
        self.fc2 = nn.Linear(max(out_ch // reduction, 8), out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        shortcut_base = out if isinstance(self.shortcut, nn.Conv1d) else x
        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        w = F.adaptive_avg_pool1d(out, 1).squeeze(-1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w)).unsqueeze(-1)
        out = out * w
        return out + self.shortcut(shortcut_base)

class PASE_MST(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        def make_stream():
            return nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
                PreActSEBlock1D(32, 64, stride=2),
                PreActSEBlock1D(64, 128, stride=2),
                nn.AdaptiveAvgPool1d(1)
            )
        self.stream_ecg = make_stream()
        self.stream_ra = make_stream()
        self.stream_rri = make_stream()
        self.stream_rrid = make_stream()
        self.stream_cpc = make_stream()

        self.cross_fusion = nn.Sequential(
            nn.Linear(5 * 128, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.3)
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, ecg, ra, rri, rrid, cpc):
        f1 = self.stream_ecg(ecg).squeeze(-1)
        f2 = self.stream_ra(ra).squeeze(-1)
        f3 = self.stream_rri(rri).squeeze(-1)
        f4 = self.stream_rrid(rrid).squeeze(-1)
        f5 = self.stream_cpc(cpc).squeeze(-1)
        fused = torch.cat([f1, f2, f3, f4, f5], dim=1)
        return self.classifier(self.cross_fusion(fused))

class DREAM(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.stage1 = self._make_stage(64, 128, blocks=2, stride=1)
        self.stage2 = self._make_stage(128, 256, blocks=2, stride=2)
        self.stage3 = self._make_stage(256, 512, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_stage(self, in_ch, out_ch, blocks, stride):
        layers = [
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.fc(self.pool(x).flatten(1))

class SpatioTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, gru_hidden=64, downsample_pool=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(out_channels), nn.ReLU(inplace=True), nn.MaxPool1d(downsample_pool)
        )
        self.bigru = nn.GRU(input_size=out_channels, hidden_size=gru_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * gru_hidden, out_channels)

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.bigru(x)
        x = self.proj(x)
        return x.permute(0, 2, 1)

class CNN_BiGRU(nn.Module):
    def __init__(self, gru_hidden=64):
        super().__init__()
        self.block1 = SpatioTemporalBlock(1, 64, gru_hidden=gru_hidden, downsample_pool=4)
        self.block2 = SpatioTemporalBlock(64, 128, gru_hidden=gru_hidden, downsample_pool=4)
        self.block3 = SpatioTemporalBlock(128, 256, gru_hidden=gru_hidden, downsample_pool=2)
        self.attn = nn.Sequential(nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 1))
        self.classifier = nn.Sequential(nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x_t = x.permute(0, 2, 1)
        weights = F.softmax(self.attn(x_t), dim=1)
        context = (x_t * weights).sum(dim=1)
        return self.classifier(context)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CNN_Transformer_LSTM(nn.Module):
    def __init__(self, in_channels=2, d_model=128, nhead=4, lstm_hidden=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True), nn.MaxPool1d(2),
            nn.Conv1d(64, d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model), nn.ReLU(inplace=True)
        )
        self.pos_encoder = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, dropout=0.2, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=lstm_hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Linear(2 * lstm_hidden, 64), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(64, 2))

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x, _ = self.lstm(x)
        x = x.mean(dim=1)
        return self.classifier(x)

def build_model_and_criterion(model_type, class_weights=None, device='cuda'):
    cw = torch.tensor(class_weights, dtype=torch.float32).to(device) if class_weights is not None else None

    if model_type == 'pimtl':
        model = PIMTL_CPC(cpc_dim=3, dropout=0.3).to(device)
        criterion = PIMTL_CPC_Loss(alpha=0.5, beta=0.3, class_weights=class_weights).to(device)
    elif model_type == 'pase_mst':
        model = PASE_MST(num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)
    elif model_type == 'dream':
        model = DREAM(in_channels=1, num_classes=2).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)
    elif model_type == 'cnn_transformer_lstm':
        model = CNN_Transformer_LSTM(in_channels=2, d_model=128).to(device)
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        raise ValueError(f"Unknown architecture: {model_type}")

    return model, criterion