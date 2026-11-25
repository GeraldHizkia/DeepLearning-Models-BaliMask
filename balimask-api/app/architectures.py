"""
architectures.py - Membangun ULANG kerangka (arsitektur) tiap model.

Kenapa perlu? File .pth dari notebook hanya berisi ANGKA bobot (state_dict),
bukan kerangka modelnya. Jadi kita harus membuat kerangka yang PERSIS sama
dengan di notebook, baru bobotnya dituang ke dalamnya.

Catatan penting:
- weights=None  -> tidak perlu download bobot ImageNet, karena nanti akan
  ditimpa oleh bobot hasil training kita sendiri.
- Head v1 berbeda dengan v2/v3, jadi ada fungsi terpisah per versi.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .config import NUM_CLASSES, MOBILENET_V1_VARIANT


# ===========================================================================
# Modul atensi versi v1 (notebooks_v1_baseline/MobileNetV3_Attention.ipynb)
# ===========================================================================
class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        mid = max(c // r, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, mid, 1)
        self.fc2 = nn.Conv2d(mid, c, 1)

    def forward(self, x):
        return x * torch.sigmoid(self.fc2(F.relu(self.fc1(self.pool(x)))))


class ChannelGateV1(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        mid = max(c // r, 8)
        self.fc1 = nn.Conv2d(c, mid, 1)
        self.fc2 = nn.Conv2d(mid, c, 1)

    def _mlp(self, v):
        return self.fc2(F.relu(self.fc1(v)))

    def forward(self, x):
        a = self._mlp(F.adaptive_avg_pool2d(x, 1))
        m = self._mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(a + m)


class SpatialGateV1(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2)

    def forward(self, x):
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(s))


class CBAMV1(nn.Module):
    def __init__(self, c, r=16, k=7):
        super().__init__()
        self.ca = ChannelGateV1(c, r)
        self.sa = SpatialGateV1(k)

    def forward(self, x):
        return self.sa(self.ca(x))


# ===========================================================================
# Modul atensi versi v2/v3 (notebooks_v2_canonical/03_MobileNetV3_CBAM_*.ipynb)
# ===========================================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        return out * self.sa(out)


class MobileNetV3_CBAM(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        base = models.mobilenet_v3_large(weights=None)
        self.features = base.features
        self.cbam = CBAM(in_planes=960, ratio=16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(1280, num_classes),
        )

    def forward(self, x):
        x = self.cbam(self.features(x))
        x = torch.flatten(self.pool(x), 1)
        return self.classifier(x)


# ===========================================================================
# Builder v1 - notebooks_v1_baseline
# ===========================================================================
def _convnext_v1():
    m = models.convnext_tiny(weights=None)
    in_f = m.classifier[2].in_features
    m.classifier[2] = nn.Sequential(
        nn.Linear(in_f, 256), nn.GELU(), nn.Dropout(p=0.3), nn.Linear(256, NUM_CLASSES)
    )
    return m


def _effnet_v1():
    m = models.efficientnet_v2_s(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(p=0.3, inplace=True), nn.Linear(in_f, NUM_CLASSES))
    return m


def _vgg16_v1():
    # v1 memakai VGG16 BIASA (tanpa BatchNorm), hanya layer FC terakhir yang diganti
    m = models.vgg16(weights=None)
    m.classifier[6] = nn.Sequential(nn.Dropout(p=0.4), nn.Linear(4096, NUM_CLASSES))
    return m


def _mobilenet_v1(variant=MOBILENET_V1_VARIANT):
    m = models.mobilenet_v3_large(weights=None)
    attn = {"none": None, "se_head": SEBlock, "cbam_head": CBAMV1, "cbam_late": CBAMV1}[variant]
    if attn is not None:
        if variant.endswith("_head"):
            m.features = nn.Sequential(*list(m.features), attn(960))
        else:  # cbam_late
            for i in (13, 14, 15):
                blk = m.features[i]
                m.features[i] = nn.Sequential(blk, attn(blk.out_channels))
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
    return m


# ===========================================================================
# Builder v2 & v3 - notebooks_v2_canonical (Canonical & Ultra-Robust)
# Struktur layer-nya sama; yang beda hanya nilai dropout (tidak memengaruhi bobot).
# ===========================================================================
def _convnext_v2():
    m = models.convnext_tiny(weights=None)
    in_f = m.classifier[2].in_features
    m.classifier[2] = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_f, NUM_CLASSES))
    return m


def _effnet_v2():
    m = models.efficientnet_v2_s(weights=None)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_f, NUM_CLASSES))
    return m


def _vgg16bn_v2():
    m = models.vgg16_bn(weights=None)
    m.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 1024), nn.ReLU(True), nn.Dropout(p=0.5),
        nn.Linear(1024, 256), nn.ReLU(True), nn.Dropout(p=0.3),
        nn.Linear(256, NUM_CLASSES),
    )
    return m


BUILDERS = {
    ("convnext_tiny", "v1_awal"): _convnext_v1,
    ("efficientnetv2_s", "v1_awal"): _effnet_v1,
    ("mobilenetv3_cbam", "v1_awal"): _mobilenet_v1,
    ("vgg16", "v1_awal"): _vgg16_v1,
}
for _ver in ("v2_clean", "v3_padat"):
    BUILDERS[("convnext_tiny", _ver)] = _convnext_v2
    BUILDERS[("efficientnetv2_s", _ver)] = _effnet_v2
    BUILDERS[("mobilenetv3_cbam", _ver)] = MobileNetV3_CBAM
    BUILDERS[("vgg16", _ver)] = _vgg16bn_v2


def build_model(arch: str, version: str) -> nn.Module:
    return BUILDERS[(arch, version)]()
