import argparse
import copy
import glob
import json
import math
import os
import random
import shutil
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, resnet34, resnet50

try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights
    WEIGHT_ENUM = {"resnet18": ResNet18_Weights, "resnet34": ResNet34_Weights, "resnet50": ResNet50_Weights}
except Exception:
    WEIGHT_ENUM = {"resnet18": None, "resnet34": None, "resnet50": None}

ARCH_FN = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50}
NUM_CLASSES = 9
EPS4, EPS6, EPS8, EPS10 = 4 / 255.0, 6 / 255.0, 8 / 255.0, 10 / 255.0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def log(msg: str, out_dir: str) -> None:
    print(msg, flush=True)
    ensure_dir(out_dir)
    with open(os.path.join(out_dir, "output_training_log.txt"), "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


def cpu_state(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in sd.items()}


def find_npz(data_arg: str) -> str:
    if data_arg and os.path.isfile(data_arg):
        return data_arg
    candidates = sorted(glob.glob("*.npz"))
    for p in candidates:
        if "train" in os.path.basename(p).lower():
            return p
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No .npz dataset found. Put train.npz next to this script or pass --data.")


def cosine_lr(base: float, epoch: int, total: int, warmup: int = 0, min_ratio: float = 0.02) -> float:
    if warmup > 0 and epoch < warmup:
        return base * float(epoch + 1) / max(1, warmup)
    t = (epoch - warmup) / max(1, total - warmup)
    return base * (min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t)))


def cyclic_cosine_lr(peak: float, floor: float, step_in_cycle: int, cycle_len: int) -> float:
    t = step_in_cycle / max(1, cycle_len - 1)
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * t))


def one_hot(y: torch.Tensor, num_classes: int = NUM_CLASSES, smoothing: float = 0.0) -> torch.Tensor:
    out = torch.zeros(y.size(0), num_classes, device=y.device, dtype=torch.float32)
    out.scatter_(1, y.view(-1, 1), 1.0)
    if smoothing > 0:
        out = out * (1.0 - smoothing) + smoothing / num_classes
    return out


def soft_ce(logits: torch.Tensor, target_probs: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    loss = -(target_probs * F.log_softmax(logits, dim=1)).sum(dim=1)
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def soft_ce_per_sample(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    return -(target_probs * F.log_softmax(logits, dim=1)).sum(dim=1)


def kl_per_sample(student_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(student_logits, dim=1)
    t = teacher_probs.clamp_min(1e-8)
    return (t * (t.log() - logp)).sum(dim=1)


def build_resnet(arch: str = "resnet34", num_classes: int = NUM_CLASSES, imagenet_init: bool = False) -> nn.Module:
    weights = None
    if imagenet_init and WEIGHT_ENUM.get(arch) is not None:
        try:
            weights = WEIGHT_ENUM[arch].IMAGENET1K_V1
        except Exception:
            weights = None
    try:
        model = ARCH_FN[arch](weights=weights)
    except Exception:
        model = ARCH_FN[arch](weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def conv1_border_zero(model: nn.Module) -> None:
    with torch.no_grad():
        w = model.conv1.weight
        if w.shape[-2:] == (7, 7):
            mask = torch.zeros(7, 7, device=w.device, dtype=w.dtype)
            mask[2:5, 2:5] = 1.0
            w.mul_(mask)
            w.mul_(49.0 / 9.0)


def safe_load_resnet(path: str, arch: str, device: torch.device) -> nn.Module:
    model = build_resnet(arch=arch, imagenet_init=False).to(device)
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def save_resnet_state(out_dir: str, fname: str, arch: str, state: Dict[str, torch.Tensor]) -> str:
    ensure_dir(out_dir)
    model = build_resnet(arch, imagenet_init=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 32, 32))
    if tuple(out.shape) != (1, NUM_CLASSES):
        raise RuntimeError(f"Bad output shape {tuple(out.shape)}")
    path = os.path.join(out_dir, fname)
    torch.save(model.state_dict(), path)
    return path


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_train_npz(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    d = np.load(path)
    x = torch.from_numpy(d["images"]).float() / 255.0
    y = torch.from_numpy(d["labels"]).long()
    if x.ndim != 4 or tuple(x.shape[1:]) != (3, 32, 32):
        raise ValueError(f"Expected (N,3,32,32), got {tuple(x.shape)}")
    return x, y


def stratified_three_way(labels: torch.Tensor, val_each: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    n = len(labels)
    tr, va, vb = [], [], []
    for c in range(NUM_CLASSES):
        idx = (labels == c).nonzero(as_tuple=True)[0]
        idx = idx[torch.randperm(len(idx), generator=g)]
        ne = max(1, round(val_each * len(idx) / n))
        va.append(idx[:ne])
        vb.append(idx[ne:2 * ne])
        tr.append(idx[2 * ne:])
    def shuf(parts):
        t = torch.cat(parts)
        return t[torch.randperm(len(t), generator=g)]
    return shuf(tr), shuf(va), shuf(vb)


class RealImageDataset(Dataset):
    def __init__(self, x, y, train=True):
        self.x, self.y, self.train = x, y, train

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        if self.train:
            x = F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze(0)
            t, l = random.randint(0, 8), random.randint(0, 8)
            x = x[:, t:t + 32, l:l + 32]
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])
        return x.contiguous(), y


class WeightedSyntheticDataset(Dataset):
    def __init__(self, npz_path, train=True):
        d = np.load(npz_path)
        self.images = torch.from_numpy(d["images"])
        self.labels = torch.from_numpy(d["labels"]).long()
        self.weights = torch.from_numpy(d["weights"]).float() if "weights" in d.files else torch.ones(len(self.labels))
        self.train = train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx].float() / 255.0
        y, w = self.labels[idx], self.weights[idx]
        if self.train:
            x = F.pad(x.unsqueeze(0), (4, 4, 4, 4), mode="reflect").squeeze(0)
            t, l = random.randint(0, 8), random.randint(0, 8)
            x = x[:, t:t + 32, l:l + 32]
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])
        return x.contiguous(), y, w


def infinite_loader(loader):
    while True:
        for b in loader:
            yield b


def locate_synthetic() -> Optional[str]:
    for p in ["output_fable2/FABLE2_synth_filtered.npz",
              "output_diffusion_rst_at/DIFFRST_synthetic_filtered_hard.npz"]:
        if os.path.isfile(p):
            return p
    alt = glob.glob("output*/**/*synth*filtered*.npz", recursive=True)
    return alt[0] if alt else None


# ---------------------------------------------------------------------------
# Attacks
# ---------------------------------------------------------------------------

def _eps_t(eps, x):
    if torch.is_tensor(eps):
        return eps.view(-1, 1, 1, 1).to(x.device)
    return torch.full((x.size(0), 1, 1, 1), float(eps), device=x.device)


def pgd_ce(model, x, y, eps, steps, x_init=None):
    e = _eps_t(eps, x)
    a = (e / 4.0).clamp(1 / 255.0, 2 / 255.0)
    if x_init is None:
        x_adv = (x + torch.empty_like(x).uniform_(-1, 1) * e).clamp(0, 1).detach()
    else:
        x_adv = torch.min(torch.max(x_init, x - e), x + e).clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + a * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - e), x + e).clamp(0, 1).detach()
    return x_adv


def margin_adv(model, x, y, eps, steps):
    e = _eps_t(eps, x)
    a = (e / 4.0).clamp(1 / 255.0, 2 / 255.0)
    x_adv = (x + torch.empty_like(x).uniform_(-1, 1) * e).clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        true = logits.gather(1, y[:, None]).squeeze(1)
        masked = logits.clone()
        masked.scatter_(1, y[:, None], -1e9)
        loss = (masked.max(dim=1).values - true).mean()
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + a * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - e), x + e).clamp(0, 1).detach()
    return x_adv


def kl_adv(model, x, teacher_probs, eps, steps):
    e = _eps_t(eps, x)
    a = (e / 4.0).clamp(1 / 255.0, 2 / 255.0)
    x_adv = (x + torch.empty_like(x).uniform_(-1, 1) * e).clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = kl_per_sample(model(x_adv), teacher_probs).mean()
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + a * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - e), x + e).clamp(0, 1).detach()
    return x_adv


def trades_adv(model, x, eps, steps):
    with torch.no_grad():
        nat = F.softmax(model(x), dim=1)
    return kl_adv(model, x, nat, eps, steps)


def _dlr_loss(logits, y):
    z_sorted, _ = logits.sort(dim=1, descending=True)
    z_y = logits.gather(1, y[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, y[:, None], -1e9)
    z_other = masked.max(dim=1).values
    return -(z_y - z_other) / (z_sorted[:, 0] - z_sorted[:, 2] + 1e-12)


def apgd_attack(model, x, y, eps, n_iter=40, loss_type="ce"):
    device = x.device
    bsz = x.size(0)
    def loss_fn(logits):
        return _dlr_loss(logits, y) if loss_type == "dlr" else F.cross_entropy(logits, y, reduction="none")
    cps = [0, max(int(0.22 * n_iter), 1)]
    while cps[-1] < n_iter:
        nxt = cps[-1] + max(cps[-1] - cps[-2] - int(0.03 * n_iter), int(0.06 * n_iter))
        cps.append(min(nxt, n_iter))
    cps = sorted(set(cps))
    step = 2.0 * eps * torch.ones(bsz, 1, 1, 1, device=device)
    x_adv = (x + torch.empty_like(x).uniform_(-eps, eps)).clamp(0, 1).detach().requires_grad_(True)
    l = loss_fn(model(x_adv))
    grad = torch.autograd.grad(l.sum(), x_adv)[0]
    best_loss = l.detach().clone()
    best_adv = x_adv.detach().clone()
    x_prev = x_adv.detach().clone()
    loss_at_cp = best_loss.clone()
    step_at_cp = step.clone()
    improved = torch.zeros(bsz, device=device)
    cp_idx = 1
    for it in range(1, n_iter + 1):
        with torch.no_grad():
            z = (x_adv.detach() + step * grad.sign())
            z = torch.min(torch.max(z, x - eps), x + eps).clamp(0, 1)
            alpha = 0.75 if it > 1 else 1.0
            x_new = x_adv.detach() + alpha * (z - x_adv.detach()) + (1 - alpha) * (x_adv.detach() - x_prev)
            x_new = torch.min(torch.max(x_new, x - eps), x + eps).clamp(0, 1)
            x_prev = x_adv.detach().clone()
        x_adv = x_new.detach().requires_grad_(True)
        l = loss_fn(model(x_adv))
        grad = torch.autograd.grad(l.sum(), x_adv)[0]
        with torch.no_grad():
            imp = l.detach() > best_loss
            improved += imp.float()
            best_adv[imp] = x_adv.detach()[imp]
            best_loss[imp] = l.detach()[imp]
            need_recompute = False
            if cp_idx < len(cps) and it == cps[cp_idx]:
                interval = cps[cp_idx] - cps[cp_idx - 1]
                cond1 = improved < 0.75 * interval
                cond2 = (step_at_cp.view(-1) == step.view(-1)) & (loss_at_cp == best_loss)
                halve = cond1 | cond2
                step[halve] = step[halve] / 2.0
                x_adv = x_adv.detach()
                x_adv[halve] = best_adv[halve]
                improved = torch.zeros(bsz, device=device)
                loss_at_cp = best_loss.clone()
                step_at_cp = step.clone()
                cp_idx += 1
                need_recompute = True
        if need_recompute:
            x_adv = x_adv.detach().requires_grad_(True)
            l = loss_fn(model(x_adv))
            grad = torch.autograd.grad(l.sum(), x_adv)[0]
    return best_adv.detach()


@torch.no_grad()
def square_attack(model, x, y, eps, n_queries=600, p_init=0.8):
    device = x.device
    b, c, h, w = x.shape
    def margin(logits):
        true = logits.gather(1, y[:, None]).squeeze(1)
        masked = logits.clone()
        masked.scatter_(1, y[:, None], -1e9)
        return true - masked.max(dim=1).values
    stripes = (torch.randint(0, 2, (b, c, 1, w), device=device).float() * 2 - 1) * eps
    x_adv = (x + stripes).clamp(0, 1)
    m = margin(model(x_adv))
    for q in range(n_queries):
        frac = p_init * (0.5 ** (q / max(1, n_queries / 6)))
        s = max(1, min(int(round(math.sqrt(frac * h * w))), h))
        if not (m > 0).any():
            break
        r = random.randint(0, h - s)
        cc = random.randint(0, w - s)
        delta = (torch.randint(0, 2, (b, c, 1, 1), device=device).float() * 2 - 1) * eps
        cand = x_adv.clone()
        cand[:, :, r:r + s, cc:cc + s] = (x[:, :, r:r + s, cc:cc + s] + delta).clamp(0, 1)
        cand = torch.min(torch.max(cand, x - eps), x + eps).clamp(0, 1)
        m_new = margin(model(cand))
        better = (m_new < m) & (m > 0)
        x_adv[better] = cand[better]
        m[better] = m_new[better]
    return x_adv


def transfer_pgd(attacker, x, y, eps, steps=20):
    """Adversarial examples crafted on a FROZEN attacker model (black-box
    transfer to the candidate). The local proxy for the hidden evaluator."""
    return pgd_ce(attacker, x, y, eps, steps)


# ---------------------------------------------------------------------------
# THE JUDGE (transfer-aware, two folds, worst-case per sample)
# ---------------------------------------------------------------------------

def judge_fold(model, fx, fy, device, attackers: List[nn.Module], eps_main, eps_low,
               apgd_iters, square_queries, batch=128, full=True):
    model.eval()
    n = fx.size(0)
    correct = torch.zeros(n, dtype=torch.bool)
    surv8 = torch.ones(n, dtype=torch.bool)
    surv4 = torch.ones(n, dtype=torch.bool)
    survT = torch.ones(n, dtype=torch.bool)
    for i in range(0, n, batch):
        xb = fx[i:i + batch].to(device)
        yb = fy[i:i + batch].to(device)
        with torch.no_grad():
            correct[i:i + batch] = (model(xb).argmax(1) == yb).cpu()
        suite = [lambda: apgd_attack(model, xb, yb, eps_main, apgd_iters, "ce")]
        if full:
            suite.append(lambda: apgd_attack(model, xb, yb, eps_main, apgd_iters, "dlr"))
            if square_queries > 0:
                suite.append(lambda: square_attack(model, xb, yb, eps_main, square_queries))
        for fn in suite:
            xa = fn()
            with torch.no_grad():
                surv8[i:i + batch] &= (model(xa).argmax(1) == yb).cpu()
        for atk in (attackers if full else attackers[:1]):
            xa = transfer_pgd(atk, xb, yb, eps_main, steps=20 if full else 10)
            with torch.no_grad():
                survT[i:i + batch] &= (model(xa).argmax(1) == yb).cpu()
        xa = apgd_attack(model, xb, yb, eps_low, max(8, apgd_iters // 2), "ce")
        with torch.no_grad():
            surv4[i:i + batch] = (model(xa).argmax(1) == yb).cpu()
    clean = correct.float().mean().item()
    rob8 = (correct & surv8 & survT).float().mean().item()   # transfer counts in worst case
    rob4 = (correct & surv4).float().mean().item()
    robT = (correct & survT).float().mean().item()
    return {"clean": clean, "rob8": rob8, "rob4": rob4, "robT": robT,
            "score": 0.5 * clean + 0.5 * rob8}


def judge_state(args, arch, state, folds, attackers, device, full=True):
    model = build_resnet(arch).to(device)
    model.load_state_dict(state, strict=True)
    res = []
    for fx, fy in folds:
        res.append(judge_fold(model, fx, fy, device, attackers, EPS8, EPS4,
                              args.judge_apgd_iters if full else args.judge_lite_iters,
                              args.judge_square_queries if full else 0,
                              batch=args.eval_batch_size, full=full))
    mean = {k: float(np.mean([r[k] for r in res])) for k in res[0]}
    spread = float(abs(res[0]["score"] - res[-1]["score"])) if len(res) > 1 else 0.0
    mean["spread"] = spread
    mean["judge_score"] = mean["score"] - 0.5 * spread
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return mean


def judge_lite(args, arch, state, fold, attackers, device, subset):
    fx, fy = fold
    if subset and subset < fx.size(0):
        fx, fy = fx[:subset], fy[:subset]
    return judge_state(args, arch, state, [(fx, fy)], attackers, device, full=False)


# ---------------------------------------------------------------------------
# AWP / EMA / soups / BN recalibration
# ---------------------------------------------------------------------------

def _diff(model, proxy):
    diff = OrderedDict()
    for (k, w), (_, pw) in zip(model.state_dict().items(), proxy.state_dict().items()):
        if w.dim() <= 1 or "weight" not in k:
            continue
        dw = pw - w
        diff[k] = w.norm() / (dw.norm() + 1e-12) * dw
    return diff


def _add(model, diff, coeff):
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in diff:
                p.add_(coeff * diff[name])


class AWP:
    def __init__(self, model, arch, gamma, proxy_lr, device):
        self.model = model
        self.gamma = gamma
        self.proxy = build_resnet(arch).to(device)
        self.opt = torch.optim.SGD(self.proxy.parameters(), lr=proxy_lr)

    def calc(self, x_adv, target_probs, weights=None):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()
        loss = -soft_ce(self.proxy(x_adv), target_probs, weights)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()
        return _diff(self.model, self.proxy)

    def perturb(self, diff):
        _add(self.model, diff, self.gamma)

    def restore(self, diff):
        _add(self.model, diff, -self.gamma)


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        src = model.state_dict()
        dst = self.module.state_dict()
        for k, v in dst.items():
            sv = src[k]
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(sv.detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(sv)

    def state_dict(self):
        return cpu_state(self.module.state_dict())


def average_states(states, coeffs=None):
    if coeffs is None:
        coeffs = [1.0 / len(states)] * len(states)
    avg = {}
    for k in states[0].keys():
        v0 = states[0][k]
        if v0.dtype.is_floating_point:
            avg[k] = sum(c * s[k].float() for c, s in zip(coeffs, states)).to(v0.dtype)
        else:
            avg[k] = states[-1][k].clone()
    return avg


@torch.no_grad()
def recalibrate_bn(arch, state, loader, device, batches=200):
    model = build_resnet(arch).to(device)
    model.load_state_dict(state, strict=True)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None
    model.train()
    seen = 0
    for x, y in loader:
        model(x.to(device, non_blocking=True))
        seen += 1
        if seen >= batches:
            break
    model.eval()
    sd = cpu_state(model.state_dict())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return sd


# ---------------------------------------------------------------------------
# SAFE re-basin: per-BasicBlock internal-channel permutation alignment.
# The channels between conv1 and conv2 inside each torchvision BasicBlock are
# exact permutation symmetries decoupled from the residual stream, so this
# alignment can never break the network.
# ---------------------------------------------------------------------------

def _block_names(arch: str) -> List[str]:
    counts = {"resnet18": [2, 2, 2, 2], "resnet34": [3, 4, 6, 3]}[arch]
    names = []
    for li, c in enumerate(counts, start=1):
        for bi in range(c):
            names.append(f"layer{li}.{bi}")
    return names


@torch.no_grad()
def _collect_bn1_acts(model: nn.Module, blocks: List[str], x: torch.Tensor) -> Dict[str, torch.Tensor]:
    feats: Dict[str, torch.Tensor] = {}
    hooks = []
    modules = dict(model.named_modules())
    def make_hook(name):
        def hook(m, inp, out):
            o = out.detach()
            feats[name] = o.permute(1, 0, 2, 3).reshape(o.size(1), -1)  # C x (B*H*W)
        return hook
    for b in blocks:
        hooks.append(modules[f"{b}.bn1"].register_forward_hook(make_hook(b)))
    model.eval()
    model(x)
    for h in hooks:
        h.remove()
    return feats


def _match_channels(fa: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
    fa = fa - fa.mean(dim=1, keepdim=True)
    fb = fb - fb.mean(dim=1, keepdim=True)
    fa = fa / (fa.norm(dim=1, keepdim=True) + 1e-8)
    fb = fb / (fb.norm(dim=1, keepdim=True) + 1e-8)
    corr = fa @ fb.t()  # C x C
    if HAVE_SCIPY:
        r, c = linear_sum_assignment((-corr).cpu().numpy())
        perm = torch.as_tensor(c[np.argsort(r)], dtype=torch.long)
    else:  # greedy fallback
        C = corr.size(0)
        perm = torch.full((C,), -1, dtype=torch.long)
        used = torch.zeros(C, dtype=torch.bool)
        flat = corr.clone()
        for _ in range(C):
            idx = torch.argmax(flat)
            i, j = int(idx // C), int(idx % C)
            perm[i] = j
            used[j] = True
            flat[i, :] = -1e9
            flat[:, j] = -1e9
    return perm


def rebasin_align(args, ref_state, mem_state, x_ref: torch.Tensor, device) -> Dict[str, torch.Tensor]:
    arch = args.arch
    blocks = _block_names(arch)
    ma = build_resnet(arch).to(device)
    ma.load_state_dict(ref_state, strict=True)
    mb = build_resnet(arch).to(device)
    mb.load_state_dict(mem_state, strict=True)
    fa = _collect_bn1_acts(ma, blocks, x_ref.to(device))
    fb = _collect_bn1_acts(mb, blocks, x_ref.to(device))
    new_state = {k: v.clone() for k, v in cpu_state(mem_state).items()}
    total_gain = 0.0
    for b in blocks:
        perm = _match_channels(fa[b], fb[b])
        gain = float((torch.arange(len(perm)) != perm).float().mean())
        total_gain += gain
        new_state[f"{b}.conv1.weight"] = new_state[f"{b}.conv1.weight"][perm]
        for suf in ["weight", "bias", "running_mean", "running_var"]:
            new_state[f"{b}.bn1.{suf}"] = new_state[f"{b}.bn1.{suf}"][perm]
        new_state[f"{b}.conv2.weight"] = new_state[f"{b}.conv2.weight"][:, perm]
    del ma, mb
    if device.type == "cuda":
        torch.cuda.empty_cache()
    log(f"[rebasin] aligned {len(blocks)} blocks, mean permuted fraction={total_gain/len(blocks):.3f}", args.out)
    return new_state


# ---------------------------------------------------------------------------
# Stage 1 - checkpoint collection
# ---------------------------------------------------------------------------

CHAMPION_CANDIDATES = [
    "BEST_SERVER_0626096_resnet34.pt",
    "output_fable2/fable_submission_partams.pt",
    "fable_submission_partams.pt",
    "output_diffusion_rst_at/BEST_SERVER_0621040_resnet34.pt",
    "output_diffusion_rst_at/DIFFRST_submit_best_eps8.pt",
    "output/BEST_SERVER_0610966_resnet34.pt",
    "output/submit_best_eps8.pt",
]

POOL_CANDIDATES = [
    "output_oracle_shell/ORACLE_SHELL_RESNET34_SUBMISSION.pt",
    "output_nextgen/NEXTGEN_ONEFILE_RESNET34_SUBMISSION.pt",
    "output_diffusion_rst_at/DIFFRST_submit_best_balanced.pt",
    "output_diffusion_rst_at/DIFFRST_submit_best_eps8.pt",
    "output/submit_best_eps8.pt",
]


def collect_checkpoints(args, device):
    champ_path = args.champion if args.champion and os.path.isfile(args.champion) else None
    if champ_path is None:
        for c in CHAMPION_CANDIDATES:
            if os.path.isfile(c):
                champ_path = c
                break
    pool = []
    seen = {champ_path}
    for c in POOL_CANDIDATES:
        if c and os.path.isfile(c) and c not in seen:
            pool.append(c)
            seen.add(c)
    log(f"Champion checkpoint: {champ_path}", args.out)
    log(f"Frozen pool ({len(pool)}): {pool}", args.out)
    return champ_path, pool


# ---------------------------------------------------------------------------
# Stage 2 - zoo training
# ---------------------------------------------------------------------------

def robust_train(args, name, init_state, epochs, lr, eps_weights, transfer_frac,
                 attackers, train_x, train_y, synth_path, foldA, device, seed):
    """Shared robust trainer: TRADES + adv-CE + AWP + EMA, eps sampled from
    eps_weights over {4,6,8,10}; with prob transfer_frac the adversarial batch
    is crafted on a random FROZEN attacker (transfer-adversarial training)."""
    out_path = os.path.join(args.out, f"output_member_{name}.pt")
    if os.path.isfile(out_path) and not args.force_members:
        log(f"Member {name} exists: {out_path}", args.out)
        return out_path
    log(f"\n--- Training zoo member '{name}' (epochs={epochs}, lr={lr}, eps_w={eps_weights}, transfer={transfer_frac}) ---", args.out)
    set_seed(seed)
    arch = args.arch
    model = build_resnet(arch).to(device)
    if init_state is not None:
        model.load_state_dict(init_state, strict=True)
    else:
        m = build_resnet(arch, imagenet_init=True)
        conv1_border_zero(m)
        model.load_state_dict(m.state_dict(), strict=True)
        del m
    model.train()
    ema = EMA(model, decay=args.ema_decay)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    awp = AWP(model, arch, gamma=args.awp_gamma, proxy_lr=0.01, device=device)
    loader = DataLoader(RealImageDataset(train_x, train_y, train=True), batch_size=args.real_batch_size,
                        shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.workers > 0)
    synth_iter = None
    if synth_path:
        sd = WeightedSyntheticDataset(synth_path, train=True)
        if len(sd) > 0:
            synth_iter = infinite_loader(DataLoader(sd, batch_size=args.synth_batch_size, shuffle=True,
                                                    num_workers=args.workers, pin_memory=True, drop_last=True,
                                                    persistent_workers=args.workers > 0))
    eps_pool = [EPS4, EPS6, EPS8, EPS10]
    best = {"score": -1.0, "state": None}
    for ep in range(epochs):
        t0 = time.time()
        cur = cosine_lr(lr, ep, epochs, warmup=min(2, epochs // 4), min_ratio=0.05)
        for g in opt.param_groups:
            g["lr"] = cur
        model.train()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w = torch.ones(x.size(0), device=device)
            if synth_iter is not None:
                sx, sy, sw = next(synth_iter)
                x = torch.cat([x, sx.to(device, non_blocking=True)], dim=0)
                y = torch.cat([y, sy.to(device, non_blocking=True)], dim=0)
                w = torch.cat([w, sw.to(device, non_blocking=True).clamp(0.5, 2.0) * args.synth_loss_weight], dim=0)
            eps = random.choices(eps_pool, weights=eps_weights, k=1)[0]
            if attackers and random.random() < transfer_frac:
                atk = random.choice(attackers)
                x_adv = transfer_pgd(atk, x, y, eps, steps=args.attack_steps)
            else:
                r = random.random()
                if r < 0.55:
                    x_adv = trades_adv(model, x, eps, args.attack_steps)
                elif r < 0.85:
                    x_adv = pgd_ce(model, x, y, eps, args.attack_steps)
                else:
                    x_adv = margin_adv(model, x, y, eps, args.attack_steps)
            target = one_hot(y, smoothing=args.label_smooth)
            diff = None
            if ep >= args.awp_start:
                diff = awp.calc(x_adv, target, w)
                awp.perturb(diff)
            logits_clean = model(x)
            logits_adv = model(x_adv)
            loss = (soft_ce(logits_clean, target, w)
                    + args.beta * F.kl_div(F.log_softmax(logits_adv, dim=1),
                                           F.softmax(logits_clean.detach(), dim=1), reduction="batchmean")
                    + args.adv_ce * soft_ce(logits_adv, target, w))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if diff is not None:
                awp.restore(diff)
            ema.update(model)
        if ep % args.eval_every == 0 or ep >= epochs - 3:
            for kind, st in [("raw", cpu_state(model.state_dict())), ("ema", ema.state_dict())]:
                m = judge_lite(args, arch, st, foldA, attackers, device, args.judge_subset)
                log(f"[{name}] ep {ep:03d}/{epochs} {kind} c{m['clean']:.3f} r8{m['rob8']:.3f} rT{m['robT']:.3f} "
                    f"judge{m['judge_score']:.4f} {time.time()-t0:.0f}s", args.out)
                if m["clean"] >= args.clean_floor and m["judge_score"] > best["score"]:
                    best = {"score": m["judge_score"], "state": st}
    if best["state"] is None:
        best["state"] = ema.state_dict()
    save_resnet_state(args.out, f"output_member_{name}.pt", arch, best["state"])
    log(f"Member {name} saved (judge-lite {best['score']:.4f})", args.out)
    return out_path


# ---------------------------------------------------------------------------
# Stage 3/4A - oracle ensemble + distillation route
# ---------------------------------------------------------------------------

class OracleEnsemble:
    """Confidence-weighted soft labels: global member weights from judge-lite
    scores; per-example weighting by each member's own confidence."""

    def __init__(self, models: List[nn.Module], global_w: List[float]):
        self.models = models
        w = torch.tensor(global_w, dtype=torch.float32)
        self.gw = (w * 8.0).softmax(dim=0).tolist()

    @torch.no_grad()
    def soft(self, x):
        num = None
        den = None
        for m, g in zip(self.models, self.gw):
            p = F.softmax(m(x), dim=1)
            conf = p.max(dim=1, keepdim=True).values
            contrib = g * conf
            num = p * contrib if num is None else num + p * contrib
            den = contrib if den is None else den + contrib
        return num / den.clamp_min(1e-8)

    def adv(self, x, y, eps, steps):
        """PGD on the ENSEMBLE's averaged logits (white-box vs the oracle)."""
        e = _eps_t(eps, x)
        a = (e / 4.0).clamp(1 / 255.0, 2 / 255.0)
        x_adv = (x + torch.empty_like(x).uniform_(-1, 1) * e).clamp(0, 1).detach()
        for _ in range(steps):
            x_adv.requires_grad_(True)
            logits = None
            for m, g in zip(self.models, self.gw):
                o = m(x_adv)
                logits = g * o if logits is None else logits + g * o
            loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + a * grad.sign()
            x_adv = torch.min(torch.max(x_adv, x - e), x + e).clamp(0, 1).detach()
        return x_adv


def route_a_distill(args, oracle, init_state, train_x, train_y, synth_path, foldA, attackers, device):
    out_path = os.path.join(args.out, "output_routeA_distilled.pt")
    if os.path.isfile(out_path) and not args.force_routes:
        log(f"Route A exists: {out_path}", args.out)
        return torch.load(out_path, map_location="cpu")
    log("\n=== Route A: ensemble distillation ===", args.out)
    arch = args.arch
    model = build_resnet(arch).to(device)
    model.load_state_dict(init_state, strict=True)
    model.train()
    ema = EMA(model, decay=args.ema_decay)
    opt = torch.optim.SGD(model.parameters(), lr=args.distill_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    awp = AWP(model, arch, gamma=args.awp_gamma, proxy_lr=0.01, device=device)
    loader = DataLoader(RealImageDataset(train_x, train_y, train=True), batch_size=args.real_batch_size,
                        shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.workers > 0)
    synth_iter = None
    if synth_path:
        sd = WeightedSyntheticDataset(synth_path, train=True)
        if len(sd) > 0:
            synth_iter = infinite_loader(DataLoader(sd, batch_size=args.synth_batch_size, shuffle=True,
                                                    num_workers=args.workers, pin_memory=True, drop_last=True,
                                                    persistent_workers=args.workers > 0))
    eps_pool = [EPS4, EPS6, EPS8, EPS10]
    best = {"score": -1.0, "state": None}
    for ep in range(args.distill_epochs):
        t0 = time.time()
        cur = cosine_lr(args.distill_lr, ep, args.distill_epochs, warmup=1, min_ratio=0.05)
        for g in opt.param_groups:
            g["lr"] = cur
        model.train()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w = torch.ones(x.size(0), device=device)
            if synth_iter is not None:
                sx, sy, sw = next(synth_iter)
                x = torch.cat([x, sx.to(device, non_blocking=True)], dim=0)
                y = torch.cat([y, sy.to(device, non_blocking=True)], dim=0)
                w = torch.cat([w, sw.to(device, non_blocking=True).clamp(0.5, 2.0) * args.synth_loss_weight], dim=0)
            with torch.no_grad():
                t_probs = oracle.soft(x)
            eps = random.choices(eps_pool, weights=args.member_eps_weights_robust, k=1)[0]
            x_adv = oracle.adv(x, y, eps, args.attack_steps) if random.random() < 0.5 \
                else kl_adv(model, x, t_probs, eps, args.attack_steps)
            target = one_hot(y, smoothing=args.label_smooth)
            diff = None
            if ep >= 1:
                diff = awp.calc(x_adv, t_probs, w)
                awp.perturb(diff)
            loss = (args.w_clean_kl * (kl_per_sample(model(x), t_probs) * w).mean()
                    + args.w_adv_kl * (kl_per_sample(model(x_adv), t_probs) * w).mean()
                    + args.w_ce * soft_ce(model(x), target, w))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if diff is not None:
                awp.restore(diff)
            ema.update(model)
        if ep % args.eval_every == 0 or ep >= args.distill_epochs - 3:
            for kind, st in [("raw", cpu_state(model.state_dict())), ("ema", ema.state_dict())]:
                m = judge_lite(args, arch, st, foldA, attackers, device, args.judge_subset)
                log(f"[routeA] ep {ep:03d}/{args.distill_epochs} {kind} c{m['clean']:.3f} r8{m['rob8']:.3f} "
                    f"rT{m['robT']:.3f} judge{m['judge_score']:.4f} {time.time()-t0:.0f}s", args.out)
                if m["clean"] >= args.clean_floor and m["judge_score"] > best["score"]:
                    best = {"score": m["judge_score"], "state": st}
    if best["state"] is None:
        best["state"] = ema.state_dict()
    save_resnet_state(args.out, "output_routeA_distilled.pt", arch, best["state"])
    return best["state"]


# ---------------------------------------------------------------------------
# Stage 4B - merging route (lineage soup + re-basin-aligned fresh member)
# ---------------------------------------------------------------------------

def quick_anneal(args, state, epochs, train_x, train_y, foldA, attackers, device, tag):
    """Short robust anneal after a merge: TRADES eps {6,8} + AWP + EMA."""
    a = argparse.Namespace(**vars(args))
    out = robust_train(a, f"anneal_{tag}", state, epochs, args.anneal_lr,
                       eps_weights=[0.0, 0.35, 0.55, 0.10], transfer_frac=0.25,
                       attackers=attackers, train_x=train_x, train_y=train_y,
                       synth_path=None, foldA=foldA, device=device, seed=args.seed + 777)
    return torch.load(out, map_location="cpu")


def route_b_merge(args, champ_state, lineage_states, fresh_state, train_x, train_y,
                  bn_loader, foldA, attackers, device):
    log("\n=== Route B: soups + safe re-basin merge ===", args.out)
    arch = args.arch
    out: List[Tuple[str, Dict[str, torch.Tensor]]] = []
    # B1: same-lineage soup (same basin, no alignment needed)
    pool = [champ_state] + lineage_states
    if len(pool) >= 2:
        soup = recalibrate_bn(arch, average_states(pool), bn_loader, device, args.bn_batches)
        soup = quick_anneal(args, soup, args.anneal_epochs, train_x, train_y, foldA, attackers, device, "lineage")
        out.append(("routeB_lineage_soup", soup))
    # B2: + re-basin-aligned fresh member (cross-basin diversity)
    if fresh_state is not None:
        x_ref = torch.stack([RealImageDataset(train_x, train_y, train=False)[i][0]
                             for i in range(min(args.rebasin_images, len(train_x)))])
        aligned = rebasin_align(args, champ_state, fresh_state, x_ref, device)
        merged = recalibrate_bn(arch, average_states(pool + [aligned],
                                                     coeffs=None), bn_loader, device, args.bn_batches)
        merged = quick_anneal(args, merged, args.anneal_epochs, train_x, train_y, foldA, attackers, device, "rebasin")
        out.append(("routeB_rebasin_merge", merged))
    return out


# ---------------------------------------------------------------------------
# Stage 5 - CVaR attack-distribution annealing (+ SWA tail)
# ---------------------------------------------------------------------------

def cvar_anneal(args, start_state, attackers, train_x, train_y, synth_path, foldA, bn_loader, device):
    log("\n=== Stage 5: CVaR attack-distribution annealing + SWA tail ===", args.out)
    arch = args.arch
    model = build_resnet(arch).to(device)
    model.load_state_dict(start_state, strict=True)
    model.train()
    ema = EMA(model, decay=args.ema_decay)
    opt = torch.optim.SGD(model.parameters(), lr=args.cvar_lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    awp = AWP(model, arch, gamma=args.awp_gamma, proxy_lr=0.01, device=device)
    loader = DataLoader(RealImageDataset(train_x, train_y, train=True), batch_size=args.cvar_batch_size,
                        shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.workers > 0)
    synth_iter = None
    if synth_path:
        sd = WeightedSyntheticDataset(synth_path, train=True)
        if len(sd) > 0:
            synth_iter = infinite_loader(DataLoader(sd, batch_size=max(16, args.cvar_batch_size // 2),
                                                    shuffle=True, num_workers=args.workers, pin_memory=True,
                                                    drop_last=True, persistent_workers=args.workers > 0))
    eps_pool = [EPS4, EPS6, EPS8, EPS10]
    eps_w = args.cvar_eps_weights

    def sample_attack(x, y, target):
        eps = random.choices(eps_pool, weights=eps_w, k=1)[0]
        kinds = ["self_pgd", "self_margin", "self_kl"] + (["transfer"] if attackers else [])
        kind = random.choice(kinds)
        if kind == "self_pgd":
            return pgd_ce(model, x, y, eps, args.attack_steps)
        if kind == "self_margin":
            return margin_adv(model, x, y, eps, args.attack_steps)
        if kind == "self_kl":
            with torch.no_grad():
                nat = F.softmax(model(x), dim=1)
            return kl_adv(model, x, nat, eps, args.attack_steps)
        return transfer_pgd(random.choice(attackers), x, y, eps, steps=args.attack_steps)

    best = {"score": -1.0, "state": None}
    total_epochs = args.cvar_epochs
    for ep in range(total_epochs):
        t0 = time.time()
        cur = cosine_lr(args.cvar_lr, ep, total_epochs, warmup=1, min_ratio=0.08)
        for g in opt.param_groups:
            g["lr"] = cur
        model.train()
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w = torch.ones(x.size(0), device=device)
            if synth_iter is not None:
                sx, sy, sw = next(synth_iter)
                x = torch.cat([x, sx.to(device, non_blocking=True)], dim=0)
                y = torch.cat([y, sy.to(device, non_blocking=True)], dim=0)
                w = torch.cat([w, sw.to(device, non_blocking=True).clamp(0.5, 2.0) * args.synth_loss_weight], dim=0)
            target = one_hot(y, smoothing=args.label_smooth)
            # sample K attacks (detached inputs)
            adv_sets = [sample_attack(x, y, target) for _ in range(args.cvar_k)]
            # pick the strongest sampled attack under no_grad (for AWP direction)
            with torch.no_grad():
                means = [float(soft_ce_per_sample(model(xa), target).mean()) for xa in adv_sets]
            worst_idx = int(np.argmax(means))
            # AWP perturb FIRST, then build every gradient-carrying forward
            diff = awp.calc(adv_sets[worst_idx], target, w)
            awp.perturb(diff)
            per_losses = [soft_ce_per_sample(model(xa), target) * w for xa in adv_sets]
            L = torch.cat(per_losses, dim=0)                     # (K*B,)
            k_keep = max(1, int(math.ceil(args.cvar_alpha * L.numel())))
            cvar_loss = torch.topk(L, k_keep).values.mean()
            clean_loss = soft_ce(model(x), target, w)
            loss = args.cvar_w_clean * clean_loss + args.cvar_w_adv * cvar_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            awp.restore(diff)
            ema.update(model)
        for kind, st in [("raw", cpu_state(model.state_dict())), ("ema", ema.state_dict())]:
            m = judge_lite(args, arch, st, foldA, attackers, device, args.judge_subset)
            log(f"[cvar] ep {ep:03d}/{total_epochs} {kind} c{m['clean']:.3f} r8{m['rob8']:.3f} rT{m['robT']:.3f} "
                f"judge{m['judge_score']:.4f} {time.time()-t0:.0f}s", args.out)
            if m["clean"] >= args.clean_floor and m["judge_score"] > best["score"]:
                best = {"score": m["judge_score"], "state": st}
                save_resnet_state(args.out, "output_cvar_best.pt", arch, st)

    # short SWA tail from the best CVaR state (flatness is what the server rewards)
    snapshots = []
    model.load_state_dict({k: v.to(device) for k, v in (best["state"] or ema.state_dict()).items()}, strict=True)
    for cyc in range(args.swa_cycles):
        for e_in in range(args.swa_cycle_len):
            cur = cyclic_cosine_lr(args.swa_peak_lr, args.swa_peak_lr * 0.1, e_in, args.swa_cycle_len)
            for g in opt.param_groups:
                g["lr"] = cur
            model.train()
            for x, y in loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                target = one_hot(y, smoothing=args.label_smooth)
                eps = random.choices([EPS6, EPS8], weights=[0.4, 0.6], k=1)[0]
                x_adv = trades_adv(model, x, eps, args.attack_steps)
                diff = awp.calc(x_adv, target)
                awp.perturb(diff)
                loss = (soft_ce(model(x), target)
                        + args.beta * F.kl_div(F.log_softmax(model(x_adv), dim=1),
                                               F.softmax(model(x).detach(), dim=1), reduction="batchmean"))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                awp.restore(diff)
        snapshots.append(cpu_state(model.state_dict()))
        log(f"[swa] snapshot {len(snapshots)}", args.out)
    cands = {"cvar_best": best["state"] or ema.state_dict()}
    if len(snapshots) >= 2:
        cands["cvar_swa"] = recalibrate_bn(arch, average_states(snapshots), bn_loader, device, args.bn_batches)
    return cands


# ---------------------------------------------------------------------------
# Stage 6 - final selection
# ---------------------------------------------------------------------------

def final_select(args, candidates, folds, attackers, bn_loader, device):
    log("\n=== Stage 6: FINAL selection (full transfer-aware judge, two folds) ===", args.out)
    arch = args.arch
    pool = [s for _, s in candidates]
    if len(pool) >= 2:
        soup = recalibrate_bn(arch, average_states(pool), bn_loader, device, args.bn_batches)
        candidates = candidates + [("final_soup_all", soup)]
    best = None
    report = []
    for name, st in candidates:
        m = judge_state(args, arch, st, folds, attackers, device, full=True)
        log(f"[FINAL] {name}: c{m['clean']:.4f} r8{m['rob8']:.4f} rT{m['robT']:.4f} r4{m['rob4']:.4f} "
            f"score{m['score']:.4f} spread{m['spread']:.4f} judge{m['judge_score']:.4f}", args.out)
        report.append({"name": name, "metrics": m})
        if m["clean"] >= args.clean_floor and (best is None or m["judge_score"] > best[2]["judge_score"]):
            best = (name, st, m)
    if best is None:
        report.sort(key=lambda r: r["metrics"]["score"], reverse=True)
        nm = report[0]["name"]
        st = dict(candidates)[nm]
        best = (nm, st, report[0]["metrics"])
    name, st, m = best
    sub_path = save_resnet_state(args.out, args.submission_name, arch, st)
    unique = f"output_FINAL_{name}_{arch}_clean{m['clean']:.4f}_rob8{m['rob8']:.4f}_judge{m['judge_score']:.4f}.pt"
    unique_path = save_resnet_state(args.out, unique, arch, st)
    try:
        shutil.copyfile(sub_path, args.submission_name)
    except Exception:
        pass
    with open(os.path.join(args.out, "output_SUBMISSION_INFO.json"), "w", encoding="utf-8") as f:
        json.dump({"winner": name, "metrics": m, "submission": sub_path,
                   "unique": unique_path, "all_candidates": report}, f, indent=2)
    log("\noutput RUN DONE", args.out)
    log(f"WINNER: {name}", args.out)
    log(f"SUBMISSION FILE: {sub_path}  (also copied to ./{args.submission_name})", args.out)
    log(f"FULL-JUDGE METRICS: {m}", args.out)
    log(f"Use MODEL_NAME = {arch} and MODEL_PATH = {args.submission_name}", args.out)
    return sub_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="output / CHIMERA: zoo + oracle ensemble + dual compression + CVaR annealing")
    p.add_argument("--data", default="train.npz")
    p.add_argument("--out", default="output")
    p.add_argument("--arch", default="resnet34")
    p.add_argument("--submission-name", default="output_submission.pt")
    p.add_argument("--champion", default="")
    p.add_argument("--seed", type=int, default=77)
    p.add_argument("--val-size-each", type=int, default=1500)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--clean-floor", type=float, default=0.55)
    p.add_argument("--label-smooth", type=float, default=0.05)

    # judge
    p.add_argument("--judge-apgd-iters", type=int, default=40)
    p.add_argument("--judge-lite-iters", type=int, default=15)
    p.add_argument("--judge-square-queries", type=int, default=500)
    p.add_argument("--judge-subset", type=int, default=640)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--max-attackers", type=int, default=4)

    # shared robust knobs
    p.add_argument("--real-batch-size", type=int, default=96)
    p.add_argument("--synth-batch-size", type=int, default=64)
    p.add_argument("--synth-loss-weight", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=5.0)
    p.add_argument("--adv-ce", type=float, default=0.30)
    p.add_argument("--attack-steps", type=int, default=10)
    p.add_argument("--ema-decay", type=float, default=0.999)
    p.add_argument("--awp-gamma", type=float, default=0.002)
    p.add_argument("--awp-start", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=3)
    p.add_argument("--bn-batches", type=int, default=200)
    p.add_argument("--force-members", action="store_true")
    p.add_argument("--force-routes", action="store_true")

    # zoo
    p.add_argument("--member-epochs", type=int, default=18)
    p.add_argument("--member-lr", type=float, default=0.006)
    p.add_argument("--fresh-epochs", type=int, default=55)
    p.add_argument("--fresh-lr", type=float, default=0.05)
    p.add_argument("--member-eps-weights-clean", nargs=4, type=float, default=[0.35, 0.45, 0.20, 0.0])
    p.add_argument("--member-eps-weights-robust", nargs=4, type=float, default=[0.05, 0.25, 0.50, 0.20])
    p.add_argument("--member-eps-weights-mid", nargs=4, type=float, default=[0.15, 0.35, 0.40, 0.10])
    p.add_argument("--transfer-frac-heavy", type=float, default=0.6)
    p.add_argument("--transfer-frac-light", type=float, default=0.15)

    # distillation route
    p.add_argument("--distill-epochs", type=int, default=16)
    p.add_argument("--distill-lr", type=float, default=0.004)
    p.add_argument("--w-clean-kl", type=float, default=1.0)
    p.add_argument("--w-adv-kl", type=float, default=1.5)
    p.add_argument("--w-ce", type=float, default=0.30)

    # merging route
    p.add_argument("--anneal-epochs", type=int, default=3)
    p.add_argument("--anneal-lr", type=float, default=0.002)
    p.add_argument("--rebasin-images", type=int, default=1024)

    # CVaR stage
    p.add_argument("--cvar-epochs", type=int, default=12)
    p.add_argument("--cvar-lr", type=float, default=0.003)
    p.add_argument("--cvar-batch-size", type=int, default=64)
    p.add_argument("--cvar-k", type=int, default=3)
    p.add_argument("--cvar-alpha", type=float, default=0.30)
    p.add_argument("--cvar-w-clean", type=float, default=1.0)
    p.add_argument("--cvar-w-adv", type=float, default=1.2)
    p.add_argument("--cvar-eps-weights", nargs=4, type=float, default=[0.20, 0.30, 0.40, 0.10],
                   help="attack-eps distribution over 4/6/8/10; re-weight after fingerprinting")

    # SWA tail
    p.add_argument("--swa-cycles", type=int, default=2)
    p.add_argument("--swa-cycle-len", type=int, default=3)
    p.add_argument("--swa-peak-lr", type=float, default=0.0015)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.out)
    open(os.path.join(args.out, "output_training_log.txt"), "w", encoding="utf-8").close()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"{now()} output/CHIMERA starting on device={device}", args.out)
    if device.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}", args.out)
    log(f"Args: {json.dumps(vars(args), indent=2)}", args.out)
    if not HAVE_SCIPY:
        log("scipy not found - re-basin uses greedy matching fallback (still safe).", args.out)

    data_path = find_npz(args.data)
    x_all, y_all = load_train_npz(data_path)
    tr_idx, va_idx, vb_idx = stratified_three_way(y_all, args.val_size_each, args.seed)
    train_x, train_y = x_all[tr_idx], y_all[tr_idx]
    foldA = (x_all[va_idx], y_all[va_idx])
    foldB = (x_all[vb_idx], y_all[vb_idx])
    log(f"Loaded {data_path}: train={len(train_x)} foldA={len(foldA[0])} foldB={len(foldB[0])}", args.out)
    bn_loader = DataLoader(RealImageDataset(train_x, train_y, train=True), batch_size=256, shuffle=True,
                           num_workers=args.workers, pin_memory=True, drop_last=True,
                           persistent_workers=args.workers > 0)
    synth_path = locate_synthetic()
    log(f"Synthetic data: {synth_path}", args.out)

    # Stage 1 - collect checkpoints, build frozen attacker set
    champ_path, pool_paths = collect_checkpoints(args, device)
    attackers: List[nn.Module] = []
    for p in ([champ_path] + pool_paths)[:args.max_attackers]:
        if p:
            attackers.append(safe_load_resnet(p, args.arch, device))
    champ_state = cpu_state(safe_load_resnet(champ_path, args.arch, device).state_dict()) if champ_path else None

    # Stage 2 - zoo
    if champ_state is None:
        log("No champion found - training fresh baseline as champion first.", args.out)
        path = robust_train(args, "fresh_champion", None, args.fresh_epochs, args.fresh_lr,
                            args.member_eps_weights_mid, 0.0, attackers, train_x, train_y,
                            synth_path, foldA, device, args.seed + 11)
        champ_state = torch.load(path, map_location="cpu")
    m_clean = torch.load(robust_train(args, "knee_clean", champ_state, args.member_epochs, args.member_lr,
                                      args.member_eps_weights_clean, args.transfer_frac_light,
                                      attackers, train_x, train_y, synth_path, foldA, device, args.seed + 21),
                         map_location="cpu")
    m_robust = torch.load(robust_train(args, "robust", champ_state, args.member_epochs, args.member_lr,
                                       args.member_eps_weights_robust, args.transfer_frac_light,
                                       attackers, train_x, train_y, synth_path, foldA, device, args.seed + 31),
                          map_location="cpu")
    m_transfer = torch.load(robust_train(args, "transfer_hard", champ_state, args.member_epochs, args.member_lr,
                                         args.member_eps_weights_mid, args.transfer_frac_heavy,
                                         attackers, train_x, train_y, synth_path, foldA, device, args.seed + 41),
                            map_location="cpu")
    m_fresh = torch.load(robust_train(args, "fresh", None, args.fresh_epochs, args.fresh_lr,
                                      args.member_eps_weights_mid, args.transfer_frac_light,
                                      attackers, train_x, train_y, synth_path, foldA, device, args.seed + 51),
                         map_location="cpu")

    # Stage 3 - oracle ensemble (judge-lite weights; sanity check vs members)
    member_states = [("champion", champ_state), ("knee_clean", m_clean), ("robust", m_robust),
                     ("transfer_hard", m_transfer), ("fresh", m_fresh)]
    models, weights, member_scores = [], [], {}
    for nm, st in member_states:
        mdl = build_resnet(args.arch).to(device)
        mdl.load_state_dict(st, strict=True)
        mdl.eval()
        for q in mdl.parameters():
            q.requires_grad_(False)
        sc = judge_lite(args, args.arch, st, foldA, attackers, device, args.judge_subset)
        member_scores[nm] = sc["judge_score"]
        log(f"[ensemble] member {nm}: judge-lite {sc['judge_score']:.4f} (c{sc['clean']:.3f} r8{sc['rob8']:.3f})", args.out)
        models.append(mdl)
        weights.append(sc["judge_score"])
    oracle = OracleEnsemble(models, weights)
    log(f"[ensemble] global weights: { {nm: round(wt,3) for nm, wt in zip([n for n,_ in member_states], oracle.gw)} }", args.out)

    # Stage 4 - dual compression
    route_a = route_a_distill(args, oracle, champ_state, train_x, train_y, synth_path, foldA, attackers, device)
    route_b = route_b_merge(args, champ_state, [m_clean, m_robust, m_transfer], m_fresh,
                            train_x, train_y, bn_loader, foldA, attackers, device)
    comp = [("routeA_distilled", route_a)] + route_b
    comp_best = None
    for nm, st in comp:
        sc = judge_lite(args, args.arch, st, foldA, attackers, device, args.judge_subset)
        log(f"[compress] {nm}: judge-lite {sc['judge_score']:.4f} (c{sc['clean']:.3f} r8{sc['rob8']:.3f} rT{sc['robT']:.3f})", args.out)
        if sc["clean"] >= args.clean_floor and (comp_best is None or sc["judge_score"] > comp_best[1]):
            comp_best = (nm, sc["judge_score"], st)
    if comp_best is None:
        comp_best = ("champion_fallback", member_scores["champion"], champ_state)
    log(f"[compress] winner: {comp_best[0]} (judge-lite {comp_best[1]:.4f})", args.out)

    # Stage 5 - CVaR annealing + SWA tail
    cvar_cands = cvar_anneal(args, comp_best[2], attackers, train_x, train_y, synth_path, foldA, bn_loader, device)

    # Stage 6 - final selection
    finalists = [("champion_anchor", champ_state), (comp_best[0], comp_best[2])]
    finalists += [(k, v) for k, v in cvar_cands.items()]
    finalists.append(("routeA_distilled", route_a))
    seen = set()
    uniq = []
    for nm, st in finalists:
        if nm in seen:
            continue
        seen.add(nm)
        uniq.append((nm, st))
    final_select(args, uniq, [foldA, foldB], attackers, bn_loader, device)


if __name__ == "__main__":
    main()
