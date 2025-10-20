import os, glob, random, cv2, numpy as np, torch
from torch.utils.data import Dataset
from src.preprocessing.pipeline import preprocess_staged_rgb_single

def list_pairs(img_dir, mask_dir, exts=("jpg","jpeg","png")):
    imgs=[]
    for e in exts: imgs += sorted(glob.glob(os.path.join(img_dir, f"*.{e}")))
    pairs=[]
    for ip in imgs:
        name = os.path.splitext(os.path.basename(ip))[0]
        mp = os.path.join(mask_dir, f"{name}.png")
        if not os.path.exists(mp):
            mp_j = os.path.join(mask_dir, f"{name}.jpg")
            if os.path.exists(mp_j): mp = mp_j
        if os.path.exists(mp): pairs.append((ip, mp))
    if not pairs: raise FileNotFoundError("No image/mask pairs found.")
    return pairs


class KvasirSegDataset(Dataset):
    def __init__(self, pairs, img_size=256, is_train=True):
        self.pairs = pairs; self.size = img_size; self.is_train = is_train
    def __len__(self): return len(self.pairs)
    def _aug(self, x, y):
        if self.is_train and random.random()<0.5:
            x = np.ascontiguousarray(np.flip(x,1)); y = np.ascontiguousarray(np.flip(y,1))
        return x,y
    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        bgr = cv2.imread(ip); m = cv2.imread(mp, 0); m = (m>=128).astype(np.uint8)
        bgr = preprocess_staged_rgb_single(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb, m = self._aug(rgb, m)
        rgb = cv2.resize(rgb,(self.size,self.size)); m = cv2.resize(m,(self.size,self.size), interpolation=cv2.INTER_NEAREST)
        x = (rgb.astype(np.float32)/255.).transpose(2,0,1)
        y = m.astype(np.float32)[None,...]
        return torch.from_numpy(x), torch.from_numpy(y)
