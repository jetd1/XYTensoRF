import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
from einops import rearrange, repeat
from tqdm import tqdm
import imageio
from kornia import create_meshgrid


SCANNET_FAR = 2.0

def read_image(img_path, img_wh, blend_a=True, unpad=0):
    img = imageio.imread(img_path).astype(np.float32)/255.0
    # img[..., :3] = srgb_to_linear(img[..., :3])
    if img.shape[2] == 4:  # blend A to RGB
        if blend_a:
            img = img[..., :3]*img[..., -1:]+(1-img[..., -1:])
        else:
            img = img[..., :3]*img[..., -1:]

    if unpad > 0:
        img = img[unpad:-unpad, unpad:-unpad]

    img = cv2.resize(img, img_wh)
    img = rearrange(img, 'h w c -> (h w) c')

    return img

def get_ray_directions_j(H, W, K, device='cpu', random=False, return_uv=False, flatten=True):
    """
    Get ray directions for all pixels in camera coordinate [right down front].
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W: image height and width
        K: (3, 3) camera intrinsics
        random: whether the ray passes randomly inside the pixel
        return_uv: whether to return uv image coordinates

    Outputs: (shape depends on @flatten)
        directions: (H, W, 3) or (H*W, 3), the direction of the rays in camera coordinate
        uv: (H, W, 2) or (H*W, 2) image coordinates
    """
    grid = create_meshgrid(H, W, False, device=device)[0] # (H, W, 2)
    u, v = grid.unbind(-1)

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    if random:
        directions = \
            torch.stack([(u-cx+torch.rand_like(u))/fx,
                         (v-cy+torch.rand_like(v))/fy,
                         torch.ones_like(u)], -1)
    else: # pass by the center
        directions = \
            torch.stack([(u-cx+0.5)/fx, (v-cy+0.5)/fy, torch.ones_like(u)], -1)
    if flatten:
        directions = directions.reshape(-1, 3)
        grid = grid.reshape(-1, 2)

    if return_uv:
        return directions, grid
    return directions

def get_rays_batch(directions, c2w):
    """
    Get ray origin and directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (HW, 3) ray directions
        c2w: (N, 3, 4) transformation matrix

    Outputs:
        rays_o: (N, HW, 3), the origin of the rays in world coordinate
        rays_d: (N, HW, 3), the direction of the rays in world coordinate
    """
    n = c2w.shape[0]
    hw = directions.shape[0]
    rays_d = repeat(directions, 'hw c -> n hw 1 c', n=n) @ \
             repeat(c2w[..., :3], 'n a b -> n hw b a', hw=hw)
    rays_d = rearrange(rays_d, 'n hw 1 c -> n hw c')
    # The origin of all rays is the camera origin in world coordinate
    rays_o = repeat(c2w[..., 3], 'n c -> n hw c', hw=hw)

    return rays_o, rays_d


class ScanNetDataset(Dataset):
    def __init__(self, root_dir, split='train', downsample=1.0, is_stack=False, sampling_strategy='all_images', n_views=100):
        super().__init__()


        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample
        self.is_stack = is_stack

        self.unpad = 24
        self.white_bg = False
        self.near_far = [0.05, 1.0]
        self.scene_bbox = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])

        self.read_intrinsics()
        self.read_meta(split)

        self.ray_sampling_strategy = sampling_strategy

    def __len__(self):
        return len(self.rgbs)

    def read_intrinsics(self):
        K = np.loadtxt(os.path.join(self.root_dir, "./intrinsic/intrinsic_color.txt"))[:3, :3]
        H, W = 968 - 2 * self.unpad, 1296 - 2 * self.unpad
        K[:2, 2] -= self.unpad
        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions_j(H, W, self.K)
        self.img_wh = (W, H)

    def read_meta(self, split):
        self.all_rgbs = []
        self.poses = []

        if split == 'train':
            with open(os.path.join(self.root_dir, "train.txt"), 'r') as f:
                frames = f.read().strip().split()
                #frames = frames
        else:
            with open(os.path.join(self.root_dir, f"{split}.txt"), 'r') as f:
                frames = f.read().strip().split()
                #frames = frames

        cam_bbox = np.loadtxt(os.path.join(self.root_dir, f"cam_bbox.txt"))
        sbbox_scale = (cam_bbox[1] - cam_bbox[0]).max() + 2 * SCANNET_FAR
        sbbox_shift = cam_bbox.mean(axis=0)

        print(f'Loading {len(frames)} {split} images ...')
        for frame in tqdm(frames):
            c2w = np.loadtxt(os.path.join(self.root_dir, f"pose/{frame}.txt"))[:3]

            # add shift
            c2w[0, 3] -= sbbox_shift[0]
            c2w[1, 3] -= sbbox_shift[1]
            c2w[2, 3] -= sbbox_shift[2]
            c2w[:, 3] /= sbbox_scale

            self.poses += [c2w]

            try:
                img_path = os.path.join(self.root_dir, f"color/{frame}.jpg")
                img = read_image(img_path, self.img_wh, unpad=self.unpad)
                self.all_rgbs += [img]
            except: pass

        self.all_rgbs = torch.FloatTensor(np.stack(self.all_rgbs))  # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses)  # (N_images, 3, 4)

        rays_o, rays_d = get_rays_batch(self.directions, self.poses)
        self.all_rays = torch.cat([rays_o, rays_d], dim=-1)

        if not self.is_stack:
            self.all_rays = self.all_rays.view(-1, 6)
            self.all_rgbs = self.all_rgbs.view(-1, 3)


    def __getitem__(self, idx):
        sample = {'rays': self.all_rays[idx],
                  'rgbs': self.all_rgbs[idx]}
        return sample
