import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import imageio

class Dataset:
    def __init__(self, images, poses, focal, width=None, height=None, num_samples=32, rand_sample=True, near=2.0, far=6.0, positional_encoding_dim=16):
        self.images = images
        self.poses  = poses
        self.focal  = focal
        self.rand_sample = rand_sample
        self.num_samples = num_samples
        self.num_image   = len(images)
        self.width  = self.images.shape[2] if width  is None else width
        self.height = self.images.shape[1] if height is None else height
        self.norm   = (far - near) / num_samples
        self.t_vals = np.linspace(near, far, num_samples).reshape(1, num_samples)
        self.positional_encoding_dim = positional_encoding_dim
        self.init_data()
        
    def __getitem__(self, index):
        rays, tvals = self.sample_rays(self.all_ray_dirs[index], self.all_ray_origins[index], self.rand_sample)
        return torch.FloatTensor(rays), torch.FloatTensor(tvals), self.images[index]

    def __len__(self):
        return self.num_image

    def init_data(self):

        x, y = np.meshgrid(np.arange(self.width, dtype=np.float32), np.arange(self.height, dtype=np.float32))
        self.transformed_x = (x - self.width * 0.5) / self.focal
        self.transformed_y = (y - self.height * 0.5) / self.focal

        all_ray_dirs    = []
        all_ray_origins = []
        for i in range(len(self.images)):
            ray_dirs, ray_origins = self.make_rays(self.transformed_x, self.transformed_y, self.poses[i])
            all_ray_dirs.append(ray_dirs)
            all_ray_origins.append(ray_origins)

        self.all_ray_dirs    = np.stack(all_ray_dirs, axis=0)
        self.all_ray_origins = np.stack(all_ray_origins, axis=0)

        if self.width != self.images.shape[2] or self.height != self.images.shape[1]:

            print(f"Resize images {self.images.shape[2]} x {self.images.shape[1]} to {self.width} x {self.height}")
            resized_images = []
            for i in range(len(self.images)):
                resized_images.append(cv2.resize(self.images[i], (self.width, self.height), interpolation=cv2.INTER_CUBIC))
            self.images = np.stack(resized_images, axis=0)
        self.images = torch.FloatTensor(self.images).view(self.num_image, -1, 3)


    def sample_rays(self, ray_directions, ray_origins, rand_sample):

        # ray_origins    : n x 3
        # ray_directions : n x 3
        if rand_sample:
            nrays = len(ray_origins)
            noise = np.random.uniform(size=(nrays, self.num_samples)) * self.norm
            t_vals = self.t_vals + noise
            # self.t_vals   1     x num_samples
            #      t_vals   nrays x num_samples
        else:
            #      t_vals   1     x num_samples
            t_vals = self.t_vals

        # if rand
        #    ray_origins[:, None, :]     n x 1  x 3
        #    ray_directions[:, None, :]  n x 1  x 3
        #    t_vals[..., None]           n x ns x 1
        # else
        #    ray_origins[:, None, :]     n x 1  x 3
        #    ray_directions[:, None, :]  n x 1  x 3
        #    t_vals[..., None]           1 x ns x 1
        #                     rays  ->   n x ns x 3
        rays = ray_origins[:, None, :] + ray_directions[:, None, :] * t_vals[..., None]
        rays = self.positional_encoding(rays.reshape(-1, 3), self.positional_encoding_dim)
        return rays, t_vals


    def positional_encoding(self, rays, positional_encoding_dim):
        
        # (2 * 16 + 1) * 3
        positions = [rays]
        for i in range(positional_encoding_dim):
            for fn in [np.sin, np.cos]:
                positions.append(fn(2.0 ** i * rays))
        return np.concatenate(positions, axis=-1)


    def make_rays(self, x, y, pose):

        # 100, 100, 3
        # 坐标系在-y，-z方向上
        directions    = np.stack([x, -y, -np.ones_like(x)], axis=-1)
        camera_matrix = pose[:3, :3]
        
        # 10000 x 3
        ray_dirs = directions.reshape(-1, 3) @ camera_matrix.T
        ray_origin = pose[:3, 3].reshape(1, 3).repeat(len(ray_dirs), 0)
        return ray_dirs, ray_origin


    def rotate_360_rays(self):
        def trans_t(t):
            return np.array([
                [1,0,0,0],
                [0,1,0,0],
                [0,0,1,t],
                [0,0,0,1],
            ], dtype=np.float32)

        def rot_phi(phi):
            return np.array([
                [1,0,0,0],
                [0,np.cos(phi),-np.sin(phi),0],
                [0,np.sin(phi), np.cos(phi),0],
                [0,0,0,1],
            ], dtype=np.float32)

        def rot_theta(th) : 
            return np.array([
                [np.cos(th),0,-np.sin(th),0],
                [0,1,0,0],
                [np.sin(th),0, np.cos(th),0],
                [0,0,0,1],
            ], dtype=np.float32)

        def pose_spherical(theta, phi, radius):
            c2w = trans_t(radius)
            c2w = rot_phi(phi/180.*np.pi) @ c2w
            c2w = rot_theta(theta/180.*np.pi) @ c2w
            c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
            return c2w

        all_rays  = []
        all_tvals = []
        for th in tqdm(np.linspace(0., 360., 120, endpoint=False), desc="Make rays"):
            pose = pose_spherical(th, -30., 4.)
            ray_dirs, ray_origins = self.make_rays(self.transformed_x, self.transformed_y, pose)
            rays, tvals = self.sample_rays(ray_dirs, ray_origins, rand_sample=False)
            all_rays.append(torch.FloatTensor(rays))
            all_tvals.append(torch.FloatTensor(tvals))
        return all_rays, all_tvals


def make_datasets(data_file, width, height, num_samples, positional_encoding_dim, train_ratio=0.8):
    # "data/tiny_nerf_data.npz"
    data   = np.load(data_file)
    images = data["images"]
    poses  = data["poses"]
    focal  = data["focal"]

    num_images   = len(images)
    num_train    = int(num_images * train_ratio)
    train_images = images #[:num_train]
    train_poses  = poses  #[:num_train]
    #test_images  = images[num_train:]
    #test_poses   = poses[num_train:]

    train_set = Dataset(train_images, train_poses, focal, width=width, height=height, num_samples=num_samples, rand_sample=True, positional_encoding_dim=positional_encoding_dim)
    #test_set  = Dataset(test_images, test_poses, focal, width=width, height=height, num_samples=num_samples, rand_sample=False, positional_encoding_dim=positional_encoding_dim)
    return train_set #, test_set


def render_rays(model, rays, tvals, width, height, num_sample):

    # rays -> B x (nsample * H * W) x ((2 * positional_dims + 1) * 3)
    #      -> B x 320000 x 99
    # tvals -> B x (H * W) x nsample
    #       -> B x 10000 x 32
    # predict -> B x 320000 x 4
    device      = rays.device
    BATCH       = rays.size(0)
    HW          = width * height
    predict = model(rays)
    predict = predict.view(BATCH, HW, num_sample, 4)

    # BATCH, HW, NUM_SAMPLES, 3
    rgb     = predict[..., :-1].sigmoid()

    # BATCH, HW, NUM_SAMPLES
    sigma   = predict[..., -1].relu()

    # B x (H * W) x nsample
    # delta -> B x HW x 31
    delta_prefix   = tvals[..., 1:] - tvals[..., :-1]
    delta_addition = torch.full((BATCH, tvals.size(1), 1), 1e10, device=device)
    delta = torch.cat([delta_prefix, delta_addition ], dim=-1)

    alpha    = 1.0 - torch.exp(-sigma * delta)
    exp_term = 1.0 - alpha
    epsilon  = 1e-10
    transmittance = torch.cumprod(exp_term + epsilon, axis=-1)

    # weights  ->  B x HW x 32
    weights       = alpha * transmittance
    rgb           = torch.sum(weights[..., None] * rgb, dim=-2)
    depth         = torch.sum(weights * tvals, dim=-1)
    return rgb, depth


class NeRF(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        
        w = 256
        layers = []
        ic = ninput
        for i in range(8):
            layers.append(nn.Linear(ic, w))
            ic = w
            
            if i % 4 == 0 and i > 0:
                ic = ninput + w
        
        self.head = nn.Linear(ic, noutput)
        self.b = nn.Sequential(*layers)
    
    def forward(self, x):
        
        raw_x = x
        for i in range(len(self.b)):
            x = torch.relu(self.b[i](x))
            
            if i % 4 == 0 and i > 0:
                x = torch.cat([x, raw_x], axis=-1)
        return self.head(x)


device = "cuda:0"
width  = 100
height = 100
epochs = 30
batch  = 1
num_samples = 64
positional_encoding_dim = 6
ninput_dim = (positional_encoding_dim * 2 + 1) * 3

model = NeRF(ninput_dim, 4)
model.to(device)
optimizer = optim.Adam(model.parameters(), 5e-4)

trainset = make_datasets("data/tiny_nerf_data.npz", width, height, num_samples, positional_encoding_dim)
trainloader = DataLoader(trainset, batch, shuffle=True, num_workers=8)
rot_raylist, rot_tvallist = None, None

for e in range(epochs):

    pbar = tqdm(trainloader)
    for rays, tvals, image in pbar:
        rays  = rays.to(device)
        tvals = tvals.to(device)
        image = image.to(device)

        rgb, depth = render_rays(model, rays, tvals, width, height, num_samples)
        loss = ((rgb - image)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description(f"{e} / {epochs}, Loss: {loss.item():.6f}")
    
    if (e+1) % 2 == 0:
        imgpath = f"imgs/{e:02d}.jpg"
        pthpath = f"ckpt/{e:02d}.pth"
        print(f"Save image {imgpath}")

        temp_image = (rgb[0].view(height, width, 3).detach().cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(imgpath, temp_image[..., ::-1])
        torch.save(model.state_dict(), pthpath)

    if (e+1) % 10 == 0:
        model.eval()
        imagelist = []

        if rot_raylist is None:
            rot_raylist, rot_tvallist = trainset.rotate_360_rays()

        for i, (rays, tvals) in tqdm(enumerate(zip(rot_raylist, rot_tvallist)), desc="Rendering"):
            rays  = rays.to(device).unsqueeze(dim=0)
            tvals = tvals.to(device).unsqueeze(dim=0)

            with torch.no_grad():
                rgb, depth = render_rays(model, rays, tvals, width, height, num_samples)
            
            rgb = (rgb[0].view(height, width, 3).cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(f"rotate360/{e:03d}_{i:03d}.jpg", rgb[..., ::-1])
            imagelist.append(rgb)

        video_file = f"videos/{e:03d}_rotate360.mp4"
        print(f"Write imagelist to video file {video_file}")
        imageio.mimwrite(video_file, imagelist, fps=30, quality=7)
        model.train()

print("Program done.")