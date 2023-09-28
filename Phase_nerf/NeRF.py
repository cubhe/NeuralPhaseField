import torch
import torch.nn as nn
from run_nerf_helpers import *
import os
import imageio
import time
import math


class CameraNeRF(nn.Module):
    def __init__(self, num_img, kernel_size=30, num_hidden=3,
                 num_wide=256, short_cut=(True, True),
                 img_embed=0, depth_embed=0, fd_embed=0, pos_embed=0):
        """
        num_img: number of image, used for deciding the focus distance
        kernel_size: the size of physically equivalent blur kernel. Can be a very big number
        gaussian_function:use Gaussian function to calculate initial kernel

        img_embed: the length of the view embedding
        depth_embed: (deprecated) the embedding for the depth of current rays
        fd_embed:
        pos_embed:

        num_hidden, num_wide, short_cut: control the structure of the MLP
        """
        super().__init__()
        self.num_img = num_img
        self.short_cut1 = short_cut[0]
        self.short_cut2 = short_cut[1]
        self.kernel_size = kernel_size
        self.num_wide = num_wide

        X = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, 1)
        Y = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, 1)
        self.X_grid, self.Y_grid = torch.meshgrid(X, Y)

        # input embedding
        if img_embed > 0:
            self.img_embed_fn, self.img_embed_cnl = get_embedder(img_embed, input_dim=3)
        else:
            self.img_embed_fn, self.img_embed_cnl = None, 3

        if depth_embed > 0:
            self.require_depth = True
            self.depth_embed_fn, self.depth_embed_cnl = get_embedder(depth_embed, input_dim=1)
        else:
            self.require_depth = False
            self.depth_embed_fn, self.depth_embed_cnl = None, 1

        if fd_embed > 0:
            self.require_fd = True
            self.fd_embed_fn, self.fd_embed_cnl = get_embedder(fd_embed, input_dim=1)
        else:
            self.require_fd = False
            self.fd_embed_fn, self.fd_embed_cnl = None, 1

        if pos_embed > 0:
            self.require_pos = True
            self.pos_embed_fn, self.pos_embed_cnl = get_embedder(pos_embed, input_dim=2)
        else:
            self.require_pos = False
            self.pos_embed_fn, self.pos_embed_cnl = None, 2

        # MLP
        # out_cnl = self.kernel_size * self.kernel_size
        hiddens = [nn.Linear(num_wide, num_wide) if i % 2 == 0 else nn.ReLU()
                   for i in range((num_hidden - 1) * 2)]
        ### 参数化
        # first MLP
        self.linears1_1 = nn.Sequential(
            nn.Linear(self.depth_embed_cnl, num_wide), nn.ReLU(),
            *hiddens,
        )
        self.linears1_2 = nn.Sequential(
            nn.Linear((num_wide + self.depth_embed_cnl) if self.short_cut1 else num_wide, num_wide), nn.ReLU(),
            *hiddens,
        )
        self.linears1_3 = nn.Sequential(
            nn.Linear(self.num_wide + self.fd_embed_cnl, self.num_wide), nn.ReLU(),
            nn.Linear(self.num_wide, self.num_wide), nn.ReLU(),
        )
        self.sigma_activate = nn.Sequential(
            nn.Linear(self.num_wide, self.num_wide // 2), nn.ReLU(),
            nn.Linear(self.num_wide // 2, 2)
        )
        # second MLP
        self.linears2_1 = nn.Sequential(
            nn.Linear(self.img_embed_cnl + num_wide, num_wide), nn.ReLU(),
            *hiddens
        )
        self.linears2_2 = nn.Sequential(
            nn.Linear((num_wide + self.img_embed_cnl) if self.short_cut2 else num_wide, num_wide), nn.ReLU(),
            *hiddens
        )
        self.linears2_3 = nn.Sequential(
            nn.Linear(self.num_wide + self.pos_embed_cnl, self.num_wide), nn.ReLU(),
            nn.Linear(self.num_wide, self.num_wide), nn.ReLU(),
        )
        self.murho_activate = nn.Sequential(
            nn.Linear(self.num_wide, self.num_wide // 2), nn.ReLU(),
            nn.Linear(self.num_wide // 2, 5)
        )
        # third MLP
        self.linears3_1 = nn.Sequential(
            nn.Linear(num_wide, num_wide), nn.ReLU(),
            *hiddens
        )
        self.linears3_2 = nn.Sequential(
            nn.Linear((num_wide + self.img_embed_cnl), num_wide), nn.ReLU(),
            *hiddens
        )
        self.w_activate = nn.Sequential(
            nn.Linear(self.num_wide, self.num_wide*2), nn.ReLU(),
            nn.Linear(self.num_wide*2, self.kernel_size*self.kernel_size)
        )

        self.linears1_1.apply(init_linear_weights)
        self.linears1_2.apply(init_linear_weights)
        self.linears1_3.apply(init_linear_weights)
        self.sigma_activate.apply(init_linear_weights)
        self.linears2_1.apply(init_linear_weights)
        self.linears2_2.apply(init_linear_weights)
        self.linears2_3.apply(init_linear_weights)
        self.murho_activate.apply(init_linear_weights)
        self.linears3_1.apply(init_linear_weights)
        self.linears3_2.apply(init_linear_weights)
        self.w_activate.apply(init_linear_weights)


    def forward(self, camera_rays_info, AIF_padding_ROIs):
        """
        inputs: image shape, rays_info{ image(shape:ray_num,3), depth(shape:ray_num,1), fd(shape:ray_num,1), pos(shape:ray_num,2) }
        outputs: blur image(shape:ray_num,3)
        """
        ray_batch_num = camera_rays_info['camera_rays'].shape[0]
        img_idx = camera_rays_info['camera_focus_images_idx'].squeeze(-1)
        pix_pos = camera_rays_info['camera_pix_pos']
        self.X_grids = torch.tile(self.X_grid[None, ...], [ray_batch_num, 1, 1])
        self.Y_grids = torch.tile(self.Y_grid[None, ...], [ray_batch_num, 1, 1])

        pxs, pys = pix_pos[:, 0], pix_pos[:, 1]
        AIF_ROI = []
        for px, py,idx in zip(pxs, pys,img_idx):
            id=idx.cpu().detach().numpy()
            AIF_padding_ROI = AIF_padding_ROIs[str(id)][py, px, ...]
            AIF_ROI.append(AIF_padding_ROI)
        AIF_ROI = torch.stack(AIF_ROI, dim=0)

        depth = camera_rays_info['camera_depthsf']
        if self.depth_embed_fn is not None:
            depth = depth * np.pi  # TODO: please always check that the depth is measured in meters.
            depth = self.depth_embed_fn(depth)

        fd = camera_rays_info['camera_fd']
        if self.fd_embed_fn is not None:
            fd = fd * np.pi  # TODO: please always check that the fd is measured in meters.
            fd = self.fd_embed_fn(fd)

        imgC = camera_rays_info['camera_AIFsf']
        if self.img_embed_fn is not None:
            imgC = imgC * np.pi
            imgC = self.img_embed_fn(imgC)

        rays_pos = camera_rays_info['camera_rays_pos']
        if self.pos_embed_fn is not None:
            rays_pos = rays_pos * np.pi
            rays_pos = self.pos_embed_fn(rays_pos)

        ####参数化
        # forward
        time1 = time.time()
        x = self.linears1_1(depth)  # 4
        x = torch.cat([x, depth], dim=-1) if self.short_cut1 else x
        x = self.linears1_2(x)  # 4
        x = torch.cat([x, fd], dim=-1)
        x = self.linears1_3(x)
        sigma = self.sigma_activate(x)
        # x=torch.sigmoid(x)
        # sigma1, sigma2 = torch.sigmoid(x[:, 0])*39+1, torch.sigmoid(x[:, 1])*39+1
        sigma1, sigma2 = 5*torch.sigmoid(sigma[:, 0]) + 0.0001, 5*torch.sigmoid(sigma[:, 1]) + 0.0001
        sigma = torch.stack([sigma1, sigma2], dim=1)
        time2 = time.time()
        # assert not torch.any(torch.isnan(sigma))
        imgB1 = self.gaussian_layer(AIF_ROI, sigma)
        # assert not torch.any(torch.isnan(imgB0))
        time3 = time.time()

        x1 = torch.cat([x, imgC], dim=-1)
        x1 = self.linears2_1(x1)
        x1 = torch.cat([x1, imgC], dim=-1) if self.short_cut2 else x1
        x1 = self.linears2_2(x1)  # 4
        x1 = torch.cat([x1, rays_pos], dim=-1)
        x1 = self.linears2_3(x1)
        murho = self.murho_activate(x1)

        sigma1, sigma2 = 5*torch.sigmoid(murho[:, 0])+ 0.0001, 5*torch.sigmoid(murho[:, 1])+ 0.0001
        mu1, mu2 = murho[:, 2] + 0.0001, murho[:, 3] + 0.0001
        rho = torch.sigmoid(murho[:, 4]) * 0.1


        # mu, rho = murho[:, 0:2], murho[:, 2]
        # mu1, mu2 = mu[:, 0] + 0.001, mu[:, 1] + 0.001

        # rho = torch.tanh(rho) * 0.99
        time4 = time.time()
        # assert not torch.any(torch.isnan(mu))
        # assert not torch.any(torch.isnan(rho))
        # print(sigma[0], mu[0], rho[0])
        # print(sigma[1023], mu[1023], rho[1023])
        sigma = torch.stack([sigma1, sigma2], dim=1)
        mu = torch.stack([mu1, mu2], dim=1)
        imgB2 = self.gaussian_layer(AIF_ROI, sigma, mu, rho)

        # assert not torch.any(torch.isnan(imgB))
        # print(imgC[0],imgB[0])
        # time5 = time.time()
        x2=self.linears3_1(x1)
        x2 = torch.cat([x2, imgC], dim=-1)
        x2 = self.linears3_2(x2)
        w=self.w_activate(x2)
        w_reshape=w.reshape(w.shape[0],self.kernel_size,self.kernel_size)
        value = AIF_ROI * w_reshape[..., None, :, :]
        value = value.reshape(value.shape[0], 3, -1)
        imgB3 = (torch.sum(value, dim=2)[..., :]) / ((torch.sum(w, dim=1)[..., None]) + 0.00001)

        # print("mlp1:", time2 - time1)
        # print("g1:", time3 - time2)
        # print("mlp2:", time4 - time3)
        # print("g2:", time5 - time4)
        gauss_paras=[sigma1,sigma2,mu1,mu2,rho]
        gauss_paras=torch.stack(gauss_paras,1)

        return {'blur_img1': imgB1,'blur_img2': imgB2,  'blur_img3': imgB3, 'paras':gauss_paras}

        #return {'blur_img1': imgB1, 'blur_img2': imgB2, 'blur_img3': imgB3}

    def gaussian_layer(self, AIF_ROI, sigma, mu=torch.Size([]), rho=torch.Size([])):
        sigma1, sigma2 = sigma[:, 0], sigma[:, 1]
        # imgB = torch.zeros((rays_batch_num, 3))

        time1 = time.time()
        if mu != torch.Size([]) and rho != torch.Size([]):
            mu1, mu2 = mu[:, 0], mu[:, 1]
            a = 1. / (2. * torch.pi * sigma1[..., None, None] * sigma2[..., None, None] * (
                torch.sqrt(1 - rho[..., None, None] * rho[..., None, None])))
            b = -1. / (2. * (1 - rho[..., None, None] * rho[..., None, None])) * (
                    (self.X_grids[..., :, :] - mu1[..., None, None]) * (
                    self.X_grids[..., :, :] - mu1[..., None, None])
                    / (sigma1[..., None, None] * sigma1[..., None, None])
                    - 2 * rho[..., None, None] * (self.X_grids - mu1[..., None, None]) * (
                            self.Y_grids[..., :, :] - mu2[..., None, None])
                    / (sigma1[..., None, None] * sigma2[..., None, None])
                    + (self.Y_grids[..., :, :] - mu2[..., None, None]) * (
                            self.Y_grids[..., :, :] - mu2[..., None, None])
                    / (sigma2[..., None, None] * sigma2[..., None, None]))
            g = (a * (torch.exp(b))).type(torch.FloatTensor).cuda()
            # print(g)
            assert not torch.any(torch.isnan(g)) or torch.any(torch.isinf(g)),print("a:",a,"b:",b)
            value = AIF_ROI * g[..., None, :, :]
            assert not torch.any(torch.isnan(value))
        else:
            a = 1. / (2. * torch.pi * sigma1[..., None, None] * sigma2[..., None, None])
            b = -1. / 2. * ((self.X_grids[..., :, :] * self.X_grids[..., :, :]) / (
                    sigma1[..., None, None] * sigma1[..., None, None])
                            + (self.Y_grids[..., :, :] * self.Y_grids[..., :, :]) / (
                                    sigma2[..., None, None] * sigma2[..., None, None]))
            g = (a * (torch.exp(b))).type(torch.FloatTensor).cuda()
            # print(g)
            assert not torch.any(torch.isnan(g)) or torch.any(torch.isinf(g)),print("a:",a,"b:",b)
            value = AIF_ROI * g[..., None, :, :]
            assert not torch.any(torch.isnan(value))
        time2 = time.time()
        value = value.reshape(value.shape[0], 3, -1)
        assert not torch.any(torch.isnan(value))
        g = g.reshape(g.shape[0], -1)
        assert not torch.any(torch.isnan(g))
        imgB = (torch.sum(value, dim=2)[..., :]) / ((torch.sum(g, dim=1)[..., None]) + 0.00001)

        if torch.any(torch.isnan(imgB)):
            print(torch.sum(value, dim=2)[..., :])
            print(torch.sum(g, dim=1)[..., None])
            index = torch.where(torch.isnan(imgB))
            # for idx in index:
            #     i = idx[0]
            #     g = g[i]
            #     s1 = sigma1[i]
            #     s2 = sigma2[i]
            #     print("idx:", i, "g:", g, "s1:", s1, "s2:", s2)
            #     u1 = mu1[i]
            #     u2 = mu2[i]
            #     r = rho[i]
            #     print("u1:", u1, "u2:", u2, "r:", r)
            #     print("gsum:", torch.sum(g, dim=1)[..., None])
            #     print("vsum:", torch.sum(value, dim=2)[..., :])
        assert not torch.any(torch.isnan(imgB))

        time3 = time.time()
        # print("1",time2-time1)
        # print("2", time3 - time2)
        # print("3", time4 - time3)
        # print("4", time5 - time4)
        # times=time2-time1
        # print("gaussian:", times)
        return imgB


def init_linear_weights(m):
    if isinstance(m, nn.Linear):
        if m.weight.shape[0] in [2, 3]:
            nn.init.xavier_normal_(m.weight, 0.1)
        else:
            nn.init.xavier_normal_(m.weight)
        # nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


class NeRFAll(nn.Module):
    def __init__(self, args, camera_model=None, camera_mode=False):
        super().__init__()
        self.args = args
        self.embed_fn, self.input_ch = get_embedder(args.multires, args.i_embed)
        self.input_ch_views = 0
        self.camera_model = camera_model
        self.camera_mode = camera_mode

        self.embeddirs_fn = None
        if args.use_viewdirs:
            self.embeddirs_fn, self.input_ch_views = get_embedder(args.multires_views, args.i_embed)

        self.output_ch = 5 if args.N_importance > 0 else 4

        skips = [4]
        self.mlp_coarse = NeRF(
            D=args.netdepth, W=args.netwidth,
            input_ch=self.input_ch, output_ch=self.output_ch, skips=skips,
            input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs)

        self.mlp_fine = None
        if args.N_importance > 0:
            self.mlp_fine = NeRF(
                D=args.netdepth_fine, W=args.netwidth_fine,
                input_ch=self.input_ch, output_ch=self.output_ch, skips=skips,
                input_ch_views=self.input_ch_views, use_viewdirs=args.use_viewdirs)

        activate = {'relu': torch.relu, 'sigmoid': torch.sigmoid, 'exp': torch.exp, 'none': lambda x: x,
                    'sigmoid1': lambda x: 1.002 / (torch.exp(-x) + 1) - 0.001,
                    'softplus': lambda x: nn.Softplus()(x - 1)}
        self.rgb_activate = activate[args.rgb_activate]
        self.sigma_activate = activate[args.sigma_activate]
        self.tonemapping = ToneMapping(args.tone_mapping_type)

    def mlpforward(self, inputs, viewdirs, mlp, netchunk=1024 * 64):
        """Prepares inputs and applies network 'fn'.
            """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = self.embed_fn(inputs_flat)

        if viewdirs is not None:
            input_dirs = viewdirs[:, None].expand(inputs.shape)
            input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
            embedded_dirs = self.embeddirs_fn(input_dirs_flat)
            embedded = torch.cat([embedded, embedded_dirs], -1)

        # batchify execution
        if netchunk is None:
            outputs_flat = mlp(embedded)
        else:
            outputs_flat = torch.cat([mlp(embedded[i:i + netchunk]) for i in range(0, embedded.shape[0], netchunk)], 0)

        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs

    def raw2outputs(self, raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            raw: [num_rays, num_samples along ray, 4]. Prediction from model.
            z_vals: [num_rays, num_samples along ray]. Integration time.
            rays_d: [num_rays, 3]. Direction of each ray.
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
        """

        def raw2alpha(raw_, dists_, act_fn):
            alpha_ = - torch.exp(-act_fn(raw_) * dists_) + 1.
            return torch.cat([alpha_, torch.ones_like(alpha_[:, 0:1])], dim=-1)

        dists = z_vals[..., 1:] - z_vals[..., :-1]  # [N_rays, N_samples - 1]
        # dists = torch.cat([dists, torch.tensor([1e10]).expand(dists[..., :1].shape)], -1)

        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        rgb = self.rgb_activate(raw[..., :3])
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn_like(raw[..., :-1, 3]) * raw_noise_std

            # Overwrite randomly sampled data if pytest
            if pytest:
                np.random.seed(0)
                noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
                noise = torch.tensor(noise)

        density = self.sigma_activate(raw[..., :-1, 3] + noise)
        if not self.training and self.args.render_rmnearplane > 0:
            mask = z_vals[:, 1:]
            mask = mask > self.args.render_rmnearplane / 128
            mask = mask.type_as(density)
            density = mask * density

        alpha = - torch.exp(- density * dists) + 1.
        alpha = torch.cat([alpha, torch.ones_like(alpha[:, 0:1])], dim=-1)

        # alpha = raw2alpha(raw[..., :-1, 3] + noise, dists, act_fn=self.sigma_activate)  # [N_rays, N_samples]
        # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
        weights = alpha * \
                  torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), - alpha + (1. + 1e-10)], -1), -1)[:, :-1]

        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        depth_map = torch.sum(weights * z_vals, -1)

        # disp_map = 1. / torch.clamp_min(depth_map, 1e-10)
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map, density, acc_map, weights, depth_map

    def render_rays(self,
                    ray_batch,
                    N_samples,
                    retraw=False,
                    lindisp=False,
                    perturb=0.,
                    N_importance=0,
                    white_bkgd=False,
                    raw_noise_std=0.,
                    pytest=False):
        """Volumetric rendering.
        Args:
          ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
          N_samples: int. Number of different times to sample along each ray.
          retraw: bool. If True, include model's raw, unprocessed predictions.
          lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
          perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
            random points in time.
          N_importance: int. Number of additional times to sample along each ray.
            These samples are only passed to network_fine.
          white_bkgd: bool. If True, assume a white background.
          raw_noise_std: ...
          verbose: bool. If True, print more debugging info.
        """
        N_rays = ray_batch.shape[0]
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
        viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
        near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

        t_vals = torch.linspace(0., 1., steps=N_samples).type_as(rays_o)
        if not lindisp:
            z_vals = near * (1. - t_vals) + far * (t_vals)
        else:
            z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        z_vals = z_vals.expand([N_rays, N_samples])

        if perturb > 0.:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).type_as(rays_o)

            # Pytest, overwrite u with numpy's fixed random numbers
            if pytest:
                np.random.seed(0)
                t_rand = np.random.rand(*list(z_vals.shape))
                t_rand = torch.tensor(t_rand)

            z_vals = lower + (upper - lower) * t_rand

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

        #     raw = run_network(pts)
        raw = self.mlpforward(pts, viewdirs, self.mlp_coarse)
        rgb_map, density_map, acc_map, weights, depth_map = self.raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                             white_bkgd, pytest=pytest)

        if N_importance > 0:
            rgb_map_0, depth_map_0, acc_map_0, density_map0 = rgb_map, depth_map, acc_map, density_map

            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :,
                                                                None]  # [N_rays, N_samples + N_importance, 3]

            mlp = self.mlp_coarse if self.mlp_fine is None else self.mlp_fine
            raw = self.mlpforward(pts, viewdirs, mlp)

            rgb_map, density_map, acc_map, weights, depth_map = self.raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                                 white_bkgd, pytest=pytest)

        ret = {'rgb_map': rgb_map, 'depth_map': depth_map, 'acc_map': acc_map, 'density_map': density_map}
        if retraw:
            ret['raw'] = raw
        if N_importance > 0:
            ret['rgb0'] = rgb_map_0
            ret['depth0'] = depth_map_0
            ret['acc0'] = acc_map_0
            ret['density0'] = density_map0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

        for k in ret:
            if torch.isnan(ret[k]).any():
                print(f"! [Numerical Error] {k} contains nan.")
            if torch.isinf(ret[k]).any():
                print(f"! [Numerical Error] {k} contains inf.")
        return ret

    def forward(self, H, W, K, chunk=1024 * 32, rays=None, rays_info=None, poses=None, camera_rays_info=None,
                camera_AIF=None, **kwargs):
        """
        render rays or render poses, rays and poses should atleast specify one
        calling model.train() to render rays, where rays, rays_info, should be specified
        calling model.eval() to render an image, where poses should be specified

        optional args:
        force_naive: when True, will only run the naive NeRF, even if the cameraNeRF is specified

        """
        # training
        if self.training:
            # camera mode, one ray to one CoC
            if self.camera_model and self.camera_mode:
                # time1 = time.time()
                blur_result = self.camera_model(camera_rays_info, camera_AIF)
                blur_rgb1, blur_rgb2,blur_rgb3 = blur_result['blur_img1'], blur_result['blur_img2'], blur_result['blur_img3']
                # time2 = time.time()
                # print(f"Time| camera: {time2-time1:.5f}")
                return blur_rgb1, blur_rgb2, blur_rgb3
            else:
                # time0 = time.time()
                rgb, depth, acc, extras = self.render(H, W, K, chunk, rays, **kwargs)
                # time1 = time.time()
                # print(f"Time| nerf: {time1-time0:.5f}")
                return rgb, extras['rgb0'], {}

        #  evaluation
        else:
            blur_rgbs = []
            blur_rgb0s = []
            blur_rgb1s = []
            paras=[]
            ks = []

            t = time.time()
            test_info = {}
            for i in range(camera_rays_info['camera_rays'].shape[0]):
                # for i in range(1):
                print(i, time.time() - t)
                t = time.time()
                sh = camera_rays_info['camera_rays'][i].shape
                # Create ray batch
                test_info['camera_rays'] = torch.reshape(camera_rays_info['camera_rays'][i], [-1, 3])
                test_info['camera_rays_pos'] = torch.reshape(camera_rays_info['camera_rays_pos'][i], [-1, 2])
                test_info['camera_pix_pos'] = torch.reshape(camera_rays_info['camera_pix_pos'][i], [-1, 2])
                test_info['camera_focus_rgbsf'] = torch.reshape(camera_rays_info['camera_focus_rgbsf'][i], [-1, 3])
                test_info['camera_depthsf'] = torch.reshape(camera_rays_info['camera_depthsf'][i], [-1, 1])
                # test_info['camera_AIFkc'] = torch.reshape(camera_rays_info['camera_AIFkc'][i], [-1, 3, 9])
                test_info['camera_AIFsf'] = torch.reshape(camera_rays_info['camera_AIFsf'][i], [-1, 3])
                test_info['camera_fd'] = torch.reshape(camera_rays_info['camera_fd'][i], [-1, 1])
                test_info['camera_focus_images_idx'] = torch.reshape(camera_rays_info['camera_focus_images_idx'][i],
                                                                     [-1])

                # Batchfy and Render and reshape
                all_ret = {}
                for i in range(0, test_info['camera_rays'].shape[0], chunk):
                    # print(i)
                    ret = self.camera_model({k: v[i:i + chunk] for k, v in test_info.items()}, camera_AIF)
                    for k in ret:
                        if k not in all_ret:
                            all_ret[k] = []
                        all_ret[k].append(ret[k])
                all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

                for k in all_ret:
                    k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
                    all_ret[k] = torch.reshape(all_ret[k], k_sh)

                k_extract = ['blur_img1', 'blur_img2','blur_img3','paras']
                ret_list = [all_ret[k] for k in k_extract]
                ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}

                blur_rgb, blur_rgb0,blur_rgb1,para = ret_list
                # k2, k1 = ret_dict['k2'], ret_dict['k1']

                # ray_show_img(blur_rgb, blur_rgb.shape[0], blur_rgb.shape[1], is_torch=True)

                blur_rgbs.append(blur_rgb)
                blur_rgb0s.append(blur_rgb0)
                blur_rgb1s.append(blur_rgb1)
                paras.append(para)
                # ks.append([k2,k1])
                if i == 0:
                    print(blur_rgb.shape)

            blur_rgbs1 = torch.stack(blur_rgbs, 0)
            blur_rgbs2 = torch.stack(blur_rgb0s, 0)
            blur_rgbs3 = torch.stack(blur_rgb1s, 0)
            paras = torch.stack(paras, 0)
            return blur_rgbs1, blur_rgbs2,blur_rgbs3, paras

    def render(self, H, W, K, chunk, rays=None, c2w=None, ndc=True,
               near=0., far=1.,
               use_viewdirs=False, c2w_staticcam=None,
               **kwargs):  # the render function
        """Render rays
            Args:
              H: int. Height of image in pixels.
              W: int. Width of image in pixels.
              focal: float. Focal length of pinhole camera.
              chunk: int. Maximum number of rays to process simultaneously. Used to
                control maximum memory usage. Does not affect final results.
              rays: array of shape [2, batch_size, 3]. Ray origin and direction for
                each example in batch.
              c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
              ndc: bool. If True, represent ray origin, direction in NDC coordinates.
              near: float or array of shape [batch_size]. Nearest distance for a ray.
              far: float or array of shape [batch_size]. Farthest distance for a ray.
              use_viewdirs: bool. If True, use viewing direction of a point in space in model.
              c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
               camera while using other c2w argument for viewing directions.
            Returns:
              rgb_map: [batch_size, 3]. Predicted RGB values for rays.
              disp_map: [batch_size]. Disparity map. Inverse of depth.
              acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
              extras: dict with everything returned by render_rays().
            """
        rays_o, rays_d = rays[..., 0], rays[..., 1]

        if use_viewdirs:
            # provide ray directions as input
            viewdirs = rays_d
            if c2w_staticcam is not None:
                # special case to visualize effect of viewdirs
                rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

        sh = rays_d.shape  # [..., 3]
        if ndc:
            # for forward facing scenes
            rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

        # Create ray batch
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
        rays = torch.cat([rays_o, rays_d, near, far], -1)
        if use_viewdirs:
            rays = torch.cat([rays, viewdirs], -1)

        # Batchfy and Render and reshape
        all_ret = {}
        for i in range(0, rays.shape[0], chunk):
            ret = self.render_rays(rays[i:i + chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

        for k in all_ret:
            k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_sh)

        k_extract = ['rgb_map', 'depth_map', 'acc_map']
        ret_list = [all_ret[k] for k in k_extract]
        ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
        return ret_list + [ret_dict]

    def render_path(self, H, W, K, chunk, render_poses, render_kwargs, render_factor=0, ):
        """
        render image specified by the render_poses
        """
        if render_factor != 0:
            # Render downsampled for speed
            H = H // render_factor
            W = W // render_factor

        rgbs = []
        depths = []

        t = time.time()
        for i, c2w in enumerate(render_poses):
            print(i, time.time() - t)
            t = time.time()
            rays = get_rays(H, W, K, c2w)
            rays = torch.stack(rays, dim=-1)
            rgb, depth, acc, extras = self.render(H, W, K, chunk=chunk, rays=rays, c2w=c2w[:3, :4], **render_kwargs)

            rgbs.append(rgb)
            depths.append(depth)
            if i == 0:
                print(rgb.shape, depth.shape)

        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)

        return rgbs, depths

    def render_subpath(self, H, W, K, chunk, render_poses, render_point, images_indices, render_kwargs,
                       render_factor=0):
        """

        """
        if render_factor != 0:
            # Render downsampled for speed
            H = H // render_factor
            W = W // render_factor

        rgbs = []
        depths = []
        weights = []

        t = time.time()

        rayx, rayy = torch.meshgrid(torch.linspace(0, W - 1, W),
                                    torch.linspace(0, H - 1, H))
        rayx = rayx.t().reshape(-1, 1) + HALF_PIX
        rayy = rayy.t().reshape(-1, 1) + HALF_PIX

        for imgidx, c2w in zip(images_indices, render_poses):

            i = int(imgidx.item())
            print(i, time.time() - t)
            t = time.time()
            rays = get_rays(H, W, K, c2w)
            rays = torch.stack(rays, dim=-1).reshape(H * W, 3, 2)

            rays_info = {}

            if self.kernelsnet.require_depth:
                with torch.no_grad():
                    rgb, depth, acc, extras = self.render(H, W, K, chunk, rays, **render_kwargs)
                    rays_info["ray_depth"] = depth[..., None]

            i = i if i < self.kernelsnet.num_img else 1
            rays_info["images_idx"] = torch.ones_like(rays[:, 0:1, 0]).type(torch.long) * i
            rays_info["rays_x"] = rayx
            rays_info["rays_y"] = rayy

            new_rays, weight, _ = self.kernelsnet(H, W, K, rays, rays_info)

            new_rays = new_rays[:, render_point]
            weight = weight[:, render_point]
            rgb, depth, acc, extras = self.render(H, W, K, chunk=chunk, rays=new_rays.reshape(-1, 3, 2),
                                                  c2w=c2w[:3, :4], **render_kwargs)

            rgbs.append(rgb.reshape(H, W, 3))
            depths.append(depth.reshape(H, W))
            weights.append(weight.reshape(H, W))
            if i == 0:
                print(rgb.shape, depth.shape)

        rgbs = torch.stack(rgbs, 0)
        depths = torch.stack(depths, 0)
        weights = torch.stack(weights, 0)

        return rgbs, depths, weights
