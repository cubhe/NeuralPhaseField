import os
import time

import cv2
import imageio
from tensorboardX import SummaryWriter

from NeRF import *
from load_data import load_phase_data
from run_nerf_helpers import *
from metrics import compute_img_metric

# np.random.seed(0)
DEBUG = False

def config_parser():
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, default='./configs/demo_blurball.txt',
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', required=True,
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, required=True,
                        help='input data directory')
    parser.add_argument("--datadownsample", type=float, default=-1,
                        help='if downsample > 0, means downsample the image to scale=datadownsample')
    parser.add_argument("--tbdir", type=str, required=True,
                        help="tensorboard log directory")
    parser.add_argument("--num_gpu", type=int, default=1,
                        help=">1 will use DataParallel")
    parser.add_argument("--torch_hub_dir", type=str, default='',
                        help=">1 will use DataParallel")


    parser.add_argument("--data_path", type=str, default='', help="")

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    # generate N_rand # of rays, divide into chunk # of batch
    # then generate chunk * N_samples # of points, divide into netchunk # of batch
    parser.add_argument("--chunk", type=int, default=1024 * 32 * 6,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024 * 64 * 2,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_iters", type=int, default=50000,
                        help='number of iteration')
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--rgb_activate", type=str, default='sigmoid',
                        help='activate function for rgb output, choose among "none", "sigmoid"')
    parser.add_argument("--sigma_activate", type=str, default='relu',
                        help='activate function for sigma output, choose among "relu", "softplue"')

    # ===============================
    # Camera Model Config
    # ===============================
    parser.add_argument("--camera_type", type=str, default='phasenerf',
                        help='choose among <none>, <phasenerf>')
    parser.add_argument("--focus_distance", type=int, default=0.9,
                        help='the focus distance of the object, the range of values is the range of depth.')
    parser.add_argument("--kernel_size", type=int, default=31,
                        help='blur kernel size')
    parser.add_argument("--camera_datadir", type=str, default='./dataset/allin3d/388',
                        help='input camera model data directory')
    parser.add_argument("--camera_test_datadir", type=str, default='./dataset/allin3d/388',
                        help='input camera model data directory to test')
    parser.add_argument("--object_category", type=str, default='No388_WC_WD_abc_new_pos',
                        help='the object category in the data set used this time')
    parser.add_argument("--color_info", type=bool, default=False,
                        help='the object category in the data set used this time')
    parser.add_argument("--depth_info", type=bool, default=False,
                        help='the object category in the data set used this time')
    parser.add_argument("--MLP123", type=int, default=3,
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--camera_render_only", type=bool, default=False,
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--allin3dhold", type=int, default=8,
                        help='will take every 1/N images as Allin3D test set, paper uses 6')
    # ===============================
    # Kernel optimizing
    # ===============================
    parser.add_argument("--kernel_type", type=str, default='kernel',
                        help='choose among <none>, <itsampling>, <sparsekernel>')
    parser.add_argument("--kernel_isglobal", action='store_true',
                        help='if specified, the canonical kernel position is global')
    parser.add_argument("--kernel_start_iter", type=int, default=0,
                        help='start training kernel after # iteration')
    parser.add_argument("--kernel_ptnum", type=int, default=5,
                        help='the number of sparse locations in the kernels '
                             'that involves computing the final color of ray')
    parser.add_argument("--kernel_random_hwindow", type=float, default=0.25,
                        help='randomly displace the predicted ray position')
    parser.add_argument("--kernel_img_embed", type=int, default=32,
                        help='the dim of image laten code')
    parser.add_argument("--kernel_rand_dim", type=int, default=2,
                        help='dimensions of input random number which uniformly sample from (0, 1)')
    parser.add_argument("--kernel_rand_embed", type=int, default=3,
                        help='embed frequency of input kernel coordinate')
    parser.add_argument("--kernel_rand_mode", type=str, default='float',
                        help='<float>, <<int#, such as<int5>>>, <fix>')
    parser.add_argument("--kernel_random_mode", type=str, default='input',
                        help='<input>, <output>')
    parser.add_argument("--kernel_spatial_embed", type=int, default=0,
                        help='the dim of spatial coordinate embedding')
    parser.add_argument("--kernel_depth_embed", type=int, default=0,
                        help='the dim of depth coordinate embedding')
    parser.add_argument("--kernel_hwindow", type=int, default=10,
                        help='the max window of the kernel (sparse location will lie inside the window')
    parser.add_argument("--kernel_pattern_init_radius", type=float, default=0.1,
                        help='the initialize radius of init pattern')
    parser.add_argument("--kernel_num_hidden", type=int, default=3,
                        help='the number of hidden layer')
    parser.add_argument("--kernel_num_wide", type=int, default=64,
                        help='the wide of hidden layer')
    parser.add_argument("--kernel_shortcut", action='store_true',
                        help='if yes, add a short cut to the network')
    parser.add_argument("--align_start_iter", type=int, default=0,
                        help='start iteration of the align loss')
    parser.add_argument("--align_end_iter", type=int, default=1e10,
                        help='end iteration of the align loss')
    parser.add_argument("--kernel_align_weight", type=float, default=0,
                        help='align term weight')
    parser.add_argument("--prior_start_iter", type=int, default=0,
                        help='start iteration of the prior loss')
    parser.add_argument("--prior_end_iter", type=int, default=1e10,
                        help='end iteration of the prior loss')
    parser.add_argument("--kernel_prior_weight", type=float, default=0,
                        help='weight of prior loss (regularization)')
    parser.add_argument("--sparsity_start_iter", type=int, default=0,
                        help='start iteration of the sparsity loss')
    parser.add_argument("--sparsity_end_iter", type=int, default=1e10,
                        help='end iteration of the sparsity loss')
    parser.add_argument("--kernel_sparsity_type", type=str, default='tv',
                        help='type of sparse gradient loss', choices=['tv', 'normalize', 'robust'])
    parser.add_argument("--kernel_sparsity_weight", type=float, default=0,
                        help='weight of sparsity loss')
    parser.add_argument("--kernel_spatialvariant_trans", action='store_true',
                        help='if true, optimize spatial variant 3D translation of each sampling point')
    parser.add_argument("--kernel_global_trans", action='store_true',
                        help='if true, optimize global 3D translation of each sampling point')
    parser.add_argument("--tone_mapping_type", type=str, default='none',
                        help='the tone mapping of linear to LDR color space, <none>, <gamma>, <learn>')

    ####### render option, will not effect training ########
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_multipoints", action='store_true',
                        help='render sub image that reconstruct the blur image')
    parser.add_argument("--render_rmnearplane", type=int, default=0,
                        help='when render, set the density of nearest plane to 0')
    parser.add_argument("--render_focuspoint_scale", type=float, default=1.,
                        help='scale the focal point when render')
    parser.add_argument("--render_radius_scale", type=float, default=1.,
                        help='scale the radius of the camera path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
    parser.add_argument("--render_epi", action='store_true',
                        help='render the video with epi path')


    # ######### Unused params from the original ###########
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')
    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--camera_dataset_type", type=str, default='Allin3D',
                        help='options: Allin3D')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')
    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ################# logging/saving options ##################
    parser.add_argument("--i_print", type=int, default=250,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_tensorboard", type=int, default=250,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=2000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=2000,help='frequency of testset saving')
    parser.add_argument("--i_video", type=int, default=10000,help='frequency of render_poses video saving')

    return parser

def data_process(args, train_datas, images, i_train, H, W, K, poses):
    # nerf images
    imagesf = images
    images = (images * 255).astype(np.uint8)
    images_idx = np.arange(0, len(images))

    if args.datadownsample > 0:
        images_train = np.stack([cv2.resize(img_, None, None,
                                            1 / args.datadownsample, 1 / args.datadownsample,
                                            cv2.INTER_AREA) for img_ in imagesf], axis=0)
    else:
        images_train = imagesf

    num_img, hei, wid, _ = images_train.shape
    print(f"train on image sequence of len = {num_img}, {wid}x{hei}")
    k_train = np.array([K[0, 0] * wid / W, 0, K[0, 2] * wid / W,
                        0, K[1, 1] * hei / H, K[1, 2] * hei / H,
                        0, 0, 1]).reshape(3, 3).astype(K.dtype)
    # For random ray batching
    print('get rays')
    rays = np.stack([get_rays_np(hei, wid, k_train, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
    rays = np.transpose(rays, [0, 2, 3, 1, 4])
    train_datas['rays'] = rays[i_train].reshape(-1, 2, 3)

    xs, ys = np.meshgrid(np.arange(wid, dtype=np.float32), np.arange(hei, dtype=np.float32), indexing='xy')
    xs = np.tile((xs[None, ...] + HALF_PIX) * W / wid, [num_img, 1, 1])
    ys = np.tile((ys[None, ...] + HALF_PIX) * H / hei, [num_img, 1, 1])
    train_datas['rays_x'], train_datas['rays_y'] = xs[i_train].reshape(-1, 1), ys[i_train].reshape(-1, 1)

    train_datas['rgbsf'] = images_train[i_train].reshape(-1, 3)

    images_idx_tile = images_idx.reshape((num_img, 1, 1))
    images_idx_tile = np.tile(images_idx_tile, [1, hei, wid])
    train_datas['images_idx'] = images_idx_tile[i_train].reshape(-1, 1).astype(np.int64)

    return train_datas, imagesf


def camera_data_process(args, camera_train_datas, camera_test_datas, focus_images, depths,AIFs, AIF_image, fd,ids,
                        i_train_camera):

    # phasenerf images(allin3d)
    focus_imagessf = focus_images
    focus_images = (focus_images * 255).astype(np.uint8)
    focus_images_idx = np.arange(0, len(focus_images))

    if args.depth_info == False:
        depths=1.0*np.ones(depths.shape).astype('float32')#heka0215
    depthssf = depths
    depths = depths.astype(np.uint8)

    if args.color_info == False:
        AIFs = 0.3*np.ones(AIFs.shape).astype('float32')##heka0222
    AIFssf = AIFs
    AIFs = (AIFs * 1).astype(np.uint8)

    fd = fd/1000

    if args.datadownsample > 0:
        focus_images_train = np.stack([cv2.resize(img_, None, None,
                                                  1 / args.datadownsample, 1 / args.datadownsample,
                                                  cv2.INTER_AREA) for img_ in focus_imagessf], axis=0)
        depths_train = np.stack([cv2.resize(img_, None, None,
                                            1 / args.datadownsample, 1 / args.datadownsample,
                                            cv2.INTER_AREA) for img_ in depthssf], axis=0)
        AIF_image_train = np.stack([cv2.resize(img_, None, None,
                                               1 / args.datadownsample, 1 / args.datadownsample,
                                               cv2.INTER_AREA) for img_ in AIFssf], axis=0)
    else:
        focus_images_train = focus_imagessf
        depths_train = depthssf
        AIF_image_train = AIFssf

    focus_num_img, focus_hei, focus_wid, _ = focus_images_train.shape
    print(f"train on focus_image sequence of len = {focus_num_img}, {focus_wid}x{focus_hei}")

    # For random ray batching(Camera NeRF)
    print('get camera rays (train and test)')

    camera_rays = np.zeros((focus_num_img, focus_hei, focus_wid, 3))  # [N, H, W, 3]
    camera_train_datas['camera_rays'] = camera_rays[i_train_camera].reshape(-1, 3)
    camera_test_datas['camera_rays'] = camera_rays

    camera_xs0, camera_ys0 = np.meshgrid(np.arange(focus_wid, dtype=np.float32), np.arange(focus_hei, dtype=np.float32),
                                         indexing='xy')
    camera_pix_pos0 = np.stack([camera_xs0, camera_ys0], axis=-1).astype(np.int16)
    camera_xs = np.tile(camera_xs0[None, ...], [focus_num_img, 1, 1])
    camera_ys = np.tile(camera_ys0[None, ...], [focus_num_img, 1, 1])
    camera_pix_pos = np.stack([camera_xs, camera_ys], axis=-1).astype(np.int16)
    camera_train_datas['camera_pix_pos'] = camera_pix_pos[i_train_camera].reshape(-1, 2)
    camera_test_datas['camera_pix_pos'] = camera_pix_pos

    camera_AIFROI = pixpos2AIFROI(camera_pix_pos0, args.kernel_size, AIF_image)  # N*H*W*KS*KS
    # camera_train_datas['camera_AIFROI'] = camera_AIFROI
    # camera_test_datas['camera_AIFROI'] = camera_AIFROI

    # CoC position filed lies in [0,1]
    camera_xs, camera_ys = ((camera_xs0 - focus_wid / 2)) / (focus_wid / 2), (
        (camera_ys0 - focus_hei / 2)) / (focus_hei / 2)
    camera_xs = np.tile(camera_xs[None, ...], [focus_num_img, 1, 1])
    camera_ys = np.tile(camera_ys[None, ...], [focus_num_img, 1, 1])
    camera_rays_pos = np.stack([camera_xs, camera_ys], axis=-1)
    camera_train_datas['camera_rays_pos'] = camera_rays_pos[i_train_camera].reshape(-1, 2)
    camera_test_datas['camera_rays_pos'] = camera_rays_pos

    camera_train_datas['camera_focus_rgbsf'] = focus_images_train[i_train_camera].reshape(-1, 3)
    camera_test_datas['camera_focus_rgbsf'] = focus_images_train

    # depths_train = np.tile(depths_train[None, ...], [focus_num_img, 1, 1, 1])
    camera_train_datas['camera_depthsf'] = depths_train[i_train_camera].reshape(-1, 1)
    camera_test_datas['camera_depthsf'] = depths_train

    # AIF_image_train = np.tile(AIF_image_train[None, ...], [focus_num_img, 1, 1, 1])
    camera_train_datas['camera_AIFsf'] = AIF_image_train[i_train_camera].reshape(-1, 3)
    camera_test_datas['camera_AIFsf'] = AIF_image_train

    camera_fd = fd.reshape(
        (focus_num_img, 1, 1))
    camera_fd = np.tile(camera_fd, [1, focus_hei, focus_wid])
    camera_train_datas['camera_fd'] = camera_fd[i_train_camera].reshape(-1, 1).astype(np.float32)
    camera_test_datas['camera_fd'] = camera_fd.astype(np.float32)

    focus_images_idx_tile = ids.reshape((focus_num_img, 1, 1))
    focus_images_idx_tile = np.tile(focus_images_idx_tile, [1, focus_hei, focus_wid])
    camera_train_datas['camera_focus_images_idx'] = focus_images_idx_tile[i_train_camera].reshape(-1, 1).astype(
        np.int64)
    camera_test_datas['camera_focus_images_idx'] = focus_images_idx_tile.astype(np.int64)

    return camera_train_datas, camera_test_datas, camera_AIFROI, focus_imagessf, depthssf, AIFssf


def train(arg):
    parser = config_parser()
    args = parser.parse_args()

    # Create log dir and copy the config file
    basedir = args.basedir
    tensorboardbase = args.tbdir
    # expname = args.expname
    expname = args.object_category
    test_metric_file = os.path.join(basedir, expname, 'test_metrics.txt')
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    os.makedirs(os.path.join(tensorboardbase, expname), exist_ok=True)
    tensorboard = SummaryWriter(os.path.join(tensorboardbase, expname))
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None and not args.render_only:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
        with open(test_metric_file, 'a') as file:
            file.write(open(args.config, 'r').read())
            file.write("\n============================\n"
                       "||\n"
                       "\\/\n")

    time0 = time.time()
    if arg is not None:
        args.camera_datadir = arg[0]
        args.camera_test_datadir = arg[0]
        args.object_category = arg[1]
        print(args.camera_datadir,args.camera_test_datadir,args.object_category)
        if len(arg) > 2:
            args.color_info=arg[2]
            args.depth_info=arg[3]


    # Load Phase data
    if args.camera_dataset_type == 'phase':
        #load data including images and illuminate source location
        image,light_loc,ids=load_phase_data(args.data_path)
        print('Loaded phase_data',ids, image.shape, light_loc.shape)
        H, W= image.shape[0], image.shape[1]
    else:
        print('Unknown camera dataset type', args.camera_dataset_type, 'exiting')
        return

    #Calculate the direction for each light ray
    #But we need to transfer them into the same coordinate
    #re-arrange the data from [image,illuminate location]
    #into [pixel(3), direction(3), id(1)] where the id is the image serial naumber
    data=np.zero(7,H*W*ids)
    num=0
    for k in range(ids):
        for i in range(H):
            for j in range(W):
                data[:,num]=pixel2position(image[k,i,j,:],light_loc[k])
                data[6,num]=k
                num+=1

    #divided the data in to test and train
    #need complement
    train_phase,test_phase=Data_shuffle()


    # Define The Phase Model

    if args.camera_type == 'phasenerf':
        phase_model = phasenerf(args)
    else:
        raise RuntimeError(f"camera_type {args.camera_type} not recognized")

    # Create nerf model
    nerf = NPRF(args, camera_model=camera_model, camera_mode=True)
    nerf = nn.DataParallel(nerf, list(range(args.num_gpu)))
    optim_params = nerf.parameters()
    optimizer = torch.optim.Adam(params=optim_params, lr=args.lrate, betas=(0.9, 0.999))
    start = 0

    # Load Checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if
                 '.tar' in f]
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # Load model
        smart_load_state_dict(nerf, ckpt)

    ###########9.25################

    # figuring out the train/test configuration
    render_kwargs_train = {
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'N_samples': args.N_samples,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }
    # # NDC only good for LLFF-style forward facing data
    # if args.no_ndc:  # args.dataset_type != 'llff' or
    #     print('Not ndc!')
    #     render_kwargs_train['ndc'] = False
    #     render_kwargs_train['lindisp'] = args.lindisp
    # render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    # render_kwargs_test['perturb'] = False
    # render_kwargs_test['raw_noise_std'] = 0.

    # visualize_motionposes(H, W, K, nerf, 2)
    # visualize_kernel(H, W, K, nerf, 5)
    # visualize_itsample(H, W, K, nerf)
    # visualize_kmap(H, W, K, nerf, img_idx=1)



    global_step = start

    # Move testing data to GPU

    nerf = nerf.cuda()
    train_phase_cuda = torch.tensor(train_phase).cuda()
    test_phase_cuda = torch.tensor(test_phase).cuda()

    # Short circuit if only rendering out from trained model
    if args.camera_render_only:
        print('RENDER ONLY')
        ############### implementation needed############

    # ============================================
    # Prepare ray dataset if batching random rays
    # ============================================
    N_rand = args.N_rand
    train_datas = {}
    camera_train_datas = {}
    camera_test_datas = {}

    # nerf datas processing
    train_datas, imagesf = data_process(args, train_datas, images, i_train, H, W, K, poses)
    print('shuffle rays')
    shuffle_idx = np.random.permutation(len(train_datas['rays']))
    train_datas = {k: v[shuffle_idx] for k, v in train_datas.items()}
    print('done')

    # camera-nerf datas processing
    camera_train_datas, camera_test_datas, camera_AIFROI, focus_imagessf, depthssf, AIF_imagesf = camera_data_process(
        args, camera_train_datas, camera_test_datas,
        focus_images, depths, AIFs,AIF_image, fd, ids,i_train_camera)
    print('shuffle camera rays (train)')
    camera_shuffle_idx = np.random.permutation(len(camera_train_datas['camera_rays']))
    camera_train_datas = {k: v[camera_shuffle_idx] for k, v in camera_train_datas.items() if 'camera' in k}
    print('done')

    i_batch = 0

    # Move training data to GPU
    # images = torch.tensor(images).cuda()
    # imagesf = torch.tensor(imagesf).cuda()
    # focus_images = torch.tensor(focus_images).cuda()
    # focus_imagessf = torch.tensor(focus_imagessf).cuda()
    # depths = torch.tensor(depths).cuda()
    # depthssf = torch.tensor(depthssf).cuda()
    # AIF_image = torch.tensor(AIF_image).cuda()
    # AIF_imagesf = torch.tensor(AIF_imagesf).cuda()

    poses = torch.tensor(poses).cuda()
    train_datas = {k: torch.tensor(v).cuda() for k, v in train_datas.items()}
    camera_train_datas = {k: torch.tensor(v).cuda() for k, v in camera_train_datas.items()}
    camera_test_datas = {k: torch.tensor(v).cuda() for k, v in camera_test_datas.items()}
    camera_AIFROI = {k: torch.tensor(v).cuda() for k, v in camera_AIFROI.items()}
    # camera_AIFROI = torch.tensor(camera_AIFROI).cuda()

    N_iters = args.N_iters + 1
    print('Begin')
    print('TRAIN views(NeRF) are', i_train, '\tTRAIN views(Camera NeRF) are', i_train_camera)
    print('TEST views(NeRF) are', i_test, '\tTEST views(Camera NeRF) are', i_test_camera)
    print('VAL views(NeRF) are', i_val, '\tVAL views(Camera NeRF) are', i_val_camera)

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    start = start + 1
    time0 = time.time()
    for i in range(start, N_iters):
        # if global_step>=23999:
        #     if os.path.isfile('./log/'+args.object_category+'/testset_024000/paras.npy'):
        #         print("文件存在")
        #         print(global_step)
        #         return 0



        # Sample random ray batch
        iter_data = {k: v[i_batch:i_batch + N_rand] for k, v in train_datas.items()}
        camera_iter_data = {k: v[i_batch:i_batch + N_rand] for k, v in camera_train_datas.items()}
        batch_rays = iter_data.pop('rays').permute(0, 2, 1)

        i_batch += N_rand
        # if i_batch >= len(train_datas['rays']):
        #     print("Shuffle data after an epoch!")
        #     shuffle_idx = np.random.permutation(len(train_datas['rays']))
        #     train_datas = {k: v[shuffle_idx] for k, v in train_datas.items()}
        #     i_batch = 0
        if i_batch >= len(camera_train_datas['camera_rays']):
            print("Shuffle data after an epoch!")
            camera_shuffle_idx = np.random.permutation(len(camera_train_datas['camera_rays']))
            camera_train_datas = {k: v[camera_shuffle_idx] for k, v in camera_train_datas.items()}
            i_batch = 0

        #####  Core optimization loop  #####
        nerf.train()
        # if i == args.kernel_start_iter:
        #     torch.cuda.empty_cache()
        with torch.autograd.set_detect_anomaly(True):
            #previous
            rgb1, rgb2, rgb3 = nerf(H, W, K, chunk=args.chunk,
                                         rays=batch_rays, rays_info=iter_data, camera_rays_info=camera_iter_data,
                                         camera_AIF=camera_AIFROI,
                                         retraw=True, **render_kwargs_train)


            #katz 0928


            RI,intensity_pred=nerf(chunk=args.chunk, rays=batch_rays retraw=True, **render_kwargs_train)

            # Compute Losses
            # =====================

            target_rgb = camera_iter_data['camera_focus_rgbsf'].squeeze(-2)
            loss = img2mse(rgb1, target_rgb)
            psnr = mse2psnr(img_loss1)


            # print("loss",loss,"psnr",psnr,loss.requires_grad)
            # loss.requires_grad_(True)
            optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():
            loss.backward()
            optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        # dt = time.time() - time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'network_state_dict': nerf.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)


        if i % args.i_testset == 0 and i > 0:
            print('Start test')
            test(args, tensorboard, i, nerf, H, W, K, camera_test_datas, camera_AIFROI, render_kwargs_test,
                 i_test_camera, focus_imagessf, global_step, test_metric_file)

        if i % args.i_tensorboard == 0:
            tensorboard.add_scalar("Loss", loss.item(), global_step/args.MLP123)
            tensorboard.add_scalar("PSNR", psnr1.item(), global_step)
            # for k, v in extra_loss.items():
            #     tensorboard.add_scalar(k, v.item(), global_step)

        if i % args.i_print == 0:
            time1=time.time()
            time_s=time1-time0
            print(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR1: {psnr1.item()}  PSNR2: {psnr2.item()}  PSNR3: {psnr3.item()} Time: {time_s}")
            time0=time1


        global_step += 1


def test(args,tensorboard,i,nerf,H,W,K,camera_test_datas,camera_AIFROI,render_kwargs_test,i_test_camera,focus_imagessf,global_step,test_metric_file):

    print('use mlp_', args.MLP123)
    basedir = args.basedir
    expname = args.object_category
    testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
    os.makedirs(testsavedir, exist_ok=True)

    # ray_show_img(focus_imagessf[0], focus_imagessf.shape[1], focus_imagessf.shape[2], is_torch=True)
    if os.path.isfile(testsavedir + '/paras.npy') is not True:
        with torch.no_grad():
            nerf.eval()
            rgbs1, rgbs2, rgbs3, paras = nerf(H, W, K, args.chunk, camera_rays_info=camera_test_datas,
                                              camera_AIF=camera_AIFROI,
                                              render_kwargs=render_kwargs_test)
            rgbs1_save = rgbs1  # (rgbs - rgbs.min()) / (rgbs.max() - rgbs.min())
            rgbs2_save = rgbs2
            rgbs3_save = rgbs3

            np.save(os.path.join(testsavedir, 'paras.npy'), paras.cpu().numpy())
            # saving
            for rgb_idx, rgb in enumerate(rgbs1_save):
                rgb8 = to8b(rgb.cpu().numpy())
                dir = os.path.join(testsavedir, 'rgb1')
                if not os.path.exists(dir):
                    os.mkdir(dir)
                filename = os.path.join(dir, f'{rgb_idx:03d}.png')
                # rgb8 = cv2.cvtColor(rgb8, cv2.COLOR_BGR2RGB)
                imageio.imwrite(filename, rgb8)
            for rgb_idx, rgb in enumerate(rgbs2_save):
                rgb8 = to8b(rgb.cpu().numpy())
                dir = os.path.join(testsavedir, 'rgb2')
                if not os.path.exists(dir):
                    os.mkdir(dir)
                filename = os.path.join(dir, f'{rgb_idx:03d}.png')
                # rgb8 = cv2.cvtColor(rgb8, cv2.COLOR_BGR2RGB)
                imageio.imwrite(filename, rgb8)
            for rgb_idx, rgb in enumerate(rgbs3_save):
                rgb8 = to8b(rgb.cpu().numpy())
                dir = os.path.join(testsavedir, 'rgb3')
                if not os.path.exists(dir):
                    os.mkdir(dir)
                filename = os.path.join(dir, f'{rgb_idx:03d}.png')
                # rgb8 = cv2.cvtColor(rgb8, cv2.COLOR_BGR2RGB)
                imageio.imwrite(filename, rgb8)

            # evaluation
            rgbs = rgbs1
            rgbs1 = rgbs[i_test_camera]
            rgbs = rgbs2
            rgbs2 = rgbs[i_test_camera]
            rgbs = rgbs3
            rgbs3 = rgbs[i_test_camera]

            target_rgb_ldr = focus_imagessf[i_test_camera]

            test_mse = compute_img_metric(rgbs3, target_rgb_ldr, 'mse')
            test_psnr1 = compute_img_metric(rgbs1, target_rgb_ldr, 'psnr')
            test_psnr2 = compute_img_metric(rgbs2, target_rgb_ldr, 'psnr')
            test_psnr3 = compute_img_metric(rgbs3, target_rgb_ldr, 'psnr')
            test_ssim = compute_img_metric(rgbs3, target_rgb_ldr, 'ssim')
            test_lpips = compute_img_metric(rgbs3, target_rgb_ldr, 'lpips')
            if isinstance(test_lpips, torch.Tensor):
                test_lpips = test_lpips.item()

            tensorboard.add_scalar("Test MSE", test_mse, global_step)
            tensorboard.add_scalar("Test PSNR1", test_psnr1, global_step)
            tensorboard.add_scalar("Test PSNR2", test_psnr2, global_step)
            tensorboard.add_scalar("Test PSNR3", test_psnr3, global_step)
            tensorboard.add_scalar("Test SSIM", test_ssim, global_step)
            tensorboard.add_scalar("Test LPIPS", test_lpips, global_step)

        with open(test_metric_file, 'a') as outfile:
            outfile.write(
                f"iter{i}/globalstep{global_step}: MSE:{test_mse:.8f} PSNR1:{test_psnr1:.8f} PSNR2:{test_psnr2:.8f} PSNR3:{test_psnr3:.8f}"
                f" SSIM:{test_ssim:.8f} LPIPS:{test_lpips:.8f}\n")
        print(f"[TEST] Iter: {i} MSE:{test_mse:.8f} PSNR1:{test_psnr1:.8f} PSNR2:{test_psnr2:.8f} PSNR3:{test_psnr3:.8f} SSIM:{test_ssim:.8f} LPIPS:{test_lpips:.8f}" )

        print('Saved test set')
    else:
        print("参数文件存在")
def render_only(args, basedir, expname, start, nerf, H, W, K, camera_test_datadir, i_train_camera, test_metric_file,render_kwargs_test,fd=0,set=1):
    print('RENDER ONLY')
    focus_images, depths, AIFs, AIF_image, fd_t, ids= load_allin3d_data(args, camera_test_datadir, args.factor)
    #fd=np.array(range(800,1200,20))/1000
    print(fd)
    #focus_images, depths, AIF_image,fd = load_allin3d_data(args, args.camera_test_datadir, args.factor)
    i_test_camera=np.array([i for i in np.arange(int(focus_images.shape[0]))])
    camera_train_datas = {}
    camera_test_datas = {}
    # camera-nerf datas processing
    camera_train_datas, camera_test_datas, camera_AIFROI, focus_imagessf, depthssf, AIF_imagesf = camera_data_process(
        args, camera_train_datas, camera_test_datas, focus_images, depths, AIFs,AIF_image, fd, ids,i_train_camera)

    camera_test_datas = {k: torch.tensor(v).cuda() for k, v in camera_test_datas.items()}
    focus_imagessf = torch.tensor(focus_imagessf).cuda()
    camera_AIFROI = {k: torch.tensor(v).cuda() for k, v in camera_AIFROI.items()}
    #camera_AIFROI = torch.tensor(camera_AIFROI).cuda()

    testsavedir = os.path.join(basedir, expname,
                               f"renderonly"
                               f"_{'test' if args.render_test else 'path'}"
                               f"_{start:06d}")
    os.makedirs(testsavedir, exist_ok=True)

    with torch.no_grad():
        nerf.eval()
        rgbs1, rgbs2, rgbs3, paras = nerf(H, W, K, args.chunk, camera_rays_info=camera_test_datas,
                                          camera_AIF=camera_AIFROI,
                                          render_kwargs=render_kwargs_test)
        rgbs1_save = rgbs1  # (rgbs - rgbs.min()) / (rgbs.max() - rgbs.min())
        rgbs2_save = rgbs2
        rgbs3_save = rgbs3
        np.save(os.path.join(testsavedir, f'paras{set}.npy'), paras.cpu().numpy())
        # saving
        for rgb_idx, rgb in enumerate(rgbs1_save):
            rgb8 = to8b(rgb.cpu().numpy())
            dir = os.path.join(testsavedir, 'rgb1')
            if not os.path.exists(dir):
                os.mkdir(dir)
            filename = os.path.join(dir, f'{set*20+rgb_idx:03d}.png')
            # rgb8 = cv2.cvtColor(rgb8, cv2.COLOR_BGR2RGB)
            imageio.imwrite(filename, rgb8)
        for rgb_idx, rgb in enumerate(rgbs2_save):
            rgb8 = to8b(rgb.cpu().numpy())
            dir = os.path.join(testsavedir, 'rgb2')
            if not os.path.exists(dir):
                os.mkdir(dir)
            filename = os.path.join(dir, f'{set*20+rgb_idx:03d}.png')
            # rgb8 = cv2.cvtColor(rgb8, cv2.COLOR_BGR2RGB)
            imageio.imwrite(filename, rgb8)
        for rgb_idx, rgb in enumerate(rgbs3_save):
            rgb8 = to8b(rgb.cpu().numpy())
            dir = os.path.join(testsavedir, 'rgb3')
            if not os.path.exists(dir):
                os.mkdir(dir)
            filename = os.path.join(dir, f'{set*20+rgb_idx:03d}.png')
            # rgb8 = cv2.cvtColor(rgb8, cv2.COLOR_BGR2RGB)
            imageio.imwrite(filename, rgb8)

    #     target_rgb_ldr = focus_imagessf
    #     # evaluation
    #     test_mse = compute_img_metric(rgbs3, target_rgb_ldr, 'mse')
    #     test_psnr1 = compute_img_metric(rgbs1, target_rgb_ldr, 'psnr')
    #     test_psnr2 = compute_img_metric(rgbs2, target_rgb_ldr, 'psnr')
    #     test_psnr3 = compute_img_metric(rgbs3, target_rgb_ldr, 'psnr')
    #     test_ssim = compute_img_metric(rgbs3, target_rgb_ldr, 'ssim')
    #     test_lpips = compute_img_metric(rgbs3, target_rgb_ldr, 'lpips')
    #     if isinstance(test_lpips, torch.Tensor):
    #         test_lpips = test_lpips.item()
    #
    #     #tensorboard.add_scalar("Render Only/Test MSE", test_mse, global_step)
    #     #tensorboard.add_scalar("Render Only/Test PSNR", test_psnr, global_step)
    #     #tensorboard.add_scalar("Render Only/Test SSIM", test_ssim, global_step)
    #     #tensorboard.add_scalar("Render Only/Test LPIPS", test_lpips, global_step)
    #
    # with open(test_metric_file, 'a') as outfile:
    #     outfile.write(f"Render Only: MSE:{test_mse:.8f} PSNR1:{test_psnr1:.8f} PSNR2:{test_psnr2:.8f} PSNR3:{test_psnr3:.8f}"
    #                   f" SSIM:{test_ssim:.8f} LPIPS:{test_lpips:.8f}\n")
    print('RENDER END')
    return

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    train(args[i])
    torch.cuda.empty_cache()