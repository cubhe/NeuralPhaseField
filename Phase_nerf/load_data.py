import glob
import cv2
import numpy as np
import os, imageio
import shutil


def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')

def load_allin3d_data_old(args, basedir, factor=8):
    # Create folder to save pictures after scaling
    sfx = ''
    if factor is not None:
        sfx = '_{}'.format(factor)
        # _minify(basedir, factors=[factor])
        factor = factor
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'focus' + sfx)
    depthdir = os.path.join(basedir, 'depth' + sfx)
    AIFdir = os.path.join(basedir, 'AIF' + sfx)

    imgdir_list = os.listdir(imgdir)
    del imgdir_list[imgdir_list.index("fd.npy")]
    imgdir_list.sort(key=lambda x: int(x[4:-4]))
    imgfiles = [os.path.join(imgdir, f) for f in imgdir_list if
                f.endswith('JPG') or f.endswith('jpg') or f.endswith('png') or f.endswith('bmp')]

    depthfiles = [os.path.join(depthdir, f) for f in sorted(os.listdir(depthdir))]
    AIFfiles = [os.path.join(AIFdir, f) for f in sorted(os.listdir(AIFdir))]
    fdfile = os.path.join(basedir, 'focus' + sfx, 'fd.npy')

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    imgs = np.stack(imgs, -1)
    is_mask = True
    depths = np.load(depthfiles[1]) / 1000. if is_mask else np.load(depthfiles[0]) / 1000.
    depths = depths[:, :, np.newaxis]
    AIF = [imread(f)[..., :3] / 255. for f in AIFfiles][0]
    fd = np.load(fdfile)/1000.
    print('Loaded image data', imgs.shape, depths.shape, AIF.shape,len(fd))

    # Correct rotation matrix ordering and move variable dim to axis 0
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs

    # i_test = np.argmin(dists)
    # print('HOLDOUT view is', i_test)

    images = images.astype(np.float32)
    depths = depths.astype(np.float32)
    AIF = AIF.astype(np.float32)
    return images, fd,depths, AIF

def load_allin3d_data(args, basedir, factor=8):
    # Create folder to save pictures after scaling
    sfx = ''
    if factor is not None:
        sfx = '_{}'.format(factor)
        # _minify(basedir, factors=[factor])
        factor = factor
    else:
        factor = 1

    focusdir = os.path.join(basedir, 'focus' + sfx)
    depthdir = os.path.join(basedir, 'depth' + sfx)
    AIFdir = os.path.join(basedir, 'AIF' + sfx)

    focus_dir_list = os.listdir(focusdir)
    focus_subdir = [os.path.join(focusdir, dir) for dir in focus_dir_list]

    depthfiles_list = []
    AIFfiles_list = []
    imgfiles_list = []
    fd_list = []
    view_id_list=[]
    focus_subdir.sort(key=lambda x: int(x[-3:]))
    for subdir in focus_subdir:
        view_num = subdir[-3:]
        view_id=int(view_num)
        subdir_list = os.listdir(subdir)
        # subdir_list.sort(key=lambda x: int(x[4:-4]))
        imgfiles = [os.path.join(subdir, f) for f in subdir_list if
                    f.endswith('JPG') or f.endswith('jpg') or f.endswith('png') or f.endswith('bmp')]
        imgfiles.sort(key=lambda x: int(x[-7:-4]))
        imgfiles_len = len(imgfiles)
        imgfiles_list.append(imgfiles)

        fd_path = os.path.join(subdir, 'fd.npy')
        fd = np.load(fd_path)
        # fd=np.linspace(800,1200,150)/1000.
        fd_len=len(fd)
        fd_list.append(fd)

        depth_path = os.path.join(depthdir, 'depth'+view_num + '.npy')
        depth_files=np.tile(depth_path,(fd_len))
        depthfiles_list.append(depth_files)

        AIF_path = os.path.join(AIFdir, 'image' + view_num + '.png')
        AIF_files=np.tile(AIF_path,(fd_len))
        AIFfiles_list.append(AIF_files)

        view_id = np.tile(view_id, (fd_len))
        view_id_list.append(view_id)

    depthfiles_list =np.reshape(np.stack(depthfiles_list,axis=0),(-1))
    AIFfiles_list = np.reshape(np.stack(AIFfiles_list,axis=0),(-1))
    imgfiles_list = np.reshape(np.stack(imgfiles_list,axis=0),(-1))
    fds = np.reshape(np.stack(fd_list,axis=0),(-1))
    ids = np.reshape(np.stack(view_id_list, axis=0), (-1))

    AIFfiles = [os.path.join(AIFdir, f) for f in sorted(os.listdir(AIFdir))]
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
    AIF_image = [imread(f)[..., :3] / 255. for f in AIFfiles]
    AIF_image = np.stack(AIF_image, -1)
    AIF_image = np.moveaxis(AIF_image, -1, 0).astype(np.float32)

    imgs = [imread(f)[..., :3] / 255. for f in imgfiles_list]
    imgs = np.stack(imgs, -1)

    depths=[np.load(f) for f in depthfiles_list]
    depths = np.stack(depths, -1)

    AIFs = [imread(f)[..., :3] / 255. for f in AIFfiles_list]
    AIFs = np.stack(AIFs, -1)

    print('Loaded image data', imgs.shape, depths.shape, AIFs.shape, len(fds))

    # Correct rotation matrix ordering and move variable dim to axis 0
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs

    depths = np.moveaxis(depths, -1, 0).astype(np.float32)

    AIFs = np.moveaxis(AIFs, -1, 0).astype(np.float32)

    images = images.astype(np.float32)
    depths = depths.astype(np.float32)
    AIFs = AIFs.astype(np.float32)
    return images, depths, AIFs, AIF_image,fds,ids


def img_depth_resize(dir, factor):
    focus_path = os.path.join(dir, 'focus')
    depth_path = os.path.join(dir, 'depth')
    AIF_path = os.path.join(dir, 'AIF')
    focus_resize_path = os.path.join(dir, 'focus_{}'.format(factor))
    depth_resize_path = os.path.join(dir, 'depth_{}'.format(factor))
    AIF_resize_path = os.path.join(dir, 'AIF_{}'.format(factor))
    if not os.path.exists(focus_resize_path):
        os.mkdir(focus_resize_path)
    if not os.path.exists(focus_resize_path+r'/000'):
        os.mkdir(focus_resize_path+r'/000')
        focus_resize_path=focus_resize_path+r'/000'
    if not os.path.exists(depth_resize_path):
        os.mkdir(depth_resize_path)
    if not os.path.exists(AIF_resize_path):
        os.mkdir(AIF_resize_path)

    focus_paths = glob.glob(focus_path + "/*.bmp")
    for focus in focus_paths:
        idx = int((focus.split('/')[-1]).split('.')[0].split('_')[1])
        img = cv2.imread(focus)
        img_resize = cv2.resize(img, None, fx=1 / factor, fy=1 / factor)
        cv2.imwrite(focus_resize_path + '/foc_{:0>3d}.png'.format(idx), img_resize)

    fd = np.load(focus_path + "/fd.npy")
    depth = np.load(depth_path + "/depth.npy")
    mask = np.load(depth_path + "/mask.npy")
    depth_resize = cv2.resize(depth, None, fx=1 / factor, fy=1 / factor)
    depth_mask = depth * mask / 255
    depth_mask_resize = cv2.resize(depth_mask, None, fx=1 / factor, fy=1 / factor)
    np.save(depth_resize_path + "/depth000.npy", depth_resize)
    np.save(depth_resize_path + "/depth_mask.npy", depth_mask_resize)
    np.save(focus_resize_path + "/fd.npy", fd)

    AIF_path = glob.glob(AIF_path + "/*.tif")
    AIF = cv2.imread(AIF_path[0])
    AIF_resize = cv2.resize(AIF, None, fx=1 / factor, fy=1 / factor)
    cv2.imwrite(AIF_resize_path + '/image000.png', AIF_resize)


def data2dataset(old_dir, new_dir, num, is_more_focus=False, focus_list=[]):
    # old path
    AIF_path_file = old_dir + "/all_in_focus/" + str(num) + ".tif"
    depth_path_dir = old_dir + "/depth/" + str(num)
    focus_path_dir = old_dir + "/focus_small/" + str(num)
    fd_path_dir = old_dir + "/fd/" + str(num)
    # new path
    new_dir = new_dir + "/" + str(num)
    new_AIF_path = new_dir + "/AIF"
    new_AIF_path_file = new_AIF_path + "/AIF.tif"
    new_depth_path = new_dir + "/depth"
    new_focus_path = new_dir + "/focus"
    new_fd_path = new_dir + "/fd.npy"
    # make new dir

    path_list = [new_dir, new_AIF_path, new_focus_path]
    for path in path_list:
        if not os.path.exists(path):
            os.mkdir(path)
    # copy AIF and depth
    shutil.copy(AIF_path_file, new_AIF_path_file)
    if not os.path.exists(new_depth_path):
        shutil.copytree(depth_path_dir, new_depth_path)
    shutil.copy(fd_path_dir + "/fd.npy", new_fd_path)

    if is_more_focus:
        focus_path_dir = "/media/irvlab/新加卷1/ALLIN3D/" + str(num)
        focus_file = os.listdir(focus_path_dir)
        del focus_file[focus_file.index("fd.npy")]
        del focus_file[focus_file.index('sff_depth.npy')]
        del focus_file[focus_file.index('sff_depth.bmp')]
        del focus_file[focus_file.index('v.npy')]
        del focus_file[focus_file.index('v2.npy')]
        del focus_file[focus_file.index('v.xls')]
        fds = np.load(new_fd_path)

        if focus_list != []:

            focus_file.sort(key=lambda x: int(x[4:-4]))
            new_fds = fds[focus_list]
            np.save(new_focus_path + "/fd.npy", new_fds)
            focus_file = [focus_file[x] for x in focus_list]
        else:
            np.save(new_focus_path + "/fd.npy", fds)
            # print(fds)

        for focus in focus_file:
            if not 'simu' in focus:
                shutil.copy(focus_path_dir + "/" + focus, new_focus_path + "/" + focus)
    else:
        focus_file = os.listdir(focus_path_dir)
        for focus in focus_file:
            if not 'simu' in focus:
                shutil.copy(focus_path_dir + "/" + focus, new_focus_path + "/" + focus)
        new_fd = np.load(new_fd_path)  # 1-200
        new_fd = new_fd[np.arange(1, 200, 10) - 1]
        np.save(new_focus_path + "/fd.npy", new_fd)

def get_large_data(path,target_path):
    img_list=os.listdir(path)

    v_txt_path=img_list[img_list.index("4.txt")]
    v_txt=open(os.path.join(path,v_txt_path))
    vs=v_txt.readlines()
    fl=50
    fds=[]
    for v in vs:
        v=v.strip('\n')
        fd=1/((1/fl)-(1/(float(v)+23.6)))
        fds.append(fd)
        # print(v+" ")
    fds=np.stack(fds,axis=0)
    np.save(os.path.join(target_path,"fd.npy"),fds)
    del img_list[img_list.index("4.txt")]

    img_list=sorted(img_list,key=lambda x:int(x[7:-4]))
    focus_list = np.linspace(0, 399, 20).astype('int')
    new_img_list=[]
    new_fd_list=[]
    for i in focus_list:
        new_img_list.append(img_list[i])
        new_fd_list.append(fds[i])
    for img in new_img_list:
        idx=int(img[7:-4])
        img_path=os.path.join(path,img)
        image=cv2.imread(img_path)
        image_r=cv2.resize(image,None,fx=1/4,fy=1/4)
        cv2.imwrite(os.path.join(target_path,"focus_4/focus{:0>3d}.png".format(idx)),image_r)
        # cv2.imshow("1",image_r)
        # cv2.waitKey(0)

    new_fd_list=np.stack(new_fd_list,axis=0)
    np.save(os.path.join(target_path,"focus_4/fd.npy"),new_fd_list)


    depth=np.zeros((image_r.shape[0],image_r.shape[1]))
    AIF = np.ones(image_r.shape).astype('uint8')
    np.save(os.path.join(target_path,"depth_4/depth000.npy"),depth)
    cv2.imwrite(os.path.join(target_path,"AIF_4/image000.png"),AIF)
    print(img_list)
    # shutil.copy(focus_path_dir + "/" + focus, new_focus_path + "/" + focus)

if __name__ == '__main__':
    # num = 307
    # if num<300:
    #     data2dataset("/media/irvlab/新加卷1/data/ALLIN3D",
    #                  "/home/irvlab/heka/Project3_NeRF/code/camera_nerf/dataset/allin3d2", num,
    #                  is_more_focus=False, focus_list=list(range(0, 200, 10)))
    # else:
    #     data2dataset("/media/irvlab/新加卷1/data/ALLIN3D",
    #                  "/home/irvlab/heka/Project3_NeRF/code/camera_nerf/dataset/allin3d2", num,
    #                  is_more_focus=False, focus_list=list(range(0,200,10)))
    # img_depth_resize("/home/irvlab/heka/Project3_NeRF/code/camera_nerf/dataset/allin3d2/{}".format(num), 4)

    get_large_data("/media/irvlab/新加卷1/4","/home/irvlab/heka/Project3_NeRF/code/camera_nerf/dataset/allin3d/large")