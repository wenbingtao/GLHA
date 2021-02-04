


from __future__ import print_function

import itertools
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np

import cv2
from config import get_config
from data import loadFromDir
from geom import get_episqr, get_episym, get_sampsons, parse_geom
from six.moves import xrange
from utils import loadh5, saveh5

eps = 1e-10
use3d = False
config = None

config, unparsed = get_config()


def dump_data_pair(args):
    dump_dir, idx, ii, jj, queue = args

    # queue for monitoring
    if queue is not None:
        queue.put(idx)

    dump_file = os.path.join(
        dump_dir, "idx_sort-{}-{}.h5".format(ii, jj))

    if not os.path.exists(dump_file):
        # Load descriptors for ii
        desc_ii = loadh5(
            os.path.join(dump_dir, "kp-z-desc-{}.h5".format(ii)))["desc"]
        desc_jj = loadh5(
            os.path.join(dump_dir, "kp-z-desc-{}.h5".format(jj)))["desc"]
        #yangfan  dayin desc_ii  1/22/20:32
        #print(len(desc_ii))
        #print(desc_ii.shape)
        # compute decriptor distance matrix
        distmat = np.sqrt(
            np.sum(
                (np.expand_dims(desc_ii, 1) - np.expand_dims(desc_jj, 0))**2,
                axis=2))
        # yangfan dayin distmat  1/22/21:03
        # print(len(distmat))
        # print(dismat.shape)
        # print(dismat.shape[0])
        # print((np.expand_dims(desc_ii, 1)).shape)
        # print((np.expand_dims(desc_jj, 0)).shape)
        # print((np.expand_dims(desc_ii, 1) - np.expand_dims(desc_jj, 0)).shape)
        # print(((np.expand_dims(desc_ii, 1) - np.expand_dims(desc_jj, 0))**2).shape)
        # print((np.sum((np.expand_dims(desc_ii, 1) - np.expand_dims(desc_jj, 0))**2,axis=2)).shape)
        # Choose K best from N
        idx_sort = np.argsort(distmat, axis=1)[:, :config.obj_num_nn]

        # yangfan  dayin idx_sort  shuliang 1/22/21:42
        # print(len(idx_sort))
        # print(np.argsort(distmat, axis=1).shape)
        # print(idx_sort)
        # print(idx_sort.shape)
        idx_sort = (
            np.repeat(
                np.arange(distmat.shape[0])[..., None],
                idx_sort.shape[1], axis=1
            ),
            idx_sort
        )
        distmat = distmat[idx_sort]
        #yangfan ceshi dayin idx_sort  1/22/16:57
        #print(len(idx_sort[1]))
        #print(idx_sort[1])
        #print(idx_sort.shape)
        # Dump to disk
        dump_dict = {}
        dump_dict["idx_sort"] = idx_sort
        saveh5(dump_dict, dump_file)

# gain the scale information of SIFT
def unpackSIFTOctave(kpt):
    """unpackSIFTOctave(kpt)->(octave,layer,scale)
    @created by Silencer at 2018.01.23 11:12:30 CST
    @brief Unpack Sift Keypoint by Silencer
    @param kpt: cv2.KeyPoint (of SIFT)
    """
    _octave = kpt.octave
    octave = _octave&0xFF
    layer  = (_octave>>8)&0xFF
    if octave>=128:
        octave |= -128
    if octave>=0:
        scale = float(1/(1<<octave))
    else:
        scale = float(1<<-octave)
    return scale

#yangfan  tianjia  cv2_kp  1/22/17:15
def make_xy(num_sample, pairs, kp, z, desc,  sizze, angle, img, geom, vis, depth, geom_type,
            cur_folder):


    # Create a random folder in scratch
    dump_dir = os.path.join(cur_folder, "dump")
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    # randomly suffle the pairs and select num_sample amount
    np.random.seed(1234)
    cur_pairs = [
        pairs[_i] for _i in np.random.permutation(len(pairs))[:num_sample]
    ]
    #yangfan dayin cur_pairs 2/20/8:47
    # print(cur_pairs) ###  shengchengceshishujujishishiyong 
    cur_pairs = cur_pairs[:1] ###
    # print("ground_truth is:") ###
    print(cur_pairs)  #zhiqudiyidui shuju ###
    idx = 0
    for ii, jj in cur_pairs:
        idx += 1
        print(
            "\rExtracting keypoints {} / {}".format(idx, len(cur_pairs)),
            end="")
        sys.stdout.flush()

        # Check and extract keypoints if necessary
        for i in [ii, jj]:
            dump_file = os.path.join(dump_dir, "kp-z-desc-{}.h5".format(i))
            if not os.path.exists(dump_file):
                if kp[i] is None:
                    cv_kp, cv_desc = sift.detectAndCompute(img[i].transpose(
                        1, 2, 0), None)
                    #print(len(cv_kp))
                    #yangfan dayin cv_kp  1/23/22:05
                    #print(cv_kp)
                    cx = (img[i][0].shape[1] - 1.0) * 0.5
                    cy = (img[i][0].shape[0] - 1.0) * 0.5
                    # Correct coordinates using K
                    cx += parse_geom(geom, geom_type)["K"][i, 0, 2]
                    cy += parse_geom(geom, geom_type)["K"][i, 1, 2]
                    xy = np.array([_kp.pt for _kp in cv_kp])

                    #add _kp.size,_kp.angle 
                    Angle = np.array([_kp.angle for _kp in cv_kp])
                    Size = np.array([unpackSIFTOctave(_kp) for _kp in cv_kp])
                    angle[i] = Angle
                    sizze[i] = Size
                    #added over
                    #yangfan dayin xy  1/23/19:54
                    # print(xy)
                    #print(xy.shape)
                    # Correct focals
                    fx = parse_geom(geom, geom_type)["K"][i, 0, 0]
                    fy = parse_geom(geom, geom_type)["K"][i, 1, 1]
                    
                    kp[i] = (
                        xy - np.array([[cx, cy]])
                    ) / np.asarray([[fx, fy]])
                    desc[i] = cv_desc
                    #yangfan  tianjia guanjiandian yuanzhi  1/22/15:11
                    #cv2_kp[i] = cv_kp
                    # print(kp[i])
                if z[i] is None:
                    cx = (img[i][0].shape[1] - 1.0) * 0.5
                    cy = (img[i][0].shape[0] - 1.0) * 0.5
                    fx = parse_geom(geom, geom_type)["K"][i, 0, 0]
                    fy = parse_geom(geom, geom_type)["K"][i, 1, 1]
                    xy = kp[i] * np.asarray([[fx, fy]]) + np.array([[cx, cy]])
                    if len(depth) > 0:
                        z[i] = depth[i][
                            0,
                            np.round(xy[:, 1]).astype(int),
                            np.round(xy[:, 0]).astype(int)][..., None]
                    else:
                        z[i] = np.ones((xy.shape[0], 1))
                # Write descs to harddisk to parallize
                dump_dict = {}
                dump_dict["kp"] = kp[i]
                #added by yangfan 
                dump_dict["sizze"] = sizze[i]
                dump_dict["angle"] = angle[i]

                dump_dict["z"] = z[i]
                dump_dict["desc"] = desc[i]
                #yangfan tianjia guanjiandian yuanzhi  1/22/15:15
                #dump_dict["cv2_kp"] = cv2_kp[i]
                saveh5(dump_dict, dump_file)
            else:
                dump_dict = loadh5(dump_file)
                kp[i] = dump_dict["kp"]
                #added by yangfan
                dump_dict["sizze"] = sizze[i]
                dump_dict["angle"] = angle[i]

                z[i] = dump_dict["z"]
                desc[i] = dump_dict["desc"]
                #yangfan tianjia guanjiandian yuanzhi  1/22/15:15
                #cv2_kp[i] = dump_dict["cv2_kp"]
    print("")

    # Create arguments
    pool_arg = []
    idx = 0
    for ii, jj in cur_pairs:
        idx += 1
        pool_arg += [(dump_dir, idx, ii, jj)]
    # Run mp job
    ratio_CPU = 0.2  #0.8
    number_of_process = int(ratio_CPU * mp.cpu_count())
    pool = mp.Pool(processes=number_of_process)
    manager = mp.Manager()
    queue = manager.Queue()
    for idx_arg in xrange(len(pool_arg)):
        pool_arg[idx_arg] = pool_arg[idx_arg] + (queue,)
    # map async
    pool_res = pool.map_async(dump_data_pair, pool_arg)
    # monitor loop
    while True:
        if pool_res.ready():
            break
        else:
            size = queue.qsize()
            print("\rDistMat {} / {}".format(size, len(pool_arg)), end="")
            sys.stdout.flush()
            time.sleep(1)
    pool.close()
    pool.join()
    print("")
    # Pack data
    idx = 0
    total_num = 0
    good_num = 0
    bad_num = 0
    for ii, jj in cur_pairs:
        idx += 1
        print("\rWorking on {} / {}".format(idx, len(cur_pairs)), end="")
        sys.stdout.flush()

        # ------------------------------
        # Get dR
        R_i = parse_geom(geom, geom_type)["R"][ii]
        R_j = parse_geom(geom, geom_type)["R"][jj]
        dR = np.dot(R_j, R_i.T)
        # Get dt
        t_i = parse_geom(geom, geom_type)["t"][ii].reshape([3, 1])
        t_j = parse_geom(geom, geom_type)["t"][jj].reshape([3, 1])
        dt = t_j - np.dot(dR, t_i)
        # ------------------------------
        # Get sift points for the first image
        x1 = kp[ii]
        # yangfan dayin x1
        # print(x1.shape)
        y1 = np.concatenate([kp[ii] * z[ii], z[ii]], axis=1)
        # yangfan dayin y1 xingzhuang 1/23/14:54
        # print(y1.shape)
        # Project the first points into the second image
        y1p = np.matmul(dR[None], y1[..., None]) + dt[None]
        # yangfan dayin y1p xingzhuang 1/23/14:56
        # print(y1p.shape)
        # move back to the canonical plane
        x1p = y1p[:, :2, 0] / y1p[:, 2, 0][..., None]
        # yangfan dayin x1p[2]  1/23/15:00
        # print(x1p.shape)
        # print((y1p[:,:2,0]).shape)
        # ------------------------------
        # Get sift points for the second image
        x2 = kp[jj]
        
        size1 =sizze[ii]
        size2 =sizze[jj]
        angle1 = angle[ii]
        angle2 = angle[jj]
        print(size1.shape)

        # ------------------------------
        # create x1, y1, x2, y2 as a matrix combo
        x1mat = np.repeat(x1[:, 0][..., None], len(x2), axis=-1)
        y1mat = np.repeat(x1[:, 1][..., None], len(x2), axis=1)
        # yangfan  dayin y1mat  1/23/17:15
        # print(y1mat)

        #add sizeX,angleX
        SIZEx1mat = np.repeat(size1[..., None], len(size2), axis=-1)
        ANGLEx1mat = np.repeat(angle1[..., None], len(angle2), axis=-1)


        x1pmat = np.repeat(x1p[:, 0][..., None], len(x2), axis=-1)
        y1pmat = np.repeat(x1p[:, 1][..., None], len(x2), axis=1)

        x2mat = np.repeat(x2[:, 0][None], len(x1), axis=0)
        y2mat = np.repeat(x2[:, 1][None], len(x1), axis=0)
        # yangfan  dayin  x2[:, 0][None]  1/23/21:46
        # print((x2[:, 0][None]))
        # print((x2[:, 0][None]).shape)
        # print(x1pmat)
        # print(x2mat.shape)
        SIZEx2mat = np.repeat(size2[None], len(size1), axis=0)
        ANGLEx2mat = np.repeat(angle2[None], len(angle1), axis=0)

        # Load precomputed nearest neighbors
        idx_sort = loadh5(os.path.join(
            dump_dir, "idx_sort-{}-{}.h5".format(ii, jj)))["idx_sort"]
        # Move back to tuples
        idx_sort = (idx_sort[0], idx_sort[1])
        # yangfan dayin idx_sort xingzhuang  1/23/20:16
        # print(idx_sort)
        x1mat = x1mat[idx_sort]
        y1mat = y1mat[idx_sort]
        
        #added by yangfan
        SIZEx1mat = SIZEx1mat[idx_sort]
        ANGLEx1mat = ANGLEx1mat[idx_sort]
        SIZEx2mat = SIZEx2mat[idx_sort]
        ANGLEx2mat = ANGLEx2mat[idx_sort]

        x1pmat = x1pmat[idx_sort]
        y1pmat = y1pmat[idx_sort]

        x2mat = x2mat[idx_sort]
        y2mat = y2mat[idx_sort]
        # Turn into x1, x1p, x2
        x1 = np.concatenate(
            [x1mat.reshape(-1, 1), y1mat.reshape(-1, 1)], axis=1)
        
        x1p = np.concatenate(
            [x1pmat.reshape(-1, 1),
             y1pmat.reshape(-1, 1)], axis=1)
        x2 = np.concatenate(
            [x2mat.reshape(-1, 1), y2mat.reshape(-1, 1)], axis=1)

        sizeX = SIZEx1mat.reshape(-1,1)
        sizeY = SIZEx2mat.reshape(-1,1)
        angleX = ANGLEx1mat.reshape(-1,1)
        angleY = ANGLEx2mat.reshape(-1,1)


        # transform x1,x2 to  cv_kp.pt
        cx_ii = (img[ii][0].shape[1] - 1.0) * 0.5
        cy_ii = (img[ii][0].shape[0] - 1.0) * 0.5
        cx_ii += parse_geom(geom, geom_type)["K"][ii, 0, 2]
        cy_ii += parse_geom(geom, geom_type)["K"][ii, 1, 2]

        fx_ii = parse_geom(geom, geom_type)["K"][ii, 0, 0]
        fy_ii = parse_geom(geom, geom_type)["K"][ii, 1, 1]
        X = x1*np.asarray([[fx_ii, fy_ii]]) + np.array([[cx_ii, cy_ii]])
       

        cx_jj = (img[jj][0].shape[1] - 1.0) * 0.5
        cy_jj = (img[jj][0].shape[0] - 1.0) * 0.5
        cx_jj += parse_geom(geom, geom_type)["K"][jj, 0, 2]
        cy_jj += parse_geom(geom, geom_type)["K"][jj, 1, 2]

        fx_jj = parse_geom(geom, geom_type)["K"][jj, 0, 0]
        fy_jj = parse_geom(geom, geom_type)["K"][jj, 1, 1]
        Y = x2*np.asarray([[fx_jj, fy_jj]]) + np.array([[cx_jj, cy_jj]])
        
        # print(X)
        # print(Y)
        # ------------------------------
        # Get the geodesic distance using with x1, x2, dR, dt
        geod_d = get_episym(x1, x2, dR, dt)
        # Get *rough* reprojection errors. Note that the depth may be noisy. We
        # ended up not using this...

        CorrectIndex = []
        for idx_ in xrange(len(geod_d)):
        	if geod_d[idx_] < config.obj_geod_th:
        		CorrectIndex.append((idx_+1))

        CorrectIndex = np.array(CorrectIndex)
        np.savetxt("X.txt",X)
        np.savetxt("Y.txt",Y)
        np.savetxt("CorrectIndex.txt",CorrectIndex)
        # np.savetxt("size2",size2)
        np.savetxt("sizeX.txt",sizeX)
        np.savetxt("sizeY.txt",sizeY)
        np.savetxt("angleX.txt",angleX)
        np.savetxt("angleY.txt",angleY)
    # now. Simply return it
    print(".... done")

    res_dict = {}

    res_dict["X"] = X
    res_dict["Y"] = Y
    res_dict["CorrectIndex"] = CorrectIndex
    res_dict["sizeX"] = sizeX
    res_dict["sizeY"] = sizeY   
    res_dict["angleX"] = angleX
    res_dict["angleY"] = angleY   
    
    # X_Y_CorrectIndex_file = os.path.join(dump_dir, "X_Y_CorrectIndex.h5")
    # saveh5(res_dict, X_Y_CorrectIndex_file)
    X_Y_CorrectIndex_file = os.path.join(dump_dir, "X_Y_CorrectIndex_ang_size.h5")
    saveh5(res_dict, X_Y_CorrectIndex_file)

    return res_dict


print("-------------------------h5_to _mat-------------------------")
print("Note: h5t0_mat.py will only work on the first dataset")

# Read conditions
crop_center = config.data_crop_center
data_folder = config.data_dump_prefix
if config.use_lift:
    data_folder += "_lift"

# Prepare opencv
print("Creating Opencv SIFT instance")
if not config.use_lift:
    sift = cv2.xfeatures2d.SIFT_create(
        nfeatures=config.obj_num_kp, contrastThreshold=1e-5)

# Now start data prep
print("Preparing data for {}".format(config.data_tr.split(".")[0]))

for _set in ["train"]:
    num_sample = getattr(
        config, "train_max_{}_sample".format(_set[:2]))

    # Load the data
    print("Loading Raw Data !")
    split = _set
    #yangfan  fanhuizhi  tianjia cv2_kp  1/22/17:12
    img, geom, vis, depth, kp, desc = loadFromDir(
        getattr(config, "data_dir_" + _set[:2]) + split + "/",
        "-16x16",
        bUseColorImage=True,
        crop_center=crop_center,
        load_lift=config.use_lift)
    if len(kp) == 0:
        kp = [None] * len(img)
    if len(desc) == 0:
        desc = [None] * len(img)
    #yangfan  tianjia cv2_kp = [None] * len(img)  2 line 1/22/17:24  
    #if len(cv2_kp) == 0:
        #cv2_kp = [None] * len(img)
    z = [None] * len(img)

    sizze = [None] * len(img)
    angle = [None] * len(img)

    # Generating all possible pairs
    print("Generating list of all possible pairs for {}".format(_set))
    pairs = []
    for ii, jj in itertools.product(xrange(len(img)), xrange(len(img))):
        if ii != jj:
            if vis[ii][jj] > getattr(config, "data_vis_th_" + _set[:2]):
                pairs.append((ii, jj))
    print("{} pairs generated".format(len(pairs)))
    #yangfan dayin pairs  1/24/1:03
    #print(pairs)  [(0, 1), (1, 0)]

    # Create data dump directory name
    data_names = getattr(config, "data_" + _set[:2])
    data_name = data_names.split(".")[0]
    cur_data_folder = "/".join([
        data_folder,
        data_name,
        "numkp-{}".format(config.obj_num_kp),
        "nn-{}".format(config.obj_num_nn),
    ])
    if not config.data_crop_center:
        cur_data_folder = os.path.join(cur_data_folder, "nocrop")
    if not os.path.exists(cur_data_folder):
        os.makedirs(cur_data_folder)
    suffix = "{}-{}".format(
        _set[:2], getattr(config, "train_max_" + _set[:2] + "_sample"))
    cur_folder = os.path.join(cur_data_folder, suffix)
    if not os.path.exists(cur_folder):
        os.makedirs(cur_folder)

    # Check if we've done this folder already.
    print(" -- Waiting for the data_folder to be ready")
    ready_file = os.path.join(cur_folder, "ready")
    if not os.path.exists(ready_file):
        print(" -- No ready file {}".format(ready_file))
        print(" -- Generating data")

        # Make xy for this pair
        # data_dict = make_xy(
        #     num_sample, pairs, kp, z, desc,#cv2_kp,  #yangfan tianjia cv2_kp  1/22/17:15
        #     img, geom, vis, depth, getattr(
        #         config, "data_geom_type_" + _set[:2]),
        #     cur_folder)
        data_dict = make_xy(
            num_sample, pairs, kp, z, desc, sizze, angle,  #cv2_kp,  #yangfan tianjia cv2_kp  1/22/17:15
            img, geom, vis, depth, getattr(
                config, "data_geom_type_" + _set[:2]),
            cur_folder)

        # Let's pickle and save data. Note that I'm saving them
        # individually. This was to have flexibility, but not so much
        # necessary.
        # for var_name in data_dict:
        #     cur_var_name = var_name + "_" + _set[:2]
        #     out_file_name = os.path.join(cur_folder, cur_var_name) + ".pkl"
        #     with open(out_file_name, "wb") as ofp:
        #         pickle.dump(data_dict[var_name], ofp)

        # Mark ready
        with open(ready_file, "w") as ofp:
            ofp.write("This folder is ready\n")
    else:
        print("Done!")

#
# dump_data.py ends here
