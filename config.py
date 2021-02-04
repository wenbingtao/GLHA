# config.py ---
#
# Filename: config.py
# Description: Based on argparse usage from
#              https://github.com/carpedm20/DCGAN-tensorflow
# Author: Kwang Moo Yi
# Maintainer:
# Created: Mon Jun 26 11:06:51 2017 (+0200)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#

# Code:

import argparse

from servers import is_computecanada, is_cvlab_epfl, is_vcg_uvic


def str2bool(v):
    return v.lower() in ("true", "1")


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# -----------------------------------------------------------------------------
# Network
net_arg = add_argument_group("Network")
net_arg.add_argument(
    "--net_depth", type=int, default=12, help=""
    "number of layers")
net_arg.add_argument(
    "--net_nchannel", type=int, default=128, help=""
    "number of channels in a layer")
net_arg.add_argument(
    "--net_act_pos", type=str, default="post",
    choices=["pre", "mid", "post"], help=""
    "where the activation should be in case of resnet")
net_arg.add_argument(
    "--net_gcnorm", type=str2bool, default=True, help=""
    "whether to use context normalization for each layer")
net_arg.add_argument(
    "--net_batchnorm", type=str2bool, default=True, help=""
    "whether to use batch normalization")
net_arg.add_argument(
    "--net_bn_test_is_training", type=str2bool, default=False, help=""
    "is_training value for testing")

# -----------------------------------------------------------------------------
# Data
data_arg = add_argument_group("Data")
data_arg.add_argument(
    "--data_dump_prefix", type=str, default="./data_dump", help=""
    "prefix for the dump folder locations")
data_arg.add_argument(
    "--data_tr", type=str, default="st_peters.brown_bm_3_05", help=""
    "name of the dataset for train")
data_arg.add_argument(
    "--data_va", type=str, default="st_peters.brown_bm_3_05", help=""
    "name of the dataset for valid")
data_arg.add_argument(
    "--data_te", type=str, default="st_peters.brown_bm_3_05", help=""
    "name of the dataset for test")

# data_arg.add_argument(
#     "--data_te", type=str, default="reichstag", help=""
#     "name of the dataset for test")

# data_arg.add_argument(
#     "--data_zhao_tr", type=str, default="NARROW", help=""
#     "name of the dataset for train")
# data_arg.add_argument(
#     "--data_zhao_va", type=str, default="NARROW", help=""
#     "name of the dataset for valid")
# data_arg.add_argument(
#     "--data_zhao_te", type=str, default="NARROW", help=""
#     "name of the dataset for test")
#
# data_arg.add_argument(
#     "--data_zhao_tr", type=str, default="WIDE", help=""
#     "name of the dataset for train")
# data_arg.add_argument(
#     "--data_zhao_va", type=str, default="WIDE", help=""
#     "name of the dataset for valid")
# data_arg.add_argument(
#     "--data_zhao_te", type=str, default="WIDE", help=""
#     "name of the dataset for test")

data_arg.add_argument(
    "--data_zhao_tr", type=str, default="COLMAP", help=""
    "name of the dataset for train")
data_arg.add_argument(
    "--data_zhao_va", type=str, default="COLMAP", help=""
    "name of the dataset for valid")
data_arg.add_argument(
    "--data_zhao_te", type=str, default="COLMAP", help=""
    "name of the dataset for test")


# data_arg.add_argument(
#     "--data_zhao_te", type=str, default="gerrard", help=""
#     "name of the dataset for test")

# data_arg.add_argument(
#     "--data_zhao_te", type=str, default="person", help=""
#     "name of the dataset for test")

# data_arg.add_argument(
#     "--data_zhao_te", type=str, default="graham", help=""
#     "name of the dataset for test")

# data_arg.add_argument(
#     "--data_zhao_te", type=str, default="south", help=""
#     "name of the dataset for test")


data_arg.add_argument(
    "--data_crop_center", type=str2bool, default=False, help=""
    "whether to crop center of the image "
    "to match the expected input for methods that expect a square input")
data_arg.add_argument(
    "--use_lift", type=str2bool, default=False, help=""
    "if this is set to true, we expect lift to be dumped already for all "
    "images.")


# -----------------------------------------------------------------------------
# Objective
obj_arg = add_argument_group("obj")
obj_arg.add_argument(
    "--obj_num_kp", type=int, default=2000, help=""
    "number of keypoints per image")
obj_arg.add_argument(
    "--obj_top_k", type=int, default=-1, help=""
    "number of keypoints above the threshold to use for "
    "essential matrix estimation. put -1 to use all. ")
obj_arg.add_argument(
    "--obj_num_nn", type=int, default=1, help=""
    "number of nearest neighbors in terms of descriptor "
    "distance that are considered when generating the "
    "distance matrix")
obj_arg.add_argument(
    "--obj_num_K", type=int, default=60, help=""
    "number of nearest neighbors in computing D1,D2,"
    "which is the distance of the neighbor of x1,x2"
    )
obj_arg.add_argument(
    "--obj_geod_type", type=str, default="episym",
    choices=["sampson", "episqr", "episym"], help=""
    "type of geodesic distance")
obj_arg.add_argument(
    "--obj_geod_th", type=float, default=1e-4, help=""
    "theshold for the good geodesic distance")

# -----------------------------------------------------------------------------
# Loss
loss_arg = add_argument_group("loss")
loss_arg.add_argument(
    "--loss_decay", type=float, default=0.0, help=""
    "l2 decay")
loss_arg.add_argument(
    "--loss_classif", type=float, default=1.0, help=""
    "weight of the classification loss")
loss_arg.add_argument(
    "--loss_essential", type=float, default=0, help=""
    "weight of the essential loss")
#yangfan tianjiazhengzesunshixiangxishu 1/17/14:25
loss_arg.add_argument(
    "--loss_regularizer_b",type=float, default=0.5, help=""
    "b_weight of the regularization loss")
loss_arg.add_argument(
    "--loss_regularizer_a",type=float, default=0.5, help=""
    "a_weight of the regularization loss")
################
loss_arg.add_argument(
    "--loss_essential_init_iter", type=int, default=20000, help=""
    "initial iterations to run only the classification loss")

# -----------------------------------------------------------------------------
# Training
train_arg = add_argument_group("Train")
train_arg.add_argument(
    "--run_mode", type=str, default="train", help=""
    "run_mode")
train_arg.add_argument(
    "--train_batch_size", type=int, default=16, help=""
    "batch size")
train_arg.add_argument(
    "--train_max_tr_sample", type=int, default=10000, help=""
    "number of max training samples")
train_arg.add_argument(
    "--train_max_va_sample", type=int, default=1000, help=""
    "number of max validation samples")
train_arg.add_argument(
    "--train_max_te_sample", type=int, default=1000, help=""
    "number of max test samples")
train_arg.add_argument(
    "--train_lr", type=float, default=1e-3, help=""
    "learning rate")
train_arg.add_argument(
    "--train_iter", type=int, default=200000, help=""
    "training iterations to perform")
train_arg.add_argument(
    "--res_dir", type=str, default="./logs", help=""
    "base directory for results")
train_arg.add_argument(
    "--log_dir", type=str, default="", help=""
    "save directory name inside results")
train_arg.add_argument(
    "--test_log_dir", type=str, default="", help=""
    "which directory to test inside results")
train_arg.add_argument(
    "--val_intv", type=int, default=2000, help=""
    "validation interval")
train_arg.add_argument(
    "--report_intv", type=int, default=1000, help=""
    "summary interval")

# -----------------------------------------------------------------------------
# Visualization
vis_arg = add_argument_group('Visualization')
vis_arg.add_argument(
    "--vis_dump", type=str2bool, default=False, help=""
    "turn this on to dump data for visualization"
)
vis_arg.add_argument(
    "--tqdm_width", type=int, default=79, help=""
    "width of the tqdm bar"
)


def setup_dataset(dataset_name):
    """Expands dataset name and directories properly"""

    # Use only the first one for dump
    dataset_name = dataset_name.split(".")[0]

    # Setup the base directory depending on the environment
    if is_computecanada():
        data_dir = "/scratch/kyi/datasets/scenedata_splits/"
    elif is_vcg_uvic():
        data_dir = "/data/datasets/sfm/scenedata_splits/"
    elif is_cvlab_epfl():
        data_dir = "/cvlabdata2/home/kyi/Datasets/scenedata_splits/"
    else:
        data_dir = "./datasets/"

    # Expand the abbreviations that we use to actual folder names
    if "cogsci4" == dataset_name:
        # Load the data
        data_dir += "brown_cogsci_4---brown_cogsci_4---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.3
    elif "reichstag" == dataset_name:
        # Load the data
        data_dir += "reichstag/"
        geom_type = "Calibration"
        vis_th = 100
    #yangfan  tianjiashujuji"twoview_reichstag"  1/24/20:14
    elif "twoview_reichstag" == dataset_name:
         # Load the data
        data_dir += "twoview_reichstag/"
        geom_type = "Calibration"
        vis_th = 100
    elif "door" == dataset_name:
        # Load the data
        data_dir += "door/"
        geom_type = "Calibration"
        vis_th = 0.1
    elif "lmr30" == dataset_name:
        # Load the data
        data_dir += "lmr30/"
        geom_type = "Calibration"
        vis_th = 0.1  
    elif "test_l" == dataset_name:
        # Load the data
        data_dir += "test_l/"
        geom_type = "Calibration"
        vis_th = 0.1  
    elif "fro_test_h" == dataset_name:
        # Load the data
        data_dir += "fro_test_h/"
        geom_type = "Calibration"
        vis_th = 0.1     
    elif "sacre_coeur" == dataset_name:
        # Load the data
        data_dir += "sacre_coeur/"
        geom_type = "Calibration"
        vis_th = 100
    elif "buckingham" in dataset_name:
        # Load the data
        data_dir += "buckingham_palace/"
        geom_type = "Calibration"
        vis_th = 100
    elif "notre_dame" == dataset_name:
        # Load the data
        data_dir += "notre_dame_front_facade/"
        geom_type = "Calibration"
        vis_th = 100
    elif "st_peters" == dataset_name:
        # Load the data
        data_dir += "st_peters_square/"
        geom_type = "Calibration"
        vis_th = 100
    #yangfan tianjiashujuji"twoview_1" 1/16/14:06
    elif "twoview_1" == dataset_name:
        # Load the data
        data_dir += "twoview_1/"
        geom_type = "Calibration"
        vis_th = 100
    #yangfan tianjiashujuji"twoview_st1" 2/18/20:06
    elif "twoview_st1" == dataset_name:
        # Load the data
        data_dir += "twoview_st1/"
        geom_type = "Calibration"
        vis_th = 100
    elif "twoview_st2" == dataset_name:
        # Load the data
        data_dir += "twoview_st2/"
        geom_type = "Calibration"
        vis_th = 100
    elif "twoview_st_peters_square" == dataset_name:
        # Load the data
        data_dir += "twoview_st_peters_square/"
        geom_type = "Calibration"
        vis_th = 100
    elif "harvard_conf_big" == dataset_name:
        # Load the data
        data_dir += "harvard_conf_big---hv_conf_big_1---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.3
    elif "home_ac" == dataset_name:
        # Load the data
        data_dir += "home_ac---home_ac_scan1_2012_aug_22---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.3
    elif "fountain" in dataset_name:
        # Load the data
        data_dir += "fountain/"
        geom_type = "Calibration"
        vis_th = -1
    elif "herzjesu" == dataset_name:
        # Load the data
        data_dir += "herzjesu/"
        geom_type = "Calibration"
        vis_th = -1
    elif "gms-teddy" == dataset_name:
        # Load the data
        data_dir += "gms-teddy/"
        geom_type = "Calibration"
        vis_th = 100
    elif "gms-large-cabinet" in dataset_name:
        # Load the data
        data_dir += "gms-large-cabinet/"
        geom_type = "Calibration"
        vis_th = 100
    elif "cogsci8_05" == dataset_name:
        # Load the data
        data_dir += "brown_cogsci_8---brown_cogsci_8---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "cogsci2_05" == dataset_name:
        # Load the data
        data_dir += "brown_cogsci_2---brown_cogsci_2---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "hv_lounge1_2_05" == dataset_name:
        # Load the data
        data_dir += "harvard_corridor_lounge---hv_lounge1_2---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "hv_c10_2_05" == dataset_name:
        # Load the data
        data_dir += "harvard_c10---hv_c10_2---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "hv_s1_2_05" == dataset_name:
        # Load the data
        data_dir += "harvard_robotics_lab---hv_s1_2---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "hv_c4_1_05" == dataset_name:
        # Load the data
        data_dir += "harvard_c4---hv_c4_1---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "cs7_05" == dataset_name:
        # Load the data
        data_dir += "brown_cs_7---brown_cs7---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "cs3_05" == dataset_name:
        # Load the data
        data_dir += "brown_cs_3---brown_cs3---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "mit_46_6conf_05" == dataset_name:
        # Load the data
        data_dir += "mit_46_6conf---bcs_floor6_conf_1---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "mit_46_6lounge_05" == dataset_name:
        # Load the data
        data_dir += "mit_46_6lounge---bcs_floor6_long---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "mit_w85g_05" == dataset_name:
        # Load the data
        data_dir += "mit_w85g---g_0---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "mit_32_g725_05" == dataset_name:
        # Load the data
        data_dir += "mit_32_g725---g725_1---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "florence_hotel_05" == dataset_name:
        # Load the data
        data_dir += "hotel_florence_jx---florence_hotel_stair_room_all---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "mit_w85h_05" == dataset_name:
        # Load the data
        data_dir += "mit_w85h---h2_1---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "cogsci6_05" == dataset_name:
        # Load the data
        data_dir += "brown_cogsci_6---brown_cogsci_6---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    # New sets
    elif "home_ac_05_fix" == dataset_name:
        # Load the data
        data_dir += "home_ac---home_ac_scan1_2012_aug_22---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "harvard_conf_big_05_fix" == dataset_name:
        # Load the data
        data_dir += "harvard_conf_big---hv_conf_big_1---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "cogsci3_05" == dataset_name:
        # Load the data
        data_dir += "brown_cogsci_3---brown_cogsci_3---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "cogsci4_05_fix" == dataset_name:
        # Load the data
        data_dir += "brown_cogsci_4---brown_cogsci_4---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "home_aca_05_fix" == dataset_name:
        # Load the data
        data_dir += "home_ag---apartment_ag_nov_7_2012_scan1_erika---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "hotel_ucsd_05" == dataset_name:
        # Load the data
        data_dir += "hotel_ucsd---la2-maxpairs-10000-random---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "brown_cs_4_05" == dataset_name:
        data_dir += "brown_cs_4---brown_cs4-maxpairs-10000-random---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "hotel_ucla_ant_05" == dataset_name:
        # Load the data
        data_dir += "hotel_ucla_ant---hotel_room_ucla_scan1_2012_oct_05-maxpairs-10000-random---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "hv_lounge3_05" == dataset_name:
        data_dir += "harvard_corridor_lounge---hv_lounge_corridor3_whole_floor-maxpairs-10000-random---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "harvard_conf_big_05_rand" == dataset_name:
        # Load the data
        data_dir += "harvard_conf_big---hv_conf_big_1-maxpairs-10000-random---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "brown_bm_3_05" == dataset_name:
        # Load the data
        data_dir += "brown_bm_3---brown_bm_3-maxpairs-10000-random---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    #yangfan tianjia yongyuceshi shujuji 2019/2/21/10:39
    elif "twoview_brown" == dataset_name:
        # Load the data
        data_dir += "twoview_brown/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "two_brown2" == dataset_name:
        # Load the data
        data_dir += "two_brown2/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "tbrown661_681" == dataset_name:
        # Load the data
        data_dir += "tbrown661_681/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "tbrown_box" == dataset_name:
        # Load the data
        data_dir += "tbrown_box/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "home_pt_05" == dataset_name:
        # Load the data
        data_dir += "home_pt---home_pt_scan1_2012_oct_19-maxpairs-10000-random---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "hv_comp_05" == dataset_name:
        # Load the data
        data_dir += "harvard_computer_lab---hv_c1_1-maxpairs-10000-random---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "hv_lounge2_05" == dataset_name:
        # Load the data
        data_dir += "harvard_corridor_lounge---hv_lounge_corridor2_1-maxpairs-10000-random---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "hotel_ped_05" == dataset_name:
        # Load the data
        data_dir += "hotel_pedraza---hotel_room_pedraza_2012_nov_25-maxpairs-10000-random---skip-10-dilate-25/"
        geom_type = "Calibration"
        vis_th = 0.5


    elif "brown" == dataset_name:
        # Load the data
        data_dir += "brown/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "door" == dataset_name:
        # Load the data
        data_dir += "door/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "test_a" == dataset_name:
        # Load the data
        data_dir += "test_a/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "test_b" == dataset_name:
        # Load the data
        data_dir += "test_b/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "test_c" == dataset_name:
        # Load the data
        data_dir += "test_c/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "test_d" == dataset_name:
        # Load the data
        data_dir += "test_d/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "test_e" == dataset_name:
        # Load the data
        data_dir += "test_e/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "test_f" == dataset_name:
        # Load the data
        data_dir += "test_f/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "test_g" == dataset_name:
        # Load the data
        data_dir += "test_g/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "test_h" == dataset_name:
        # Load the data
        data_dir += "test_h/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "test_i" == dataset_name:
        # Load the data
        data_dir += "test_i/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "test_j" == dataset_name:
        # Load the data
        data_dir += "test_j/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "test_k" == dataset_name:
        # Load the data
        data_dir += "test_k/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "test_l" == dataset_name:
        # Load the data
        data_dir += "test_l/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "view1" == dataset_name:
        # Load the data
        data_dir += "view1/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "fro_test_a" == dataset_name:
        # Load the data
        data_dir += "fro_test_a/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "fro_test_b" == dataset_name:
        # Load the data
        data_dir += "fro_test_b/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "fro_test_c" == dataset_name:
        # Load the data
        data_dir += "fro_test_c/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "fro_test_d" == dataset_name:
        # Load the data
        data_dir += "fro_test_d/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "fro_test_e" == dataset_name:
        # Load the data
        data_dir += "fro_test_e/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "fro_test_f" == dataset_name:
        # Load the data
        data_dir += "fro_test_f/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "fro_test_g" == dataset_name:
        # Load the data
        data_dir += "fro_test_g/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "fro_test_h" == dataset_name:
        # Load the data
        data_dir += "fro_test_h/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "fro_test_i" == dataset_name:
        # Load the data
        data_dir += "fro_test_i/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "fro_test_j" == dataset_name:
        # Load the data
        data_dir += "fro_test_j/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "fro_test_k" == dataset_name:
        # Load the data
        data_dir += "fro_test_k/"
        geom_type = "Calibration"
        vis_th = 0.5
    elif "fro_test_l" == dataset_name:
        # Load the data
        data_dir += "fro_test_l/"
        geom_type = "Calibration"
        vis_th = 0.5

    return data_dir, geom_type, vis_th


def get_config():
    config, unparsed = parser.parse_known_args()

    # Setup the dataset related things
    for _mode in ["tr", "va", "te"]:
        data_dir, geom_type, vis_th = setup_dataset(
            getattr(config, "data_" + _mode))
        setattr(config, "data_dir_" + _mode, data_dir)
        setattr(config, "data_geom_type_" + _mode, geom_type)
        setattr(config, "data_vis_th_" + _mode, vis_th)

    return config, unparsed


def print_usage():
    parser.print_usage()

#
# config.py ends here
