


# Load data

from __future__ import print_function

from config import get_config, print_usage
from data import load_data, load_data_zhao, generateData1, generateData2, load_graphcut_test, load_ransac_test, load_RT
from network import MyNetwork

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

eps = 1e-10
use3d = False
config = None

config, unparsed = get_config()

print("-------------------------Deep Essential-------------------------")
print("Note: To combine datasets, use .")


def main(config):
    """The main function."""

    # Initialize network
    mynet = MyNetwork(config)

    # # Run propper mode
    if config.run_mode == "train":

        # Load data train and validation data
        data = {}
        data["train"] = load_data(config, "train")
        data["valid"] = load_data(config, "valid")

        # Run train
        mynet.train(data)

    # if config.run_mode == "train":
    #
    #     # Load data train and validation data
    #     data = {}
    #     data["train"] = load_data_zhao(config, "train")
    #     data["valid"] = load_data_zhao(config, "valid")
    #
    #     # Run train
    #     mynet.train(data)

    elif config.run_mode == "test":

        # Load validation and test data. Note that we also load validation to
        # visualize more than what we did for training. For training we choose
        # minimal reporting to not slow down.
        data = {}
        data["valid"] = load_data(config, "valid")
        data["test"] = load_data(config, "test")

        # Run train
    #     mynet.test(data)

    # elif config.run_mode == "test":
    #
    #     # Load validation and test data. Note that we also load validation to
    #     # visualize more than what we did for training. For training we choose
    #     # minimal reporting to not slow down.
    #     data = {}
    #     # data["valid"] = load_data_zhao(config, "valid")
    #     data["test"] = load_data_zhao(config, "test")

        # Run train
        mynet.test(data)


    elif config.run_mode == "load_RT":

        # Load validation and test data. Note that we also load validation to
        # visualize more than what we did for training. For training we choose
        # minimal reporting to not slow down.
        data = {}
        data["test"] = load_RT(config, "test")

        # Run train
        # mynet.test(data)

    elif config.run_mode == "graphcut_ransac":

        # Load validation and test data. Note that we also load validation to
        # visualize more than what we did for training. For training we choose
        # minimal reporting to not slow down.
        data = {}
        data["test"] = generateData1(config, "test")

    elif config.run_mode == "graphcut_ransac_zhao":

        # Load validation and test data. Note that we also load validation to
        # visualize more than what we did for training. For training we choose
        # minimal reporting to not slow down.
        data = {}
        data["test"] = generateData2(config, "test")

    elif config.run_mode == "graphcut_ransac_test_zhao":

        # Load validation and test data. Note that we also load validation to
        # visualize more than what we did for training. For training we choose
        # minimal reporting to not slow down.
        data = {}
        data = load_data_zhao(config, "test")
        load_graphcut_test(config, "test", data)

        # Run train
        # mynet.test(data)

    elif config.run_mode == "graphcut_ransac_test":

        # Load validation and test data. Note that we also load validation to
        # visualize more than what we did for training. For training we choose
        # minimal reporting to not slow down.
        data = {}
        data = load_data(config, "test")
        load_graphcut_test(config, "test", data)

        # Run train
        # mynet.test(data)

    elif config.run_mode == "ransac_test":

        # Load validation and test data. Note that we also load validation to
        # visualize more than what we did for training. For training we choose
        # minimal reporting to not slow down.
        data = {}
        data = load_data(config, "test")
        load_ransac_test(config, "test", data)

        # Run train
        # mynet.test(data)



    elif config.run_mode == "comp":

        # This mode is for running comparison experiments. While cleaning the
        # code, I took many parts out to make the code cleaner, which may have
        # caused some issues. We are releasing this part just to help
        # researchers, but we will not provide any support for this
        # part. Please use at your own risk.
        data = {}
        data["valid"] = load_data(config, "valid")
        data["test"] = load_data(config, "test")

        # Run train
        mynet.comp(data)


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)

#
# main.py ends here
