import collections
import configparser
import itertools
import logging as lg
import math
import os
import string
import subprocess
import sys
import time
from ast import literal_eval
from copy import deepcopy
from time import gmtime, strftime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.errors
import skimage.filters
import skimage.feature
from scipy.ndimage import binary_dilation


class Config:
    """
    Configuration Class for the app. Class handles all reads and writes from and to the config file.
    """

    def __init__(self):

        self.config = configparser.ConfigParser()
        self.configpath = str()
        self.dataset_path = str()
        self.results_path = str()
        self.output_path = str()
        self.imgl_name = str()
        self.imgr_name = str()
        self.gt_name = str()

        self.prepro_enable = False
        self.prepro_filter = str()
        self.prepro_params = str()

        self.execution_mode = str("generic")
        self.execution_target = str()
        self.execution_error_handling = str("interrupt")

        self.evaluation_gt_coefficient = float(1)
        self.evaluation_textureless_width = int(5)
        self.evaluation_textureless_threshold = float(60)
        self.evaluation_discont_width = int(14)
        self.evaluation_discont_threshold = float(15)

        self.template = str()
        self.iterators = dict()
        self.constants = dict()

    @staticmethod
    def pass_value(**argument):
        """
        Passes through method for reading config values. it checks, if the value given in the argument is empty.
        If its empty an exception is raised

        :param argument: argument in form pass_value(param_name=param_name) is required
        :return:
        """
        descriptor = list(argument.keys())[0]
        value = argument[descriptor]
        error_msg = "Value for {} not set, but required. " \
                    "Please confurire in config file or use a setter method".format(descriptor)
        if type(value) is str and value == "":
            raise ConfigurationError(error_msg)
        if type(value) is dict and value == {}:
            raise ConfigurationError(error_msg)
        return value

    def _read_value(self, init_val, section, key, mandatory=True, dtype=None):
        """
        Method to safely read config keys from configuration file

        :param init_val: initial value. If its the default for the dtype, the config is read
        :param section: Configration Section
        :param key: Configuration key
        :param mandatory: if section and key is mandatory in config file
        :param dtype: datatype of config value
        :return: value of config[SECTION][KEY]
        """
        lg.debug("init_val: {}, section: {}, key: {}, mandatory={}, dtype={}".format(init_val, section, key,
                                                                                     mandatory, dtype))
        if section not in self.config:
            if init_val not in [int(), str(), float()]:
                return init_val
            elif mandatory:
                raise ConfigurationError("Required Configuration Section {} is missing".format(section))
        if key not in self.config[section]:
            if init_val not in [int(), str(), float()]:
                return init_val
            elif mandatory:
                raise ConfigurationError("Required Configuration Key {}: {} is missing".format(section, key))
        try:
            if dtype is int:
                return self.config.getint(section, key)
            if dtype is float:
                return self.config.getfloat(section, key)
            if dtype is bool:
                return self.config.getboolean(section, key)
            else:
                dtype = str
                return self.config[section][key]
        except ValueError:
            raise ConfigurationError("Configuration for config[{}][{}] not properly set. "
                                     "Type '{}' required.".format(section, key, dtype))

    def wipe(self):
        """
        Method for wiping all values. Should be done before reading of a config file.

        :return: None
        """
        self.dataset_path = str()
        self.results_path = str()
        self.output_path = str()
        self.imgl_name = str()
        self.imgr_name = str()
        self.gt_name = str()
        self.prepro_enable = False
        self.prepro_filter = str()
        self.prepro_params = str()
        self.execution_mode = str()
        self.execution_target = str()
        self.execution_error_handling = str("interrupt")
        self.evaluation_gt_coefficient = float(1)
        self.evaluation_textureless_width = int(5)
        self.evaluation_textureless_threshold = float(60)
        self.evaluation_discont_width = int(14)
        self.evaluation_discont_threshold = float(15)
        self.template = str()
        self.iterators = dict()
        self.constants = dict()

    def read(self, configpath):
        """
        Function for reading all config values into class variables. Its using the Config.read_value() method for safely
        reading values and handling error.

        :param configpath: filepath of the config file
        :return: None
        """
        # before reading config, existing values are deleted.
        self.wipe()

        self.configpath = configpath
        self.config.read(configpath)
        self.dataset_path = self._read_value(self.dataset_path, "BASE", "dataset_dir")
        self.results_path = self._read_value(self.results_path, "BASE", "results_dir")
        self.output_path = self._read_value(self.output_path, "BASE", "output_disp_filepath")
        self.imgl_name = self._read_value(self.imgl_name, "BASE", "imgL_name")
        self.imgr_name = self._read_value(self.imgr_name, "BASE", "imgR_name")
        self.gt_name = self._read_value(self.gt_name, "BASE", "gt_name")
        self.prepro_enable = self._read_value(self.prepro_enable, "PREPROCESSING", "enable", mandatory=False,
                                              dtype=bool)
        self.prepro_filter = self._read_value(self.prepro_filter, "PREPROCESSING", "filter", mandatory=False)
        self.prepro_params = self._read_value(self.prepro_params, "PREPROCESSING", "params", mandatory=False)
        self.execution_target = self._read_value(self.execution_target, "EXECUTION", "target")
        self.execution_mode = self._read_value(self.execution_mode, "EXECUTION", "mode", mandatory=False)
        self.execution_error_handling = self._read_value(self.execution_error_handling, "EXECUTION", "error_handling",
                                                         mandatory=False)
        self.evaluation_gt_coefficient = self._read_value(self.evaluation_gt_coefficient, "EVALUATION",
                                                          "gt_coefficient", mandatory=False,
                                                          dtype=float)
        self.evaluation_textureless_width = self._read_value(self.evaluation_textureless_width, "EVALUATION",
                                                             "textureless_width", mandatory=False,
                                                             dtype=int)
        self.evaluation_textureless_threshold = self._read_value(self.evaluation_textureless_threshold, "EVALUATION",
                                                                 "textureless_threshold", mandatory=False,
                                                                 dtype=float)
        self.evaluation_discont_width = self._read_value(self.evaluation_discont_width, "EVALUATION",
                                                         "discont_width", mandatory=False,
                                                         dtype=int)
        self.evaluation_discont_threshold = self._read_value(self.evaluation_discont_threshold, "EVALUATION",
                                                             "discont_threshold", mandatory=False,
                                                             dtype=float)

        self.template = self._read_value(self.template, "BASE", "template", mandatory=False)

        if "ITERATORS" in self.config:
            for iterator in self.config["ITERATORS"]:
                self.iterators[iterator] = self._read_value(str(), "ITERATORS", iterator,
                                                            mandatory=False).replace(" ", "").split(",")

        if "CONSTANTS" in self.config:
            for constant in self.config["CONSTANTS"]:
                self.constants[constant] = self._read_value(str(), "CONSTANTS", constant,
                                                            mandatory=False)

    def _update_value(self, section, key, value):
        """
        Function for updating a value in the config object.

        :param section: section
        :param key: value name
        :param value: value
        :return: None
        """
        if value != "":
            self.config[section][key] = str(value)
            lg.debug("Config Key updated Config[{}][{}] = {}".format(section, key, value))

    def update(self):
        """
        function updates the existing configuration file and stores every key.

        :return: None
        """
        if self.configpath is "":
            raise ConfigurationError("Config must be read first before updating. "
                                     "Use AlorithmEvaluator.read_config(config_filepath)")
        # Storing basic values
        self.config["BASE"]["template"] = self.template
        self._update_value("BASE", "dataset_dir", self.dataset_path)
        self._update_value("BASE", "results_dir", self.results_path)
        self._update_value("BASE", "output_disp_filepath", self.output_path)
        self._update_value("BASE", "imgL_name", self.imgl_name)
        self._update_value("BASE", "imgR_name", self.imgr_name)
        self._update_value("BASE", "gt_name", self.gt_name)
        self._update_value("PREPROCESSING", "enable", self.prepro_enable)
        self._update_value("PREPROCESSING", "filter", self.prepro_filter)
        self._update_value("PREPROCESSING", "params", self.prepro_params)
        self._update_value("EXECUTION", "target", self.execution_target)
        self._update_value("EXECUTION", "mode", self.execution_mode)
        self._update_value("EXECUTION", "error_handling", self.execution_error_handling)
        self._update_value("EVALUATION", "gt_coefficient", self.evaluation_gt_coefficient)
        self._update_value("EVALUATION", "textureless_width", self.evaluation_textureless_width)
        self._update_value("EVALUATION", "textureless_threshold", self.evaluation_textureless_threshold)
        self._update_value("EVALUATION", "discont_width", self.evaluation_discont_width)
        self._update_value("EVALUATION", "discont_threshold", self.evaluation_discont_threshold)
        # storing of all keys of section iterators and converting values from list
        self.config["ITERATORS"] = {}
        for iterator in self.iterators:
            self.config["ITERATORS"][iterator] = ",".join(str(x) for x in self.iterators[iterator])
        # storing of all keys of section iterators
        self.config["CONSTANTS"] = {}
        for constant in self.constants:
            self.config["CONSTANTS"][constant] = str(self.constants[constant])
        # writing config to .ini file
        with open(self.configpath, 'w') as configfile:
            self.config.write(configfile)

        print("Config updated")

    def check_template(self):
        """
        Checks if all constants and iterators are in the template and so if the template is valid.
        Note: This method cannot check, if there are keywords in the template, that dont represent a set value.
        This is done during execution

        :return: True if template ok. If not an Exception is raised.
        """
        for it in self.iterators.keys():
            if not self.template.__contains__("${}$".format(it)):
                raise ConfigurationError(
                    "Template and Iterators dont match. '${}$' is missing. Check:\n{}".format(it, self.template))

        if not self.template.__contains__("$out$"):
            raise ConfigurationError("Tag $out$ not found in Template. Check:\n{}".format(self.template))

        return True


class AlgorithmEvaluator:

    def __init__(self, **kwargs):
        """
        Init function ot the Algorithm Evaluator class. A few setting can be given right away.

        :param kwargs: kwargs
        :param config_path: path to config file
        :param mode: passing argument to AlgorithmEvaluator.set_mode(mode) method.
        :param target: passing argument to AlgorithmEvaluator.set_target(target) method.
        :param error_handling: passing argument to AlgorithmEvaluator.set_error_handling(error_handling) method.
        """
        lg.basicConfig(format='%(levelname)s::%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                       level=lg.INFO)

        self.base_config = Config()
        if "config_path" in kwargs:
            self.read_config(kwargs["config_path"])
        if "mode" in kwargs:
            self.set_mode(kwargs["mode"])
        if "target" in kwargs:
            self.set_target(kwargs["target"])
        if "error_handling" in kwargs:
            self.set_error_handling(kwargs["error_handling"])

        lg.basicConfig(format='%(levelname)s::%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S',
                       level=lg.DEBUG)

        self.datasets = dict()

    @staticmethod
    def set_logging_level(level):
        """
        Function to set intern logging level based on python logging library defined levels
        :param level: level
        :return: None
        """
        lg.getLogger().setLevel(lg.getLevelName(level.upper()))

    def read_config(self, config_filepath=None):
        """
        wrapper function for reading the config file.

        :param config_filepath: filepath of config file ".ini"
        :return: None
        """
        if config_filepath is None:
            raise ConfigurationError("no filepath given.")
        _, file_extension = os.path.splitext(config_filepath)
        if file_extension != ".ini":
            raise ConfigurationError("Config file not a .ini file.")
        if not os.path.isfile(config_filepath):
            raise ConfigurationError("Configfile not found.")
        self.base_config.read(config_filepath)

    def update_config(self):
        """
        wrapper function for updating the config file

        :return: None
        """
        self.base_config.update()

    def set_mode(self, mode=None):
        """
        setter function for mode. must be one of ["exe", "python"] at the moment.

        :param mode: mode
        :return: None
        """
        # checks if input is valid
        if mode is None:
            raise ConfigurationError("No mode given.")
        if mode == "exe":
            self.base_config.execution_mode = "exe"
        elif mode == "python":
            self.base_config.execution_mode = "python"
        else:
            raise ConfigurationError("Unknown mode.")

    def set_target(self, target=None):
        """
        setter function for target executable

        :param target: target executable path
        :return: None
        """
        # checks if input is valid
        if target is None:
            raise ConfigurationError("No target given.")
        if not os.path.isfile(target):
            raise ConfigurationError("Tht target given has no file at path: {}".format(target))
        else:
            self.base_config.execution_target = target

    def set_error_handling(self, error_handling):
        """
        setter function for error_handling

        :param error_handling: "Interrupt" or "Ignore"
        :return: None
        """
        # checks if input is valid
        if error_handling is None:
            raise ConfigurationError("No error handling instruction given.")
        if error_handling.lower() not in ["interrupt", "ignore"]:
            raise ConfigurationError("Unknown error handling instruction '{}'.".format(error_handling))
        self.base_config.execution_error_handling = error_handling.lower()

    def get_iterators(self):
        """
        getter function for constants

        :return: iterators (dict)
        """
        return self.base_config.iterators

    def set_iterator(self, name=None, steps=None, remove=False):
        """
        Method for adding or editing iterators with checking of validity

        :param name: Iterator descriptor
        :param steps: Steps to iterate over
        :param remove: set true to delete iterator.
        :return: None
        """
        # checks if input is valid
        if name is None or type(name) is not str:
            raise TypeError("Name must be type string.")
        # if remove is set, its checked, wether name exists in stored iterators. If so the entry is removed.
        if remove:
            try:
                del self.base_config.iterators[name]
                return
            except KeyError:
                raise ConfigurationError("Iterator {} not found".format(name))
        # checks if steps is of type sequence
        if not isinstance(steps, collections.Sequence):
            raise TypeError("Steps must be type sequence.")
        # checks every entry of sequence for type
        for step in list(steps):
            if type(step) not in [float, int, str]:
                raise TypeError("Steps items must be either type string, int or float.")

        self.base_config.iterators[name] = list(steps)

    def get_constants(self):
        """
        getter function for constants

        :return: constants (dict)
        """
        return self.base_config.constants

    def set_constant(self, name=None, value=None, remove=False):
        """
        Method for adding or editing iterators with checking of validity

        :param name: Constant descriptor
        :param value: value
        :param remove: set true to delete constant.
        :return:
        """
        # checks if input is valid
        if name is None or type(name) is not str:
            raise TypeError("Name must be type string.")
        # if remove is set, its checked, wether name exists in stored constants. If so the entry is removed.
        if remove:
            try:
                del self.base_config.constants[name]
                return
            except KeyError:
                raise ConfigurationError("Constant {} not found".format(name))
        # checks value for type
        if type(value) not in [float, int, str]:
            raise TypeError("Value must be either type string, int or float.")

        self.base_config.constants[name] = value

    def set_evaluation_parameter(self, parameter, value):
        """
        setter function for evaluation parameters gt_coefficient, textureless_width, textureless_threshold,
        discont_width, discont_threshold

        :param parameter: parameter descriptor
        :param value: value
        :return: None
        """
        # checks if input is valid
        if parameter is None or type(parameter) is not str:
            raise TypeError("Parameter must be type string.")
        if value is None:
            raise ValueError("No value given.")
        # checks for each subgroup of possible parameters, if input is valid. If so, the parameter is set.
        if parameter in ["gt_coefficient", "textureless_threshold", "discont_threshold"]:
            if type(value) not in [int, float]:
                raise ConfigurationError("Value must be type float or int.")
            if parameter == "gt_coefficient":
                self.base_config.evaluation_gt_coefficient = float(value)
            if parameter == "textureless_threshold":
                self.base_config.evaluation_textureless_threshold = float(value)
            if parameter == "discont_threshold":
                self.base_config.evaluation_discont_threshold = float(value)
        elif parameter in ["textureless_width", "discont_width"]:
            if type(value) is not int:
                raise TypeError("Value must be type int.")
            if parameter == "textureless_width":
                self.base_config.evaluation_textureless_width = int(value)
            if parameter == "discont_width":
                self.base_config.evaluation_discont_width = int(value)
        else:
            raise ConfigurationError("Unknown Parameter {}.".format(parameter))

    def set_execution_template(self, template):
        """
        setter function for template

        :param template: template
        :return: None
        """
        # checks if input is valid
        if template is None:
            raise ConfigurationError("No template given.")
        self.base_config.template = template

    def check_execution_template(self):
        """
        Wrapper function for checking the validity of the execution template.
        Note: This method cannot check, if there are keywords in the template, that dont represent a set value.
        This is done during execution

        :return: If template is valid (bool)
        """
        try:
            self.base_config.check_template()
        except ConfigurationError as e:
            lg.error("Unsufficient Template.")
            lg.debug(str(e))
            return False
        return True

    def set_preprocessing(self, filter_name, params=None):
        """
        function to set preprocessing filter for evaluation
        :param filter_name: name of filter from list
        :param params: (dict) of params and corresponding values
        :return: None
        """
        # check input types
        if type(filter_name) is not str:
            raise ConfigurationError("Filter name must be str")
        if type(params) is not dict:
            raise ConfigurationError("Filter name must be dict")
        self.base_config.prepro_enable = True
        self.base_config.prepro_filter = filter_name
        self.base_config.prepro_params = str(params)

    def remove_preprocessing(self):
        """
        function removes all preprocessing settings
        :return: None
        """
        self.base_config.prepro_enable = False
        self.base_config.prepro_filter = str()
        self.base_config.prepro_params = str()

    @staticmethod
    def daw():
        """
        When the window is stuck, use this. Closes all opencv windows.

        :return: None
        """
        cv2.destroyAllWindows()

    def show_region(self, mode, image_of_dataset=None):
        """
        wrapper function for showing the region based of various evaluation parameters.

        :param mode: "textureless", "discontinued"
        :param image_of_dataset: on which image from the dataset dir the region should be displayed.
        If None, the first image is used.
        :return: None
        """
        # first all windows are closed.
        cv2.destroyAllWindows()
        mode = mode.lower()
        # dataset must be found to use image path
        self._localize_dataset(verbose=False)
        # if datasets are found get paths of selected images. When none is selected, the first one is used.
        if image_of_dataset is not None:
            lg.info(self.datasets)
            if image_of_dataset not in list(self.datasets.keys()):
                raise KeyError("No subdirectory with that name found in folder datasets")
            else:
                img_path = self.datasets[image_of_dataset]["left"]
                gt_path = self.datasets[image_of_dataset]["gt"]
        else:
            img_path = self.datasets[list(self.datasets.keys())[0]]["left"]
            gt_path = self.datasets[list(self.datasets.keys())[0]]["gt"]
        # depending selected mode, the Evaluator function for displaying the region is called.
        if mode == "textureless":
            Evaluator(config=self.base_config, img_base_path=img_path,
                      eval_mode=False)._get_textureless_map(display=True)
        elif mode == "discontinued":
            Evaluator(config=self.base_config, img_gt_path=gt_path, img_base_path=img_path,
                      eval_mode=False)._get_discontinued_map(display=True)
        else:
            raise ConfigurationError("Mode must be type str and must be 'textureless' or 'discontinued'")

    def get_dataset(self):
        """
        getter function for all dataset keys
        :return: dataset keys
        """
        return list(self.datasets.keys())

    def _apply_filter(self, dataset):
        """
        function to filter images for preprocessing
        :param dataset: dataset name
        :return: None
        """
        found = False
        filter_func = None
        # Try to find function in two libraries
        try:
            filter_func = getattr(skimage.filters, self.base_config.prepro_filter)
            found = True
        except AttributeError:
            pass
        try:
            filter_func = getattr(skimage.feature, self.base_config.prepro_filter)
            found = True
        except AttributeError:
            pass
        if not found:
            raise ConfigurationError("Selected filter function {} not available in skimage.feature or skimage.filter"
                                     .format(self.base_config.prepro_filter))

        # convert arguments string to dict
        if self.base_config.prepro_params is not "":
            try:
                args = literal_eval(self.base_config.prepro_params)
            except SyntaxError:
                raise ConfigurationError(
                    "Configuration for Preprocessing not correct. Parameters do not convert to dict.\n"
                    "Check: {}".format(self.base_config.prepro_params))
        else:
            args = dict()
        # path modifications
        path_l = self.datasets[dataset]["left"]
        path_l_filtered = os.path.splitext(path_l)[0] + "_filtered" + os.path.splitext(path_l)[1]
        path_r = self.datasets[dataset]["right"]
        path_r_filtered = os.path.splitext(path_r)[0] + "_filtered" + os.path.splitext(path_r)[1]

        lg.debug("Preprocessing dataset with function {}, args={}".format(filter_func, args))

        # load images and filter them
        imgl = cv2.imread(path_l, 0)
        imgl_f = np.uint8(filter_func(imgl, **args) * 255)
        cv2.imwrite(path_l_filtered, imgl_f)

        imgr = cv2.imread(path_l, 0)
        imgr_f = np.uint8(filter_func(imgr, **args) * 255)
        cv2.imwrite(path_r_filtered, imgr_f)

        # save paths
        self.datasets[dataset]["left_f"] = path_l_filtered
        self.datasets[dataset]["right_f"] = path_r_filtered

    def _localize_dataset(self, verbose=True):
        """
        Internal function for searching through all folders of the dataset dirctory and cataloging images in a dict.
        Images must be correctly named, or the corresponding dataset wont be used.

        :param verbose: If True a console output for each found dataset is given.
        :return: None
        """
        # get initial dir
        start_dir = os.getcwd()

        # getting names of images
        img_left_name = self.base_config.pass_value(imgl_name=self.base_config.imgl_name)
        img_right_name = self.base_config.pass_value(imgr_name=self.base_config.imgr_name)
        img_gt_name = self.base_config.pass_value(gt_name=self.base_config.gt_name)
        # Changeing directory to dataset path
        base_dir = self.base_config.pass_value(dataset_path=self.base_config.dataset_path)
        os.chdir(base_dir)
        # Listing all subdirectories
        all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
        if verbose:
            lg.info("Localizing datasets.")
        if self.base_config.prepro_enable:
            lg.info("Preprocessing datasets ENABLED")
        # iterating over subdirectories
        for subdir in all_subdirs:
            # change path to subdir
            path = os.path.join(base_dir, subdir)
            os.chdir(path)
            if os.path.isfile(img_left_name) and os.path.isfile(img_right_name) and os.path.isfile(img_gt_name):
                left_pic_path = os.path.abspath(img_left_name)
                right_pic_path = os.path.abspath(img_right_name)
                gt_path = os.path.abspath(img_gt_name)

                # check dimensions
                dim_list = list()
                for pic in [img_left_name, img_right_name, img_gt_name]:
                    pic_data = cv2.imread(pic, 0)
                    dim_list.append(pic_data.shape)
                dim_check = False
                if dim_list[0][0] == dim_list[1][0] == dim_list[2][0]:
                    if dim_list[0][1] == dim_list[1][1] == dim_list[2][1]:
                        dim_check = True

                if dim_check and verbose:
                    lg.debug("Dimensions: {}x{}".format(dim_list[0][0], dim_list[0][1]))
                elif verbose:
                    lg.warning("Images for dataset '{}' not available. Dimensions dont fit".format(subdir))

                # if images are existing in directory, store their path in the dictionary
                self.datasets[subdir] = dict(left=left_pic_path, right=right_pic_path, gt=gt_path)
                # apply filters
                if self.base_config.prepro_enable:
                    self._apply_filter(subdir)

                if verbose:
                    lg.debug(path)
            else:
                if verbose:
                    lg.warning("Images for dataset '{}' not available.".format(subdir))
        # change working directory back to start
        os.chdir(start_dir)

    def _build_shell_command(self, permutation, it_names, dataset):
        """
        Internal function to build the shell command based on current permutations and the current dataset.
        In the process all $xx$ marked keywords are tried to be removed. If some are left after the process,
        an exception is raised.

        :param permutation: Current permutations of iterators (sequence)
        :param it_names: names of those iterators (sequence)
        :param dataset: dataset name
        :return: parsed template
        """
        # go to base directory
        os.chdir(self.base_config.pass_value(dataset_path=self.base_config.dataset_path))
        const = self.base_config.constants

        # replace iterator identifiers
        template = self.base_config.pass_value(template=self.base_config.template)
        for num, it_n in enumerate(it_names):
            template = template.replace("${}$".format(it_n), permutation[num])
        # replace constant identifiers
        for num, c in enumerate(const.keys()):
            if template.__contains__("${}$".format(c)):
                template = template.replace("${}$".format(c), const[c])

        # replace image identifiers
        if self.base_config.prepro_enable:
            template = template.replace("$imgL$", self.datasets[dataset]["left_f"])
            template = template.replace("$imgR$", self.datasets[dataset]["right_f"])
        else:
            template = template.replace("$imgL$", self.datasets[dataset]["left"])
            template = template.replace("$imgR$", self.datasets[dataset]["right"])
        template = template.replace("$out$", self.base_config.pass_value(output_path=self.base_config.output_path))

        # optional built in idetifiers
        template = template.replace("$base_dir$",
                                    self.base_config.pass_value(dataset_path=self.base_config.dataset_path))
        template = template.replace("$curr_dataset_dir$", os.path.abspath(os.path.join(self.datasets[dataset]["left"],
                                                                                       os.pardir)))
        # if template contains still a keyword raise a exception
        if template.__contains__('$'):
            raise ConfigurationError("There are some keywords in the template, that are not covered by constants or "
                                     "iterators. Please check the template: {}".format(template))

        return template

    def run(self, results_name="results.csv"):
        """
        Main function to run the set up evaluation process. Function iterates over all datasets. It executes
        the stereo algorithm for each permutation of set iterators and with constants.

        :param results_name: name of the resulting .csv file. Default: "results.csv"
        :return: None
        """
        # check if name is suiting:
        if not results_name.endswith(".csv"):
            raise EvaluationError("The filename given must be of type .csv")
        # validating filename
        valid_chars = "-_.() {}{}".format(string.ascii_letters, string.digits)
        validated = ''.join(c for c in results_name if c in valid_chars)
        validated = validated.replace(' ', '_')
        if validated != results_name:
            lg.info("Given filename was invalid, new filename is '{}'".format(validated))
            results_name = validated

        if not self.check_execution_template():
            raise ConfigurationError("Execution Template not sufficient!")
        # go to base directory
        base_dir = self.base_config.pass_value(dataset_path=self.base_config.dataset_path)
        os.chdir(base_dir)

        # get permutations of iterators
        iterators = self.base_config.iterators
        permutations = list(itertools.product(*iterators.values()))
        iterator_names = list(iterators.keys())
        constants = self.base_config.constants
        lg.debug(constants)
        # add constants to list for results csv
        voi = {str(c): constants[c] for c in constants.keys()}

        lg.debug("Iterators: {}".format(iterator_names))
        lg.debug("Permutaions: {}".format(permutations))

        self._localize_dataset()
        total_iteratos = len(self.datasets.keys()) * len(permutations)
        counter = 0
        # run through each image in dataset
        for dataset in self.datasets.keys():
            # run through each permutation of iterator values
            for perm in permutations:
                # monitoring
                counter += 1
                lg.info("Process step {}/{}.".format(counter, total_iteratos))
                lg.info("Executing evaluation for image pair '{}'".format(dataset))
                lg.debug("Configuration: {}".format({iterator_names[x]: perm[x] for x in range(len(perm))}))
                # build the shell command with currend permutations and dataset
                shell_cmd = self._build_shell_command(permutation=perm, it_names=iterator_names, dataset=dataset)
                # start the Executor process
                ret_code = Executor(config=self.base_config, command=shell_cmd).execute()
                # if execution is successful, return code should be True
                if not ret_code:
                    lg.error("Execution interrupted")
                    break

                # add iterators to list for results csv
                for num, it in enumerate(iterator_names):
                    voi[str(it)] = perm[num]
                # run the evaluation process
                Evaluator(config=self.base_config,
                          img_ev_path=self.base_config.pass_value(output_path=self.base_config.output_path),
                          img_gt_path=self.datasets[dataset]["gt"], img_base_path=self.datasets[dataset]["left"],
                          dataset_name=dataset, v_oi=voi, results_name=results_name)
            # for-else clause breaks the nested for-loop in case the return code is False
            else:
                continue
            break
        else:
            lg.info("Done")

    def plot(self, y_category, x_category, filters=None, title="Plot", style="default", unique=False,
             results_name="results.csv", relative_base=None, axis_label=None, error_disable=False):
        """
        Wrapper function for plotting graphs
        A plot can be created by selecting up to 8 data categories (y_category), which value is plotted grouped by a data category (x_category).
        The whole data from which the plot is created can be filtered by multiple filters.

        :param y_category: categories the data is grouped along x axis. multiple can be selected. maximum amount is 6. (list)
        :param x_category: data values plotted along y axis
        :param filters: categories and values, the dataset is filtered (dict) of format {column: value, column2: value2}
        :param title: title of the plot
        :param style: style of the plot. whisker boxplot or dafult line plot with error bars
        :param unique: whether the name of the resulting file should be unique.
        :param results_name: name of results file located in the set results directory
        :param relative_base: to which value of the x_cat the relative aggregation is based on
        :param axis_label: Tuple of x_axis and y_axis labels
        :param error_disable: Disables error bar
        :return: None
        """
        # check if input is valid
        if y_category is None or type(y_category) not in [str, list]:
            raise TypeError("y_category must be type str or list.")
        if type(y_category) is list:
            for y in y_category:
                if type(y) is not str:
                    raise TypeError("elements of y_category must be type str.")
        if x_category is None or type(x_category) is not str:
            raise TypeError("x_category must be type str.")
        if filters is not None and type(filters) is not dict:
            raise TypeError("filters must be type dict.")
        if type(title) is not str:
            raise TypeError("title must be type str.")
        if type(style) is not str:
            raise TypeError("style must be type str.")
        if unique is not None and type(unique) is not bool:
            raise TypeError("unique must be type bool.")
        if type(results_name) is not str:
            raise TypeError("results_name must be type str.")
        if relative_base is not None and type(relative_base) not in [str, int, float]:
            raise TypeError("results_name must be type str, int or float")
        if axis_label is not None and type(axis_label[0]) is not str and type(axis_label[1]) is not str:
            raise TypeError("Axis labels must be str")
        if type(error_disable) is not bool:
            raise TypeError("error_disable must be bool")

        Plotter(self.base_config, y_category, x_category, filters, title, style, unique, results_name,
                relative_base, axis_label, error_disable).plot()


class Executor:
    """
    Execution class for executing python and exe programs
    """

    def __init__(self, config, command):
        """
        Initializer of class
        :param config: base config from main class
        :param command: shell command to be executed
        """
        self.config = config
        self.mode = self.config.pass_value(execution_mode=self.config.execution_mode)
        self.error_handling = self.config.execution_error_handling.lower()

        self.command = command
        self.target = self.config.pass_value(execution_target=self.config.execution_target)

        self.eval_keys = list()

    def _execute_generic(self):
        """
        Internal function to execute a generic executable with given arguments

        :return: None
        """
        try:
            # function calls anything from shell. if its executable it is going to execute
            p = subprocess.call([self.target] + self.command.split(" "))
            lg.debug(str(p))
        except OSError:
            raise ConfigurationError("Executable-call not possible. Please check execution mode.")

    def _execute_python(self):
        """
        Internal function to start a python script with given arguments in the current python env

        :return: None
        """
        sys.argv = [self.target] + self.command.split(" ")
        # locate current python environtment executable
        python_loc = sys.executable
        # using located executable and not the system default
        p = subprocess.call([python_loc] + sys.argv)
        if p == 1:
            raise ExecutionError

    def execute(self):
        """
        Main executing function. Catches errors and passes the return code.

        :return: Return Code (bool)
        """

        if self.error_handling.lower() not in ["ignore", "interrupt"]:
            raise ConfigurationError("Unknown error_handling mode '{}'".format(self.error_handling))
        # run execution and catchign ExecutionError
        try:
            if self.mode == "generic":
                self._execute_generic()
            elif self.mode == "python":
                self._execute_python()
        except ExecutionError:
            # select between error handling modes
            if self.error_handling == "ignore":
                lg.warning("Execution not possible. Error ignored.")
                return True
            else:  # self.error_handling == "interrupt"
                lg.error("An Error occured during execution. "
                         "Please fix target program or adjust confuiguration.")
                return False
        # if everything worked, return True and wait a bit for possbile OS delays of saving images
        time.sleep(3)
        return True


class Evaluator:
    """
    Evaluator class. All image metrics are calculated here and the results file is written
    """

    def __init__(self, config, img_base_path, eval_mode=True, img_gt_path=None, img_ev_path=None, dataset_name=None,
                 v_oi=None, results_name=None):
        """
        Initializer of class

        :param config: base config from main class
        :param img_base_path: path to left image
        :param eval_mode: if evaluation is necessary
        :param img_gt_path:  path to gt image
        :param img_ev_path: path to calculated image that is evaluated
        :param dataset_name: name of the current dataset
        :param v_oi: all vairables set in constants and iterators for the resulting csv file.
        """

        # read images are converted to float64 to avoid overflow issues in uint8
        self.base_config = config
        # check if images are correctly loading
        if img_ev_path is not None:
            img_load_ev = cv2.imread(img_ev_path, 0)
            if img_load_ev is None:
                raise EvaluationError("Output disparity image is not loading correctly. "
                                      "Output_distp_filepath in config could be not set correctly.\n"
                                      "Check: {}".format(self.base_config.output_path))
            self.img_ev = np.float64(img_load_ev)
        else:
            self.img_ev = None
        if img_gt_path is not None:
            img_load_gt = cv2.imread(img_gt_path, 0)
            if img_load_gt is None:
                raise EvaluationError("Ground truth disparity image is not loading correctly.")
            # multiply image with factor for rescaled disparity images
            self.img_gt = np.float64(img_load_gt)
            self.img_gt *= self.base_config.evaluation_gt_coefficient
        else:
            self.img_gt = None

        img_load_base = cv2.imread(img_base_path, 0)
        if img_load_base is None:
            raise EvaluationError("Left input image image is not loading correctly.")
        self.img_base = np.float64(img_load_base)

        self.dataset_name = dataset_name
        self.iteration_values = v_oi
        self.results_filename = results_name

        self.results_dict = dict()
        if eval_mode:
            # evaluation is directly run after initialisation
            self._run_evaluations()
            # results are written to file
            self._write_results_file()

    def _run_evaluations(self):
        """
        Internal function to run evaluations and image metrics. All results are stored in a dictionary

        :return: results dict
        """
        # basic image propertis
        lg.info("Evaluating")
        self.results_dict["width"] = self.img_base.shape[1]
        self.results_dict["height"] = self.img_base.shape[0]

        self.results_dict["dens_img"] = np.round(self._density(self.img_ev), 4)
        self.results_dict["dens_gt"] = np.round(self._density(self.img_gt), 4)
        # to scale density with real availabilty of disparity
        self.results_dict["dens_rel"] = np.round(self.results_dict["dens_img"] / self.results_dict["dens_gt"], 4)

        self.results_dict["err_total_1"] = np.round(self._masked_error_score(self.img_gt, self.img_ev, 1), 4)
        self.results_dict["err_total_5"] = np.round(self._masked_error_score(self.img_gt, self.img_ev, 5), 4)
        self.results_dict["err_total_10"] = np.round(self._masked_error_score(self.img_gt, self.img_ev, 10), 4)

        # mask evaluation image with map of all textureless regions
        tl_map = self._get_textureless_map()
        img_ev_tl = deepcopy(self.img_ev)
        img_ev_tl[tl_map != 1] = 0
        self.results_dict["dens_lowtexture"] = np.round(self._density(img_ev_tl), 4)
        self.results_dict["err_lowtexture_1"] = np.round(self._masked_error_score(self.img_gt, img_ev_tl, 1), 4)
        self.results_dict["err_lowtexture_5"] = np.round(self._masked_error_score(self.img_gt, img_ev_tl, 5), 4)
        self.results_dict["err_lowtexture_10"] = np.round(self._masked_error_score(self.img_gt, img_ev_tl, 10), 4)
        # mask evaluation image with map of all texture rich regions
        img_ev_th = deepcopy(self.img_ev)
        img_ev_th[tl_map == 1] = 0
        self.results_dict["dens_hightexture"] = np.round(self._density(img_ev_th), 4)
        self.results_dict["err_hightexture_1"] = np.round(self._masked_error_score(self.img_gt, img_ev_th, 1), 4)
        self.results_dict["err_hightexture_5"] = np.round(self._masked_error_score(self.img_gt, img_ev_th, 5), 4)
        self.results_dict["err_hightexture_10"] = np.round(self._masked_error_score(self.img_gt, img_ev_th, 10), 4)

        # mask evaluation image with map of all discontinued disparity regions
        d_map = self._get_discontinued_map()
        img_ev_d = deepcopy(self.img_ev)
        img_ev_d[d_map != 1] = 0
        self.results_dict["dens_disc"] = np.round(self._density(img_ev_d), 4)
        self.results_dict["err_disc_1"] = np.round(self._masked_error_score(self.img_gt, img_ev_d, 1), 4)
        self.results_dict["err_disc_5"] = np.round(self._masked_error_score(self.img_gt, img_ev_d, 5), 4)
        self.results_dict["err_disc_10"] = np.round(self._masked_error_score(self.img_gt, img_ev_d, 10), 4)
        # mask evaluation image with map of all non-discontinued disparity regions
        img_ev_nd = deepcopy(self.img_ev)
        img_ev_nd[d_map == 1] = 0
        self.results_dict["dens_nondisc"] = np.round(self._density(img_ev_nd), 4)
        self.results_dict["err_nondisc_1"] = np.round(self._masked_error_score(self.img_gt, img_ev_nd, 1), 4)
        self.results_dict["err_nondisc_5"] = np.round(self._masked_error_score(self.img_gt, img_ev_nd, 5), 4)
        self.results_dict["err_nondisc_10"] = np.round(self._masked_error_score(self.img_gt, img_ev_nd, 10), 4)

        # quality metrics
        self.results_dict["psnr_total"] = np.round(self._masked_psnr(self.img_gt, self.img_ev), 2)
        self.results_dict["psnr_lowtexture"] = np.round(self._masked_psnr(self.img_gt, img_ev_th), 2)
        self.results_dict["psnr_hightexture"] = np.round(self._masked_psnr(self.img_gt, img_ev_tl), 2)
        self.results_dict["psnr_disc"] = np.round(self._masked_psnr(self.img_gt, img_ev_d), 2)
        self.results_dict["psnr_nondisc"] = np.round(self._masked_psnr(self.img_gt, img_ev_nd), 2)

        self.results_dict["rms_total"] = np.round(self._masked_rms(self.img_gt, self.img_ev), 2)
        self.results_dict["rms_lowtexture"] = np.round(self._masked_rms(self.img_gt, img_ev_th), 2)
        self.results_dict["rms_hightexture"] = np.round(self._masked_rms(self.img_gt, img_ev_tl), 2)
        self.results_dict["rms_disc"] = np.round(self._masked_rms(self.img_gt, img_ev_d), 2)
        self.results_dict["rms_nondisc"] = np.round(self._masked_rms(self.img_gt, img_ev_nd), 2)

        # ssim implementation is not masked and not very good metric
        # self.results_dict["ssim"] = np.round(sm.compare_ssim(self.img_gt, self.img_ev), 4)

        return self.results_dict

    def _write_results_file(self):
        """
        Internal function to write the image evaluation results dictionary to a csv file.
        The header with column nams is generated based on given iterators, constants and evaluation metrics

        :return: None
        """
        # generating destination path
        results_path = os.path.join(self.base_config.pass_value(results_path=self.base_config.results_path),
                                    self.results_filename)

        # list of all index columns is put together
        index_list = ["index", "dataset"] + list(self.iteration_values.keys()) + list(self.results_dict.keys())
        # list gets converted to a string separeted by semicolon
        index_str = ";".join(index_list)
        lg.debug(index_list)

        # creating results file if its not yet existing
        if not os.path.isfile(results_path):
            with open(results_path, 'wt') as fr:
                # write header
                fr.write(index_str)

        # opening file and reading all contents
        with open(results_path) as f:
            content = f.readlines()
        # splitting lines and values to list of lists
        content = [";".join(x.strip().split(";")[1:]) for x in content]

        # prepare line of values in dict format
        value_dict = {"index": str(len(content)), "dataset": self.dataset_name}
        value_dict.update(self.iteration_values)
        value_dict.update(self.results_dict)
        lg.debug(value_dict)
        # join values to string in order of the header
        data_line = ";".join([str(value_dict[index]) for index in index_list])

        # check if current evaluation result is already in the results file
        # if entry is not in there, write as a new line
        if ";".join(data_line.split(";")[1:]) not in content:
            with open(results_path, 'a') as rf:
                lg.info("Writing to results table: " + data_line)
                rf.write("\n" + data_line)

    @staticmethod
    def show_map(image, region, name="Preview of region"):
        """
        Function to show a region colored in an image

        :param image: image
        :param region: region
        :param name: name of window
        :return:
        """
        # converting initial image to unit8 color
        image_out = cv2.cvtColor(np.uint8(image), cv2.COLOR_GRAY2BGR)
        overlay = np.zeros_like(image_out)
        # defining region in overlay
        overlay[region == 1, 2] = 255
        # blend overly over image with transparency of alpha=0.5
        cv2.addWeighted(overlay, 0.5, image_out, 1 - 0.5, 0, image_out)
        # display image
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.imshow(name, image_out)
        cv2.waitKey(0)

    def _get_textureless_map(self, display=False):
        """
        internal function to create the low texture masked image regions.

        :param display: if the map should be displayed
        :return: masked region
        """
        # filter with sobel
        sobelx64f = cv2.Sobel(self.img_base, cv2.CV_64F, 1, 0, ksize=3)
        # take abs and convert to uint8 and square
        abs_sobel64f = np.absolute(sobelx64f)
        sobel_8u = np.uint8(abs_sobel64f) ** 2
        # take average by 2d convolution with square np.ones array
        size_of_smoothening = self.base_config.evaluation_textureless_width
        kernel = np.ones((size_of_smoothening, size_of_smoothening), np.uint8) / (size_of_smoothening ** 2)
        edgemap = cv2.filter2D(sobel_8u, -1, kernel)
        # cv2.imshow("raw", np.uint8(edgemap))
        # cv2.waitKey(0)

        region = np.zeros_like(edgemap)
        region[edgemap < self.base_config.evaluation_textureless_threshold] = 1

        if display:
            self.show_map(self.img_base, region, "Textureless region")

        return region

    def _get_discontinued_map(self, display=False):
        """
        function to create create a map of discontinued disparity image regions.

        :param display: if the map should be displayed
        :return: masked region
        """
        # calcualte gradient (sum of both axis)
        gradient0 = np.abs(np.diff(np.float64(self.img_gt), axis=0))
        gradient0 = np.pad(gradient0, ((0, 1), (0, 0)), mode='edge')
        gradient1 = np.abs(np.diff(np.float64(self.img_gt), axis=1))
        gradient1 = np.pad(gradient1, ((0, 0), (0, 1)), mode='edge')
        gradient = gradient0 + gradient1

        # delete unacessable disparity values by deleting all differences that are on a adge to a black value in the gt
        # secondary a dilation is applied
        gradient[binary_dilation(self.img_gt == 0)] = 0

        # cv2.imshow("raw", np.uint8(gradient))
        # cv2.waitKey(0)
        # all values over threshold are 1, all below are 0
        discmap = np.uint8(np.zeros_like(gradient))
        discmap[gradient > self.base_config.evaluation_discont_threshold] = 1

        # calcualte iterations
        iterations = int((self.base_config.evaluation_discont_width - 1) / 2)
        discmap = binary_dilation(discmap, iterations=iterations)

        if display:
            self.show_map(self.img_base, discmap, "Discontinued disparity region")

        return discmap

    @staticmethod
    def _masked_rms(ground, img):
        """
        Internal function to calculated a masked root mean squared error over the image. Masked regions in both images are not taken into account.
        If a value in one of both images is equal to 0, its masked.

        :param ground: ground truth image
        :param img: evaluated image
        :return: RMS value (float)
        """
        # creating masked numpy array for both images
        img_masked = np.ma.masked_equal(img, 0)
        ground_masked = np.ma.masked_equal(ground, 0)
        # calculating rms
        return np.sqrt(np.mean(np.square(img_masked - ground_masked)))

    def _masked_psnr(self, ground, img):
        """
        Internal function to calculated a masked peak signal to noise ratio over the image.
        Masked regions in both images are not taken into account.
        If a value in one of both images is equal to 0, its masked.

        :param ground: ground truth image
        :param img: evaluated image
        :return: PSNR value (float)
        """
        # first calculate mse
        mse = self._masked_rms(ground, img) ** 2
        if mse == 0:
            return 100
        p_max = 255.0
        # convert mse to psnr
        return 20 * math.log10(p_max / math.sqrt(mse))

    @staticmethod
    def _density(img):
        """
        Internal function to calculate the density of a image. Density is the portion of pixels that arent invalid/masked.
        If a value in the image is equal to 0, its masked.

        :param img: image
        :return: density protion (float)
        """
        return 1 - np.count_nonzero(img == 0) / np.size(img)

    @staticmethod
    def _masked_error_score(ground, img, error_margin):
        """
        Internal function to calculate the error score of an image comparing the pixels to another image.
        If a pixel diverts from its ground truth by more than the error margin of the image (255),
        its counted as an error.
        Errors are only calculated in pixels, that are valid (unequal to 0) in both images.
        The Error score is calculated with [Amount of error pixels]/[Amount of total valid pixels].

        :param ground: ground truth image
        :param img: evaluated image
        :param error_margin: error margin
        :return: Error score (what percentage of valid pixels is off by more than the error margin)
        """
        # define masked numpy arrays
        img_ma = np.ma.masked_equal(np.float64(img), 0)
        ground_ma = np.ma.masked_equal(np.float64(ground), 0)
        # create an array where every pixel valid for both images
        results = np.ones_like(ground)
        results[(img == 0) | (ground == 0)] = 0
        # amount of pixels = 0 is the amount of valid pixels
        valid_values = np.ma.masked_equal(results, 0).count()
        # scaling error margin
        # error_margin = (error_margin / 100) * 255
        # the amount of pixel differences smaller than the error margin is counted and divided by the number of valid pixels
        # noinspection PyUnresolvedReferences
        return 1 - (np.abs(img_ma - ground_ma) <= error_margin).sum() / valid_values


class Plotter:
    """
    Plotter class for easily creating categorizing plots and filtering data
    """

    def __init__(self, config, y_cat, x_cat, filters=None, title=None, style=None, unique=None, results_name=None,
                 relative_base=None, axis_label=None, error_disable=None):
        """
        Initializer of class

        :param y_cat: categories the data is grouped along x axis
        :param x_cat: data values plotted along y axis
        :param filters: categories and values, the dataset is filtered (dict) of format {column: value, column2: value2}
        :param title: title of the plot
        :param style: style of the plot. whisker boxplot or dafult line plot with error bars
        :param unique: whether the name of the resulting file should be unique.
        :param results_name: name of results file located in the set results directory
        :param relative_base: to which value of the x_cat the relative aggregation is based on
        :param axis_label: Tuple of axis labels
        :param error_disable: To disable error bars
        """
        self.base_config = config

        self.y_cat = y_cat
        self.x_cat = x_cat
        self.filters = filters
        self.title = title
        self.style = style
        self.unique = unique
        self.results_name = results_name
        self.axis_label = axis_label
        self.error_disable = error_disable

        self.relative = False
        self.relative_base = relative_base

        if relative_base is not None:
            self.relative = True

        # data from the results
        self.data = self._read_data()

    def _read_data(self):
        """
        Internal function to read the data into a pandas array.

        :return: Dataframe
        """
        results_path = os.path.join(self.base_config.results_path, self.results_name)
        # check if file exists
        if os.path.isfile(self.base_config.results_path):
            raise OSError("{} not found. Please run evaluation first or check filepath.".format(results_path))
        # catch the error, when the file is empty
        try:
            return pd.read_csv(results_path, sep=";", na_values="n.a")
        except pandas.errors.EmptyDataError:
            raise ConfigurationError("{} file is empty. "
                                     "Please run evaluation first or check filepath.".format(results_path))

    @staticmethod
    def isfloat(value):
        """
        function for float checking
        :param value: value
        :return: if string is float
        """
        try:
            float(value)
            return True
        except ValueError:
            return False

    def _check_data(self):
        """
        Internal function to check if selected columns from x_cat, y_cat and filters are found in the dataframe.

        :return: If the data is complete
        """
        columns = list(self.data.columns)
        # check y_cat and x_cat
        if self.y_cat not in columns:
            raise KeyError("y_cat {} is not found in loaded results.csv".format(self.y_cat))
        if self.x_cat not in columns:
            raise KeyError("x_cat {} is not found in loaded results.csv".format(self.y_cat))
        # check filters
        if self.filters:
            for f in self.filters.keys():
                if f not in columns:
                    raise KeyError("filter {} is not found in loaded results.csv".format(f))

        return True

    def plot(self):
        """
        Main plotting function

        :return: None
        """

        # deepcopy dataframe to avoid possible view instead of copy
        df = deepcopy(self.data)
        # create a deepcopy for later mean calculation
        df_m = deepcopy(self.data)
        df = df.replace('--', np.nan, regex=True)
        # for every filter filter the dataframe by that value and category
        if self.filters:
            for f in self.filters.keys():
                df = df[df[f] == self.filters[f]]

        # Running relative aggregation on demand
        if self.relative:
            # aggregating based on base value
            df_r = deepcopy(df)
            # setting xcat as index and duplicate as column
            df_r.set_index(self.x_cat, inplace=True)
            df_r[self.x_cat] = df_r.index

            # get unique names of datasets
            datasets_names = list(set(df_r["dataset"].values))

            # evaluation column names
            columns = ["dens_img", "dens_gt", "dens_rel", "err_total_1", "err_total_5", "err_total_10",
                       "dens_lowtexture", "err_lowtexture_1", "err_lowtexture_5", "err_lowtexture_10",
                       "dens_hightexture", "err_hightexture_1", "err_hightexture_5", "err_hightexture_10", "dens_disc",
                       "err_disc_1", "err_disc_5", "err_disc_10", "dens_nondisc", "err_nondisc_1", "err_nondisc_5",
                       "err_nondisc_10", "psnr_total", "psnr_lowtexture", "psnr_hightexture", "psnr_disc",
                       "psnr_nondisc", "rms_total", "rms_lowtexture", "rms_hightexture", "rms_disc", "rms_nondisc"]
            # for every dataset folder get reference series based on base_value and subtract that from every other
            # element in that sub dataframe
            for name in datasets_names:
                # get filtered df for dataset
                base_value_frame = df_r[df_r["dataset"] == name][columns].apply(pd.to_numeric, errors="coerce")
                # get series for reference/base value
                base_value_series = base_value_frame.loc[self.relative_base]
                # subtract that from dataframe
                sub_frame = base_value_frame.sub(base_value_series)
                # return values to original dataframe
                df_r.loc[df_r["dataset"] == name, columns] = sub_frame.values

            # hand over dataframe to main process
            df = deepcopy(df_r)

        # plotting for style default
        if self.style == "default":
            df_d = deepcopy(df)
            # get colormaps
            markercolors = plt.get_cmap("Dark2").colors
            ebarcolors = plt.get_cmap("Set2").colors
            # check if its type list and length is not to long
            if type(self.y_cat) is not list:
                self.y_cat = [self.y_cat]
            if len(self.y_cat) > len(markercolors):
                raise ConfigurationError("A Maximum of {} y_categories is supported.".format(len(markercolors)))

            # create second dataframe to create error bars and mean of values
            df_d.set_index(self.x_cat, inplace=True)
            # delete all non numeric values
            df2 = df_d[self.y_cat].apply(pd.to_numeric, errors="coerce")

            # group by grouping category (x_cat) along x axis
            gp = df2.groupby(level=self.x_cat)
            # calculate means and errors
            means = gp.mean()
            error = gp.std()
            fig, ax = plt.subplots()

            # for each y category plot a means line and a errorbar
            for num, cat in enumerate(self.y_cat):
                means[cat].plot(ax=ax, marker="^", markersize=8, markerfacecolor='w',
                                markeredgewidth=1.5, markeredgecolor=markercolors[num],
                                color="k", linestyle="--")
                if not self.error_disable:
                    ax.errorbar(x=error.index, y=means[cat].fillna(0), yerr=error[cat], fmt="", linestyle="",
                                capsize=6, ecolor=ebarcolors[num], label=None)

            # add a legend to the plot
            ax.legend()

        # plotting for style whisker
        elif self.style == "whisker":
            df_w = deepcopy(df)
            # catching condition, that a list of y_cat is given for style whisker
            if type(self.y_cat) is list:
                if len(self.y_cat) == 1:
                    self.y_cat = self.y_cat[0]
                else:
                    raise ConfigurationError("Only 1 Y-Category can be given for style 'whisker'")

            groups = sorted(list(set(df[self.x_cat].values.tolist())))
            data = [df_w[df_w[self.x_cat] == g][self.y_cat].values.tolist() for g in sorted(groups)]

            # delete all non numeric entries
            # delete all nan entries
            for num, dataline in enumerate(data):
                new_line = list()
                for num2, datapoint in enumerate(dataline):
                    if self.isfloat(datapoint) and not math.isnan(float(datapoint)):
                        new_line.append(float(datapoint))
                data[num] = new_line

            fig, ax = plt.subplots()
            ax.boxplot(data, notch=False, labels=groups)
        else:
            raise ConfigurationError("Style '{}' unknown. Please use 'default' or 'whisker'".format(self.style))

        if self.relative:
            df_m.set_index(self.x_cat, inplace=True)
            # delete all non numeric values
            df_mc = df_m[self.y_cat].apply(pd.to_numeric, errors="coerce")

            # group by grouping category (x_cat) along x axis
            gp = df_mc.groupby(level=self.x_cat)
            # calculate means and errors
            means = gp.mean()
            # Make string of reference values
            means_list = [np.round(means.loc[self.relative_base, cat], 3) for cat in self.y_cat]
            means_str_list = ["{}={}".format(cat, means_list[num]) for num, cat in enumerate(self.y_cat)]
            means_str = "Reference Values (at {}={}): {}".format(self.x_cat, self.relative_base,
                                                                 ", ".join(means_str_list))
            # set text in figure
            plt.figtext(0.99, 0.02, means_str, horizontalalignment='right', wrap=True)

        # set title and axis descriptors
        if self.title == "Plot" and self.relative:
            self.title = "Relative Plot"
        ax.set_title(self.title)
        if self.axis_label is not None:
            plt.xlabel(self.axis_label[0])
            plt.ylabel(self.axis_label[1], wrap=True)
        else:
            plt.xlabel(self.x_cat)
            plt.ylabel(self.y_cat, wrap=True)
        # adjust layout
        fig.tight_layout(rect=[0.05, 0.04, 1, 1])

        # if unique plot name is required, add date and time to filename
        name = os.path.splitext(self.results_name)[0]
        if self.unique:
            filename = strftime("{}_%Y%m%d_%H%M%S".format(name), gmtime())
        else:
            filename = "{}.png".format(name)

        # save plot to png file
        plot_path = os.path.join(self.base_config.results_path, filename)
        plt.savefig(plot_path, dpi=300)
        lg.info("Plot saved to: {}".format(plot_path))

        time.sleep(1)


class ConfigurationError(Exception):
    pass


class ExecutionError(Exception):
    pass


class EvaluationError(Exception):
    pass


if __name__ == "__main__":
    lg.info("Start")

    a = AlgorithmEvaluator(config_path="/Users/benedikt/PycharmProjects/software/eval_config.ini")
    a.plot(y_category=["dens_img"], x_category="window_size", style="default",
           results_name="lvl1_NCC.csv", unique=True)
