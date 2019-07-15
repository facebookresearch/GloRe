# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

def get_config(name, **kwargs):

    logging.debug("loading network configs of: {}".format(name.upper()))

    config = {}

    logging.info("Preprocessing:: using MXNet default mean & std.")
    config['mean'] = [124 / 255, 117 / 255, 104 / 255]
    config['std'] = [1 / (.0167 * 255)] * 3

    logging.info("data:: {}".format(config))
    return config
