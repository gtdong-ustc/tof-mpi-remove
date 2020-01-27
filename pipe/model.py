import sys
sys.path.insert(0, './network/')

from dear_kpn_no_rgb import dear_kpn_no_rgb
from sample_pyramid_add_kpn import sample_pyramid_add_kpn
from sample_pyramid_add_kpn_NoRefine import sample_pyramid_add_kpn_NoRefine
from sample_pyramid_add_kpn_NoFusion import sample_pyramid_add_kpn_NoFusion
from sample_pyramid_add_kpn_NoRefineFusion import sample_pyramid_add_kpn_NoRefineFusion
from sample_pyramid_add_kpn_FiveLevel import sample_pyramid_add_kpn_FiveLevel
from sample_pyramid_add_kpn_FourLevel import sample_pyramid_add_kpn_FourLevel
from dear_kpn_no_rgb_DeepToF import dear_kpn_no_rgb_DeepToF



NETWORK_NAME = {
    'sample_pyramid_add_kpn':sample_pyramid_add_kpn,
    'sample_pyramid_add_kpn_NoRefine':sample_pyramid_add_kpn_NoRefine,
    'sample_pyramid_add_kpn_NoFusion':sample_pyramid_add_kpn_NoFusion,
    'sample_pyramid_add_kpn_NoRefineFusion':sample_pyramid_add_kpn_NoRefineFusion,
    'sample_pyramid_add_kpn_FiveLevel':sample_pyramid_add_kpn_FiveLevel,
    'sample_pyramid_add_kpn_FourLevel':sample_pyramid_add_kpn_FourLevel,
    'dear_kpn_no_rgb_DeepToF':dear_kpn_no_rgb_DeepToF,
    'dear_kpn_no_rgb':dear_kpn_no_rgb,
}

ALL_NETWORKS = dict(NETWORK_NAME)


def get_network(name, x, flg, regular, batch_size, range):
    """
    this function is used to selected the network
    :param name: network name
    :param x: network input, such as depth, amplitude, raw measurement, 
    :param flg: Indicates whether the code is in training mode
    :param regular: Regularization parameter(not be used)
    :param batch_size: 
    :param range: kernel Deformable range(used in deformable kernel network)
    :return: 
    """
    if name not in NETWORK_NAME.keys():
        print('Unrecognized network, pick one among: {}'.format(ALL_NETWORKS.keys()))
        raise Exception('Unknown network selected')
    selected_network = ALL_NETWORKS[name]
    return selected_network(x, flg, regular, batch_size, range)
