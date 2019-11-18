import sys
sys.path.insert(0, './network/')

from deformable_kpn import deformable_kpn
from deformable_kpn_modify_1 import deformable_kpn_modify_1
from deformable_kpn_raw import deformable_kpn_raw
from kpn import kpn
from kpn_raw import kpn_raw
from deeptof import deeptof
from dear_kpn import dear_kpn
from dear_unet import dear_unet
from deformable_pyramid_kpn import deformable_pyramid_kpn



NETWORK_NAME = {
    'deformable_kpn': deformable_kpn,
    'deformable_kpn_modify_1': deformable_kpn_modify_1,
    'deformable_kpn_raw': deformable_kpn_raw,
    'deformable_pyramid_kpn': deformable_pyramid_kpn,
    'kpn': kpn,
    'kpn_raw': kpn_raw,
    'deeptof': deeptof,
    'dear_unet': dear_unet,
    'dear_kpn': dear_kpn,
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
