import sys
sys.path.insert(0, './network/')

from deformable_kpn import deformable_kpn
from deformable_kpn_raw import deformable_kpn_raw
from kpn import kpn
from kpn_raw import kpn_raw
from deeptof import deeptof


NETWORK_NAME = {
    'deformable_kpn': deformable_kpn,
    'deformable_kpn_raw': deformable_kpn_raw,
    'kpn': kpn,
    'kpn_raw': kpn_raw,
    'deeptof': deeptof,
}

ALL_NETWORKS = dict(NETWORK_NAME)


def get_network(name, x, flg, regular, batch_size, range):
    if name not in NETWORK_NAME.keys():
        print('Unrecognized network, pick one among: {}'.format(ALL_NETWORKS.keys()))
        raise Exception('Unknown network selected')
    selected_network = ALL_NETWORKS[name]
    return selected_network(x, flg, regular, batch_size, range)
