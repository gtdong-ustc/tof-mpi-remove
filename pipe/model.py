import sys
sys.path.insert(0, './network/')

from deformable_kpn import deformable_kpn
from deformable_kpn_modify_1 import deformable_kpn_modify_1
from deformable_kpn_raw import deformable_kpn_raw
from kpn import kpn
from kpn_raw import kpn_raw
from deeptof import deeptof
from dear_kpn import dear_kpn
from dear_kpn_only_depth import dear_kpn_only_depth
from dear_kpn_no_rgb import dear_kpn_no_rgb
from dear_unet import dear_unet
from deformable_pyramid_kpn import deformable_pyramid_kpn
from deformable_full_pyramid_kpn import deformable_full_pyramid_kpn
from deformable_pyramid_add_dof_kpn import deformable_pyramid_add_dof_kpn
from deformable_full_variant_1_pyramid_kpn import deformable_full_variant_1_pyramid_kpn
from deformable_full_variant_2_pyramid_kpn import deformable_full_variant_2_pyramid_kpn
from deformable_full_variant_3_pyramid_kpn import deformable_full_variant_3_pyramid_kpn
from deformable_full_variant_4_pyramid_kpn import deformable_full_variant_4_pyramid_kpn
from deformable_full_variant_5_pyramid_kpn import deformable_full_variant_5_pyramid_kpn
from deformable_full_variant_6_pyramid_kpn import deformable_full_variant_6_pyramid_kpn
from deformable_full_variant_7_pyramid_kpn import deformable_full_variant_7_pyramid_kpn
from deformable_full_variant_8_pyramid_kpn import deformable_full_variant_8_pyramid_kpn
from deformable_full_variant_7_2_pyramid_kpn import deformable_full_variant_7_2_pyramid_kpn
from ddfn import ddfn
from pyramid_ddfn import pyramid_ddfn
from pyramid import pyramid
from pyramid_add_kpn_without_unet import pyramid_add_kpn_without_unet
from pyramid_add_kpn_without_unet_change_channel import pyramid_add_kpn_without_unet_change_channel
from pyramid_add_kpn_without_unet_change_dilate import pyramid_add_kpn_without_unet_change_dilate
from pyramid_add_rgb import pyramid_add_rgb
from pyramid_no_refinement import pyramid_no_refinement
from pyramid_no_refinement_add_rgb import pyramid_no_refinement_add_rgb
from pyramid_kpn import pyramid_kpn
from pyramid_add_kpn import pyramid_add_kpn
from pyramid_only_refinement import pyramid_only_refinement
from sample_pyramid_add_kpn import sample_pyramid_add_kpn
from sample_pyramid_add_kpn_only_depth import sample_pyramid_add_kpn_only_depth
from sample_pyramid_change_spp_add_kpn import sample_pyramid_change_spp_add_kpn
from sample_pyramid_add_kpn_residual import sample_pyramid_add_kpn_residual
from sample_pyramid_change_regression import sample_pyramid_change_regression
from sample_pyramid_change_regression_1 import sample_pyramid_change_regression_1
from sample_pyramid_change_regression_1_add_kpn import sample_pyramid_change_regression_1_add_kpn
from sample_pyramid_change_regression_2 import sample_pyramid_change_regression_2
from sample_pyramid_change_regression_2_add_kpn import sample_pyramid_change_regression_2_add_kpn
from sample_pyramid_change_regression_4 import sample_pyramid_change_regression_4
from sample_pyramid_change_regression_3 import sample_pyramid_change_regression_3
from sample_pyramid_add_one_scale import sample_pyramid_add_one_scale
from sample_pyramid_substract_one_scale import sample_pyramid_substract_one_scale
from sample_pyramid_convert_transpose_conv import sample_pyramid_convert_transpose_conv
from sample_pyramid_with_psp import sample_pyramid_with_psp
from sample_pyramid_with_spp_full_para import sample_pyramid_with_spp_full_para
from sample_pyramid_with_spp_full_para_substract_scale import sample_pyramid_with_spp_full_para_substract_scale
from sample_pyramid_with_psp_add_connect import sample_pyramid_with_psp_add_connect
from sample_pyramid_with_spp_add_one_scale import sample_pyramid_with_spp_add_one_scale
from sample_pyramid_with_spp_add_scale_uconv import sample_pyramid_with_spp_add_scale_uconv
from sample_pyramid_with_spp_change_add_one_scale import sample_pyramid_with_spp_change_add_one_scale
from sample_pyramid_with_sppno_change_add_one_scale import sample_pyramid_with_sppno_change_add_one_scale
from sample_pyramid_add_res2net import sample_pyramid_add_res2net
from sample_pyramid_with_spp_add_se_block import sample_pyramid_with_spp_add_se_block
from sample_pyramid_with_spp_add_scale_and_se import sample_pyramid_with_spp_add_scale_and_se
from sample_pyramid_with_spp_add_se_block_0 import sample_pyramid_with_spp_add_se_block_0
from sample_pyramid_add_kpn_NoRefine import sample_pyramid_add_kpn_NoRefine
from sample_pyramid_add_kpn_NoFusion import sample_pyramid_add_kpn_NoFusion
from sample_pyramid_add_kpn_NoRefineFusion import sample_pyramid_add_kpn_NoRefineFusion
from sample_pyramid_add_kpn_FiveLevel import sample_pyramid_add_kpn_FiveLevel
from sample_pyramid_add_kpn_FourLevel import sample_pyramid_add_kpn_FourLevel
from dear_kpn_no_rgb_DeepToF import dear_kpn_no_rgb_DeepToF



NETWORK_NAME = {
    'deformable_kpn': deformable_kpn,
    'deformable_kpn_modify_1': deformable_kpn_modify_1,
    'deformable_kpn_raw': deformable_kpn_raw,
    'deformable_pyramid_kpn': deformable_pyramid_kpn,
    'deformable_full_pyramid_kpn':deformable_full_pyramid_kpn,
    'deformable_pyramid_add_dof_kpn':deformable_pyramid_add_dof_kpn,
    'deformable_full_variant_1_pyramid_kpn':deformable_full_variant_1_pyramid_kpn,
    'deformable_full_variant_2_pyramid_kpn':deformable_full_variant_2_pyramid_kpn,
    'deformable_full_variant_3_pyramid_kpn':deformable_full_variant_3_pyramid_kpn,
    'deformable_full_variant_4_pyramid_kpn':deformable_full_variant_4_pyramid_kpn,
    'deformable_full_variant_5_pyramid_kpn':deformable_full_variant_5_pyramid_kpn,
    'deformable_full_variant_6_pyramid_kpn':deformable_full_variant_6_pyramid_kpn,
    'deformable_full_variant_7_pyramid_kpn':deformable_full_variant_7_pyramid_kpn,
    'deformable_full_variant_7_2_pyramid_kpn': deformable_full_variant_7_2_pyramid_kpn,
    'deformable_full_variant_8_pyramid_kpn':deformable_full_variant_8_pyramid_kpn,
    'ddfn': ddfn,
    'pyramid_ddfn':pyramid_ddfn,
    'pyramid_kpn':pyramid_kpn,
    'pyramid_add_kpn':pyramid_add_kpn,
    'pyramid_add_kpn_without_unet':pyramid_add_kpn_without_unet,
    'pyramid_add_kpn_without_unet_change_channel':pyramid_add_kpn_without_unet_change_channel,
    'pyramid_add_kpn_without_unet_change_dilate':pyramid_add_kpn_without_unet_change_dilate,
    'pyramid':pyramid,
    'pyramid_add_rgb':pyramid_add_rgb,
    'pyramid_no_refinement':pyramid_no_refinement,
    'pyramid_no_refinement_add_rgb':pyramid_no_refinement_add_rgb,
    'pyramid_only_refinement':pyramid_only_refinement,
    'sample_pyramid_add_kpn':sample_pyramid_add_kpn,
    'sample_pyramid_add_kpn_only_depth':sample_pyramid_add_kpn_only_depth,
    'sample_pyramid_change_spp_add_kpn':sample_pyramid_change_spp_add_kpn,
    'sample_pyramid_add_kpn_residual':sample_pyramid_add_kpn_residual,
    'sample_pyramid_change_regression':sample_pyramid_change_regression,
    'sample_pyramid_change_regression_1':sample_pyramid_change_regression_1,
    'sample_pyramid_change_regression_1_add_kpn':sample_pyramid_change_regression_1_add_kpn,
    'sample_pyramid_change_regression_2':sample_pyramid_change_regression_2,
    'sample_pyramid_change_regression_2_add_kpn':sample_pyramid_change_regression_2_add_kpn,
    'sample_pyramid_change_regression_3':sample_pyramid_change_regression_3,
    'sample_pyramid_change_regression_4':sample_pyramid_change_regression_4,
    'sample_pyramid_add_one_scale':sample_pyramid_add_one_scale,
    'sample_pyramid_substract_one_scale':sample_pyramid_substract_one_scale,
    'sample_pyramid_convert_transpose_conv':sample_pyramid_convert_transpose_conv,
    'sample_pyramid_with_psp':sample_pyramid_with_psp,
    'sample_pyramid_with_spp_full_para':sample_pyramid_with_spp_full_para,
    'sample_pyramid_with_spp_full_para_substract_scale':sample_pyramid_with_spp_full_para_substract_scale,
    'sample_pyramid_with_psp_add_connect':sample_pyramid_with_psp_add_connect,
    'sample_pyramid_with_spp_add_one_scale':sample_pyramid_with_spp_add_one_scale,
    'sample_pyramid_with_spp_add_scale_uconv':sample_pyramid_with_spp_add_scale_uconv,
    'sample_pyramid_with_spp_change_add_one_scale':sample_pyramid_with_spp_change_add_one_scale,
    'sample_pyramid_with_sppno_change_add_one_scale':sample_pyramid_with_sppno_change_add_one_scale,
    'sample_pyramid_add_res2net':sample_pyramid_add_res2net,
    'sample_pyramid_with_spp_add_se_block':sample_pyramid_with_spp_add_se_block,
    'sample_pyramid_with_spp_add_scale_and_se':sample_pyramid_with_spp_add_scale_and_se,
    'sample_pyramid_with_spp_add_se_block_0':sample_pyramid_with_spp_add_se_block_0,
    'sample_pyramid_add_kpn_NoRefine':sample_pyramid_add_kpn_NoRefine,
    'sample_pyramid_add_kpn_NoFusion':sample_pyramid_add_kpn_NoFusion,
    'sample_pyramid_add_kpn_NoRefineFusion':sample_pyramid_add_kpn_NoRefineFusion,
    'sample_pyramid_add_kpn_FiveLevel':sample_pyramid_add_kpn_FiveLevel,
    'sample_pyramid_add_kpn_FourLevel':sample_pyramid_add_kpn_FourLevel,
    'dear_kpn_no_rgb_DeepToF':dear_kpn_no_rgb_DeepToF,
    'kpn': kpn,
    'kpn_raw': kpn_raw,
    'deeptof': deeptof,
    'dear_unet': dear_unet,
    'dear_kpn': dear_kpn,
    'dear_kpn_only_depth':dear_kpn_only_depth,
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
