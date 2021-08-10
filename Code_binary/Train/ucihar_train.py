import os
import json
os.chdir(os.path.join(os.getcwd(), 'Code', 'Search'))

from modules.layers import *
from data_providers.ucihar import UCIHARDataProvider

def proxyless_base(net_config=None, n_classes=6, bn_param=(0.1, 1e-3), dropout_rate=0.05):
    assert net_config is not None, 'Please input a network config'
    # net_config_path = download_url(net_config)
    net_config_path = net_config
    net_config_json = json.load(open(net_config_path, 'r'))

    net_config_json['classifier2']['out_features'] = n_classes
    net_config_json['classifier2']['dropout_rate'] = dropout_rate

    net = ProxylessNASNets.build_from_config(net_config_json)
    net.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    return net

net = proxyless_base(net_config='./logs/ucihar/warmp20_gr/learned_net/net.config')