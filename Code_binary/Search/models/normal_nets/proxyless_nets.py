# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.
# import os
# import sys
# sys.path.append(os.path.join('H:\\Develop\\For_Journal_Paper', 'Code', 'Search'))
# os.chdir(os.path.join('H:\\Develop\\For_Journal_Paper', 'Code', 'Search'))

from modules.layers import *
import json

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


class ConvResidualBlock(MyModule):

    def __init__(self, conv, shortcut):
        super(ConvResidualBlock, self).__init__()

        self.conv = conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.conv.is_zero_layer(): # ZeroLayer is identity
            res = x
        elif self.shortcut is None or self.shortcut.is_zero_layer():
            res = self.conv(x)
        else:
            conv_x = self.conv(x)
            skip_x = self.shortcut(x)
            res = skip_x + conv_x
        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.conv.module_str, self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': ConvResidualBlock.__name__,
            'conv': self.conv.config,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        conv = set_layer_from_config(config['conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return ConvResidualBlock(conv, shortcut)

    def get_flops(self, x):
        flops1, conv_x = self.conv.get_flops(x)
        if self.shortcut:
            flops2, _ = self.shortcut.get_flops(x)
        else:
            flops2 = 0

        return flops1 + flops2, self.forward(x)

class PoolingBlock(MyModule):

    def __init__(self, pool):
        super(PoolingBlock, self).__init__()

        self.pool = pool

    def forward(self, x):
        res = self.pool(x)
        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.pool.module_str, self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': PoolingBlock.__name__,
            'pool': self.pool.config,
        }

    @staticmethod
    def build_from_config(config):
        pool = set_layer_from_config(config['pool'])
        return PoolingBlock(pool)

    def get_flops(self, x):
        return 0, self.forward(x)

class ProxylessNASNets(MyNetwork):

    def __init__(self, blocks, pooling_blocks, classifier1, classifier2):
        super(ProxylessNASNets, self).__init__()

        self.blocks = nn.ModuleList(blocks)
        self.pooling_blocks = nn.ModuleList(pooling_blocks)
        # self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier1 = classifier1
        self.classifier2 = classifier2

    def forward(self, x):
        for block, pool_block in zip(self.blocks, self.pooling_blocks):
            x = block(x)
            x = pool_block(x)
        # x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier1(x)
        x = self.classifier2(x)
        return x

    @property
    def module_str(self):
        _str = ''
        for block in self.blocks:
            _str += block.unit_str + '\n'
        return _str

    @property
    def config(self):
        return {
            'name': ProxylessNASNets.__name__,
            'bn': self.get_bn_param(),
            'blocks': [
                block.config for block in self.blocks
            ],
            'pool_blocks': [
                pool_block.config for pool_block in self.pooling_blocks
            ],
            'classifier1': self.classifier1.config,
            'classifier2': self.classifier2.config,
        }

    @staticmethod
    def build_from_config(config):
        classifier1 = set_layer_from_config(config['classifier1'])
        classifier2 = set_layer_from_config(config['classifier2'])
        blocks = []
        pool_blocks = []
        for block_config in config['blocks']:
            blocks.append(ConvResidualBlock.build_from_config(block_config))
        for block_config in config['pool_blocks']:
            pool_blocks.append(PoolingBlock.build_from_config(block_config))

        net = ProxylessNASNets(blocks, pool_blocks, classifier1, classifier2)
        if 'bn' in config:
            net.set_bn_param(**config['bn'])
        else:
            net.set_bn_param(momentum=0.1, eps=1e-3)

        return net

    def get_flops(self, x):
        flop = 0
        for block, pool_block in zip(self.blocks, self.pooling_blocks):
            delta_flop, x = block.get_flops(x)
            flop += delta_flop
            x = pool_block(x)

        # x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten

        delta_flop, x = self.classifier1.get_flops(x)
        flop += delta_flop
        delta_flop, x = self.classifier2.get_flops(x)
        return flop, x


# Not Used
class MobileInvertedResidualBlock(MyModule):

    def __init__(self, mobile_inverted_conv, shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv.is_zero_layer():
            res = x
        elif self.shortcut is None or self.shortcut.is_zero_layer():
            res = self.mobile_inverted_conv(x)
        else:
            conv_x = self.mobile_inverted_conv(x)
            skip_x = self.shortcut(x)
            res = skip_x + conv_x
        return res

    @property
    def module_str(self):
        return '(%s, %s)' % (
            self.mobile_inverted_conv.module_str, self.shortcut.module_str if self.shortcut is not None else None
        )

    @property
    def config(self):
        return {
            'name': MobileInvertedResidualBlock.__name__,
            'mobile_inverted_conv': self.mobile_inverted_conv.config,
            'shortcut': self.shortcut.config if self.shortcut is not None else None,
        }

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)

    def get_flops(self, x):
        flops1, conv_x = self.mobile_inverted_conv.get_flops(x)
        if self.shortcut:
            flops2, _ = self.shortcut.get_flops(x)
        else:
            flops2 = 0

        return flops1 + flops2, self.forward(x)


if __name__ == "__main__":
    os.getcwd()
    net = proxyless_base(net_config='./logs/ucihar/warmp20_gr/learned_net/net.config')
    input = torch.randn(1, 6, 128)
    print(net.get_flops(input)[0] * 1e-6)