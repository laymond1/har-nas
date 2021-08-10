# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

from queue import Queue
import copy
import math

from modules.mix_op import *
from models.normal_nets.proxyless_nets import *
from utils import LatencyEstimator


# import sys
# import os
# os.getcwd()
# sys.path.append(os.path.join(os.getcwd(), 'Code', 'Search'))

# SuperProxylessNASNets([64, 64, 64], [1,1,1], )

class SuperProxylessNASNets(ProxylessNASNets):

    def __init__(self, width_stages, n_cell_stages, conv_candidates, stride_stages,
                 input_channel, n_classes, seg_size, width_mult=1, bn_param=(0.1, 1e-3), dropout_rate=0.05):
        self._redundant_modules = None
        self._unused_modules = None

        self.n_classes = n_classes
        self.seg_size = seg_size
        self.input_channel = input_channel

        # blocks
        blocks = []
        pooling_blocks = []
        count = 0
        self.seg_length = math.floor((seg_size + 2) /3)
        for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages): # channel_size, num_of_cell, stride
            for i in range(n_cell):
                if i == 0:
                    stride = s
                else:
                    stride = 1
                # conv
                if stride == 1 and self.input_channel == width: # and count != 0: # same channel size
                    modified_conv_candidates = conv_candidates + ['Zero']
                else:
                    modified_conv_candidates = conv_candidates

                # Mixed Operation
                if count == 0: # first layer
                    conv_op = MixedEdge(candidate_ops=build_candidate_ops(
                    modified_conv_candidates, self.input_channel, width, stride, 'weight_bn_act',
                    ), )
                    maxpool = PoolingLayer(self.input_channel, width, 'max', kernel_size=3, stride=3)
                    self.seg_length = math.floor((seg_size + 2) /3)
                else:
                    conv_op = MixedEdge(candidate_ops=build_candidate_ops(
                        modified_conv_candidates, self.input_channel, width, stride, 'weight_bn_act',
                    ), )
                    maxpool = PoolingLayer(self.input_channel, width, 'max', kernel_size=3, stride=3)
                    self.seg_length = math.floor((self.seg_length + 2) /3)

                # shortcut 일단 제외
                # if stride == 1 and input_channel == width:
                #     shortcut = IdentityLayer(input_channel, input_channel)
                # else:
                shortcut = None
                conv_block = ConvResidualBlock(conv_op, shortcut)
                blocks.append(conv_block)
                pool_block = PoolingBlock(maxpool)
                pooling_blocks.append(pool_block)
                self.input_channel = width
            count += 1
        self.last_channel = self.seg_length * self.input_channel

        # classifier1 = LinearLayer(last_channel, 512, act_func='relu')
        classifier1 = LinearLayer(self.last_channel, 512, act_func='relu')
        classifier2 = LinearLayer(512, n_classes, dropout_rate=dropout_rate)
        super(SuperProxylessNASNets, self).__init__(blocks, pooling_blocks, classifier1, classifier2)

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

    @property
    def config(self):
        raise ValueError('not needed')

    @staticmethod
    def build_from_config(config):
        raise ValueError('not needed')

    """ weight parameters, arch_parameters & binary gates """

    def architecture_parameters(self):
        for name, param in self.named_parameters():
            if 'AP_path_alpha' in name:
                yield param

    def binary_gates(self): # not used
        for name, param in self.named_parameters():
            if 'AP_path_wb' in name:
                yield param

    def weight_parameters(self): 
        for name, param in self.named_parameters():
            if 'AP_path_alpha' not in name and 'AP_path_wb' not in name:
                yield param

    """ architecture parameters related methods """

    @property
    def redundant_modules(self):
        if self._redundant_modules is None:
            module_list = []
            for m in self.modules():
                if m.__str__().startswith('MixedEdge'):
                    module_list.append(m)
            self._redundant_modules = module_list
        return self._redundant_modules

    def entropy(self, eps=1e-8):
        entropy = 0
        for m in self.redundant_modules:
            module_entropy = m.entropy(eps=eps)
            entropy = module_entropy + entropy
        return entropy

    def init_arch_params(self, init_type='normal', init_ratio=1e-3):
        for param in self.architecture_parameters():
            if init_type == 'normal':
                param.data.normal_(0, init_ratio)
            elif init_type == 'uniform':
                param.data.uniform_(-init_ratio, init_ratio)
            else:
                raise NotImplementedError

    def reset_binary_gates(self): # not used
        for m in self.redundant_modules:
            try:
                m.binarize()
            except AttributeError:
                print(type(m), ' do not support binarize')

    def set_arch_param_grad(self):
        for m in self.redundant_modules:
            try:
                m.set_arch_param_grad()
            except AttributeError:
                print(type(m), ' do not support `set_arch_param_grad()`')

    def rescale_updated_arch_param(self):
        for m in self.redundant_modules:
            try:
                m.rescale_updated_arch_param()
            except AttributeError:
                print(type(m), ' do not support `rescale_updated_arch_param()`')

    """ training related methods """

    def unused_modules_off(self):
        self._unused_modules = []
        for m in self.redundant_modules:
            unused = {}
            if MixedEdge.MODE in ['full', 'two', 'full_v2']:
                involved_index = m.active_index + m.inactive_index
            else:
                involved_index = m.active_index
            for i in range(m.n_choices):
                if i not in involved_index:
                    unused[i] = m.candidate_ops[i]
                    m.candidate_ops[i] = None
            self._unused_modules.append(unused)

    def unused_modules_back(self):
        if self._unused_modules is None:
            return
        for m, unused in zip(self.redundant_modules, self._unused_modules):
            for i in unused:
                m.candidate_ops[i] = unused[i]
        self._unused_modules = None

    def set_chosen_op_active(self):
        for m in self.redundant_modules:
            try:
                m.set_chosen_op_active()
            except AttributeError:
                print(type(m), ' do not support `set_chosen_op_active()`')

    def set_active_via_net(self, net):
        assert isinstance(net, SuperProxylessNASNets)
        for self_m, net_m in zip(self.redundant_modules, net.redundant_modules):
            self_m.active_index = copy.deepcopy(net_m.active_index)
            self_m.inactive_index = copy.deepcopy(net_m.inactive_index)

    def expected_latency(self, latency_model: LatencyEstimator):
        expected_latency = 0

        # classifier 1 & 2
        expected_latency += latency_model.predict(
            'Logit', self.seg_length*64, 512  # in_features, out_features
        )
        expected_latency += latency_model.predict(
            'Logit', 512, self.n_classes
        )
        # blocks
        fsize = self.seg_size
        for block in self.blocks: # num of blocks : 3 (layers)
            shortcut = block.shortcut
            if shortcut is None or shortcut.is_zero_layer():
                idskip = 0
            else:
                idskip = 1

            b_conv = block.conv
            probs_over_ops = b_conv.current_prob_over_ops
            for i, op in enumerate(b_conv.candidate_ops): # Mixed Operator
                if op is None: # or op.is_zero_layer(): # ZeroLayer is skiped
                    continue
                elif op.is_zero_layer():
                    op_latency = latency_model.predict(
                    op.module_str, [fsize, op.in_channels], [fsize, op.in_channels] # Zerolayer has same channels
                )    
                else:
                    op_latency = latency_model.predict(
                        op.module_str[4:], [fsize, op.in_channels], [fsize, op.out_channels],
                        kernel=op.kernel_size, stride=op.stride, dilation=op.dilation
                    )
                expected_latency = expected_latency + op_latency * probs_over_ops[i]
            fsize = math.floor((fsize + 2)//3) # after 1st layer, decrease fsize by pooling layer
        return expected_latency

    def expected_flops(self, x):
        expected_flops = 0
        # blocks
        for block, pool_block in zip(self.blocks, self.pooling_blocks):
            mb_conv = block.conv
            if not isinstance(mb_conv, MixedEdge):
                delta_flop, x = block.get_flops(x)
                expected_flops = expected_flops + delta_flop
                continue

            if block.shortcut is None:
                shortcut_flop = 0
            else:
                shortcut_flop, _ = block.shortcut.get_flops(x)
            expected_flops = expected_flops + shortcut_flop

            probs_over_ops = mb_conv.current_prob_over_ops
            for i, op in enumerate(mb_conv.candidate_ops):
                if op is None or op.is_zero_layer():
                    continue
                op_flops, _ = op.get_flops(x)
                expected_flops = expected_flops + op_flops * probs_over_ops[i]
            x = block(x)
            x = pool_block(x)
        # classifier
        # x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        delta_flop, x = self.classifier1.get_flops(x)
        delta_flop, x = self.classifier2.get_flops(x)
        expected_flops = expected_flops + delta_flop
        return expected_flops

    def convert_to_normal_net(self):
        queue = Queue()
        queue.put(self)
        while not queue.empty():
            module = queue.get()
            for m in module._modules:
                child = module._modules[m]
                if child is None:
                    continue
                if child.__str__().startswith('MixedEdge'):
                    module._modules[m] = child.chosen_op
                else:
                    queue.put(child)
        return ProxylessNASNets(list(self.blocks), list(self.pooling_blocks), self.classifier1, self.classifier2)