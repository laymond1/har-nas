import torch
from modules.layers import *
from modules.mix_op import *
from operator_pt.operators import *
from torch.utils.mobile_optimizer import optimize_for_mobile

import sys
import os

# sys.path.append(os.path.join(os.getcwd(), 'Code', 'Search'))

class Zero(nn.Module):
    def forward(self, x):
        x = x
        return x

def operator_desc(op):
    op_type=op.module_str[4:]
    in_C=op.in_channels
    out_C=op.out_channels
    k=op.kernel_size
    s=op.stride
    d=op.dilation
    return op_type, in_C, out_C, k, s, d

def main(dataset):
    # Load Dataset Provider
    if dataset == 'ucihar':
        from data_providers.ucihar import UCIHARDataProvider
        data_provider = UCIHARDataProvider()
    elif dataset == 'unimib':
        from data_providers.unimib import UniMiBDataProvider
        data_provider = UniMiBDataProvider()
    elif dataset == 'wisdm2019':
        from data_providers.wisdm2019 import WISDM2019DataProvider
        data_provider = WISDM2019DataProvider()
    elif dataset == 'wisdm':
        from data_providers.wisdm import WISDMDataProvider
        data_provider = WISDMDataProvider()
    else:
        raise ValueError

    # Search Space
    conv_candidates = [
        'Zero',
        '3x1_Conv', '3x1_DilatedConv', '3x1_DepthConv', '3x1_DilatedDepthConv',
        '5x1_Conv', '5x1_DilatedConv', '5x1_DepthConv', '5x1_DilatedDepthConv',
        '7x1_Conv', '7x1_DilatedConv', '7x1_DepthConv', '7x1_DilatedDepthConv',
        '9x1_Conv', '9x1_DilatedConv', '9x1_DepthConv', '9x1_DilatedDepthConv',
    ]

    length_list = [data_provider.data_length]
    length = data_provider.data_length
    ops_list1 = build_candidate_ops(conv_candidates, data_provider.data_shape[0], 64, 1, None)
    length = math.floor((length+2)//3)
    length_list.append(length)
    ops_list2 = build_candidate_ops(conv_candidates, 64, 64, 1, None)
    length = math.floor((length+2)//3)
    length_list.append(length)
    ops_list3 = build_candidate_ops(conv_candidates, 64, 64, 1, None)

    ops_block_list = [ops_list1, ops_list2, ops_list3]
    count = 0
    for ops_list, length in zip(ops_block_list, length_list):
        for op in ops_list:
            example = torch.randn((1, op.in_channels, 128))
            if 'Zero' in op.module_str:
                if op.in_channels <= 20:
                    continue
                op_type = 'Zero'
                in_C = op.in_channels
                op = Zero()
                op.cpu()
                op.eval()
                mobile_op = torch.jit.script(op, example)
                mobile_op.save(
                    "./operator_pt/{dataset}/{op_type}-{length}x{in_C}-{length}x{out_C}.pt".format(
                    dataset=dataset, op_type=op_type, length=length, in_C=in_C, out_C=in_C
                    ))
            else:
                op_type, in_C, out_C, k, s, d = operator_desc(op)
                op = ConvOp(in_C, out_C, k, s, d)
                op.cpu()
                op.eval()
                mobile_op = torch.jit.script(op, example)
                # mobile_op = optimize_for_mobile(mobile_op)
                mobile_op.save(
                    "./operator_pt/{dataset}/{op_type}-{length}x{in_C}-{length}x{out_C}-{k}-{s}-{d}.pt".format(
                    dataset=dataset, op_type=op_type, length=length, in_C=in_C, out_C=out_C, k=k, s=s, d=d
                    ))
        length = math.floor((length+2)//3)
        print(count)
        print(length)
        count += 1

    # Classifier 1 & 2
    example = torch.randn((1, length*64))
    classifier1 = Classifier(length*64, 512, activation=True)
    classifier1.cpu()
    classifier1.eval()
    mobile_classifier1 = torch.jit.script(classifier1, example)
    mobile_classifier1.save(
        "./operator_pt/{dataset}/Logit-{in_f}-{out_f}.pt".format(
        dataset=dataset, in_f=classifier1.in_features, out_f=classifier1.out_features
        ))

    example = torch.randn((1, 512))
    classifier2 = Classifier(512, data_provider.n_classes, dropout_rate=0.05)
    classifier2.cpu()
    classifier2.eval()
    mobile_classifier2 = torch.jit.script(classifier2, example)
    mobile_classifier2.save( 
        "./operator_pt/{dataset}/Logit-{in_f}-{out_f}.pt".format(
        dataset=dataset, in_f=classifier2.in_features, out_f=classifier2.out_features
        ))


if __name__ == "__main__":
    main(dataset='wisdm')