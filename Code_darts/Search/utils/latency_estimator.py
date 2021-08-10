# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

import yaml
import os
import sys
import csv
try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve


def download_url(url, model_dir='~/.torch/proxyless_nas', overwrite=False):
    target_dir = url.split('//')[-1]
    target_dir = os.path.dirname(target_dir)
    model_dir = os.path.expanduser(model_dir)
    model_dir = os.path.join(model_dir, target_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file) or overwrite:
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return cached_file


class LatencyEstimator(object):
    def __init__(self, dataset, target_hardware):
        LUT = os.path.join('LUT', target_hardware, dataset)

        with open(LUT + '.txt', 'r', encoding='utf-8') as f:
            rdr = csv.reader(f)
            self._latency = {}
            for line in rdr:
                self._latency[line[0]] = line[1]

    @staticmethod
    def repr_shape(shape):
        if isinstance(shape, (list, tuple)):
            return 'x'.join(str(_) for _ in shape)
        elif isinstance(shape, str):
            return shape
        elif isinstance(shape, int):
            return str(shape)
        else:
            return TypeError

    def predict(self, ltype: str, _input, output, kernel=None, stride=None, dilation=None, ):
        """
        : param ltype:

        """
        infos = [ltype, self.repr_shape(_input), self.repr_shape(output), ]

        if ltype in ('Conv','DilatedConv'):
            assert None not in (kernel, stride, dilation)
            infos += ['kernel:%d' % kernel, 'stride:%d' % stride, 'dilation:%d' % dilation]
        key = '-'.join(infos)
        return float(self._latency[key]) * 1e-3

if __name__ == "__main__":
    # LatnecyLoss(channels=[6,32], steps=[2,2], strides=[1,1], dilations=1, input_size=128)
    est = LatencyEstimator()
    s = est.predict('Conv', _input=(128, 6), output=(128, 64), kernel=3, stride=1, dilation=1)
    logits = est.predict('Logits', _input=(320), output=(512))
    # identity = est.predict('Identity', _input=(128, 32), output=(128,32))
    print(s)
    print(logits)
    # print(identity)


# class LatencyEstimator(object):
#     def __init__(self, url='https://file.lzhu.me/projects/proxylessNAS/LatencyTools/mobile_trim.yaml'):
#         fname = download_url(url, overwrite=True)

#         with open(fname, 'r') as fp:
#             self.lut = yaml.load(fp)

#     @staticmethod
#     def repr_shape(shape):
#         if isinstance(shape, (list, tuple)):
#             return 'x'.join(str(_) for _ in shape)
#         elif isinstance(shape, str):
#             return shape
#         else:
#             return TypeError

#     def predict(self, ltype: str, _input, output, expand=None, kernel=None, stride=None, idskip=None, ):
#         """
#         :param ltype:
#             Layer type must be one of the followings
#                 1. `Conv`: The initial 3x3 conv with stride 2.
#                 2. `Conv_1`: The upsample 1x1 conv that increases num_filters by 4 times.
#                 3. `Logits`: All operations after `Conv_1`.
#                 4. `expanded_conv`: MobileInvertedResidual
#         :param _input: input shape (h, w, #channels)
#         :param output: output shape (h, w, #channels)
#         :param expand: expansion ratio
#         :param kernel: kernel size
#         :param stride:
#         :param idskip: indicate whether has the residual connection
#         """
#         infos = [ltype, 'input:%s' % self.repr_shape(_input), 'output:%s' % self.repr_shape(output), ]

#         if ltype in ('expanded_conv',):
#             assert None not in (expand, kernel, stride, idskip)
#             infos += ['expand:%d' % expand, 'kernel:%d' % kernel, 'stride:%d' % stride, 'idskip:%d' % idskip]
#         key = '-'.join(infos)
#         return self.lut[key]['mean']


# if __name__ == '__main__':
#     est = LatencyEstimator()
#     s = est.predict('expanded_conv', _input=(112, 112, 16), output=(56, 56, 24), expand=3, kernel=5, stride=2, idskip=0)
#     print(s)
