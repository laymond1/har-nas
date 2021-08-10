# ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware
# Han Cai, Ligeng Zhu, Song Han
# International Conference on Learning Representations (ICLR), 2019.

from models.normal_nets.proxyless_nets import ProxylessNASNets
from run_manager import RunConfig


def get_net_by_name(name):
    if name == ProxylessNASNets.__name__:
        return ProxylessNASNets
    else:
        raise ValueError('unrecognized type of network: %s' % name)


class UCIHARRunConfig(RunConfig):

    def __init__(self, n_epochs=150, init_lr=5e-4, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='ucihar', train_batch_size=256, test_batch_size=256, valid_size=None, target_hardware=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0., no_decay_keys='bn',
                 model_init='he_fout', init_div_groups=False, validation_frequency=1, print_frequency=10,
                 n_worker=0, **kwargs):
        super(UCIHARRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size, target_hardware,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            model_init, init_div_groups, validation_frequency, print_frequency
        )

        self.n_worker = n_worker

        print(kwargs.keys())

    @property
    def data_config(self):
        return {
            'train_batch_size': self.train_batch_size,
            'test_batch_size': self.test_batch_size,
            'valid_size': self.valid_size,
            'n_worker': self.n_worker,
        }

class UniMiBRunConfig(RunConfig):

    def __init__(self, n_epochs=150, init_lr=5e-4, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='unimib', train_batch_size=256, test_batch_size=256, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0., no_decay_keys='bn',
                 model_init='he_fout', init_div_groups=False, validation_frequency=1, print_frequency=10,
                 n_worker=0, **kwargs):
        super(UniMiBRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            model_init, init_div_groups, validation_frequency, print_frequency
        )

        self.n_worker = n_worker

        print(kwargs.keys())

    @property
    def data_config(self):
        return {
            'train_batch_size': self.train_batch_size,
            'test_batch_size': self.test_batch_size,
            'valid_size': self.valid_size,
            'n_worker': self.n_worker,
        }

class WISDM19RunConfig(RunConfig):

    def __init__(self, n_epochs=150, init_lr=5e-4, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='wisdm2019', train_batch_size=256, test_batch_size=256, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0., no_decay_keys='bn',
                 model_init='he_fout', init_div_groups=False, validation_frequency=1, print_frequency=10,
                 n_worker=0, **kwargs):
        super(WISDM19RunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            model_init, init_div_groups, validation_frequency, print_frequency
        )

        self.n_worker = n_worker

        print(kwargs.keys())

    @property
    def data_config(self):
        return {
            'train_batch_size': self.train_batch_size,
            'test_batch_size': self.test_batch_size,
            'valid_size': self.valid_size,
            'n_worker': self.n_worker,
        }

class WISDMRunConfig(RunConfig):

    def __init__(self, n_epochs=150, init_lr=5e-4, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='wisdm2019', train_batch_size=256, test_batch_size=256, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0., no_decay_keys='bn',
                 model_init='he_fout', init_div_groups=False, validation_frequency=1, print_frequency=10,
                 n_worker=0, **kwargs):
        super(WISDMRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            model_init, init_div_groups, validation_frequency, print_frequency
        )

        self.n_worker = n_worker

        print(kwargs.keys())

    @property
    def data_config(self):
        return {
            'train_batch_size': self.train_batch_size,
            'test_batch_size': self.test_batch_size,
            'valid_size': self.valid_size,
            'n_worker': self.n_worker,
        }