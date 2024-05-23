import torch
from types import SimpleNamespace
from pprint import pprint
from exp.exp_informer import Exp_Informer

class ExpSetting:
    def __init__(self, **kwargs):
        # 定义默认参数
        default_args = {
            'model': 'informer',
            'data': 'ETTh1',
            'root_path': './data/ETT/',
            'data_path': 'ETTh1.csv',
            'features': 'M',
            'target': 'OT',
            'freq': 'h',
            'checkpoints': './checkpoints/',
            'seq_len': 96,
            'label_len': 48,
            'pred_len': 24,
            'enc_in': 7,     # encoder input size
            'dec_in': 7,     # decoder input size
            'c_out': 7,      # output size
            'd_model': 512,
            'n_heads': 8,
            'e_layers': 2,
            'd_layers': 1,
            's_layers': '3,2,1',
            'd_ff': 2048,
            'factor': 5,
            'padding': 0,
            'distil': True,
            'dropout': 0.05,
            'attn': 'prob',
            'embed': 'timeF',
            'activation': 'gelu',
            'output_attention': False,
            'do_predict': False,
            'mix': True,
            'cols': None,
            'num_workers': 0,
            'itr': 1,        # 没有理解这里原本设置成2的意义，我改成了1
            'train_epochs': 6,
            'batch_size': 32,
            'patience': 3,
            'learning_rate': 0.0001,
            'des': 'test',
            'loss': 'mse',
            'lradj': 'type1',
            'use_amp': False,
            'inverse': False,
            'use_gpu': True,
            'gpu': 0,
            'use_multi_gpu': False,
            'devices': '0,1,2,3',
        }

        # 更新默认参数字典
        default_args.update(kwargs)

        # 使用命名空间对象来存储参数，最终实际参数也是在self.args中
        self.args = SimpleNamespace(**default_args)
        self.args.use_gpu = True if torch.cuda.is_available() and self.args.use_gpu else False
        if self.args.use_gpu and self.args.use_multi_gpu:
            self.args.devices = self.args.devices.replace(' ','')
            device_ids = self.args.devices.split(',')
            self.args.device_ids = [int(id_) for id_ in device_ids]
            self.args.gpu = self.args.device_ids[0]
        
        

        data_parser = {
            'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
            'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
            'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
            'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
            'WTH':{'data':'WTH.csv','T':'WetBulbCelsius','M':[12,12,12],'S':[1,1,1],'MS':[12,12,1]},
            'ECL':{'data':'ECL.csv','T':'MT_320','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
            'Solar':{'data':'solar_AL.csv','T':'POWER_136','M':[137,137,137],'S':[1,1,1],'MS':[137,137,1]},
        }
        if self.args.data in data_parser.keys():
            data_info = data_parser[self.args.data]
            self.args.data_path = data_info['data']
            self.args.target = data_info['T']
            self.args.enc_in, self.args.dec_in, self.args.c_out = data_info[self.args.features]

        self.args.s_layers = [int(s_l) for s_l in self.args.s_layers.replace(' ','').split(',')]
        self.args.detail_freq = self.args.freq
        self.args.freq = self.args.freq[-1:]
        
        pprint('Args in experiment:')
        pprint(self.args)
        
    # 示例方法：获取参数
    def get_param(self, param):
        return getattr(self.args, param, None)

    # 示例方法：设置参数
    def set_param(self, param, value):
        if hasattr(self.args, param):
            setattr(self.args, param, value)
        else:
            raise KeyError(f"Parameter '{param}' is not a valid parameter")

    def train(self, **kwargs):
        for arg in kwargs:
            self.set_param(arg, kwargs[arg])
        args = self.args  
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
                args.embed, args.distil, args.mix, args.des, 1)
        exp = Exp_Informer(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        torch.cuda.empty_cache()
        
    def test(self, **kwargs):
        for arg in kwargs:
            self.set_param(arg, kwargs[arg])
        args = self.args
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
                args.embed, args.distil, args.mix, args.des, 1)
        exp = Exp_Informer(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()
        
    def predict(self, **kwargs):
        self.set_param('do_predict', True)
        for arg in kwargs:
            self.set_param(arg, kwargs[arg])
        args = self.args
        setting = '{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(args.model, args.data, args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
                args.embed, args.distil, args.mix, args.des, 1)
        exp = Exp_Informer(args)
        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)
        torch.cuda.empty_cache()
