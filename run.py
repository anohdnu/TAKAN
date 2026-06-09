from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
import argparse
import torch
import numpy as np
import yaml
import json
import random
from trainer import Trainer

def create_args():
    
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()

    # Standard Args
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0],
                         help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--log_dir', type=str, default="outputs/out",
                         help="Save experiments results in dir for future plotting!")
    parser.add_argument('--learner_type', type=str, default='default', help="The type (filename) of learner")
    parser.add_argument('--learner_name', type=str, default='NormalNN', help="The class name of learner")
    parser.add_argument('--debug_mode', type=int, default=0, metavar='N',
                        help="activate learner specific settings for debug_mode")
    parser.add_argument('--repeat', type=int, default=1, help="Repeat the experiment N times")
    parser.add_argument('--overwrite', type=int, default=0, metavar='N', help='Train regardless of whether saved model exists')

    # CL Args          
    parser.add_argument('--oracle_flag', default=False, action='store_true', help='Upper bound for oracle')
    parser.add_argument('--upper_bound_flag', default=False, action='store_true', help='Upper bound')
    parser.add_argument('--memory', type=int, default=0, help="size of memory for replay")
    parser.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
    parser.add_argument('--DW', default=False, action='store_true', help='dataset balancing')
    parser.add_argument('--prompt_param', nargs="+", type=float, default=[1, 1, 1],
                         help="e prompt pool size, e prompt length, g prompt length")

    # Config Arg
    parser.add_argument('--config', type=str, default="configs/config.yaml",
                         help="yaml experiment config input")
    parser.add_argument(
        '--ood_scores',
        nargs='+',
        type=str,
        default=['all'],
        choices=['msp', 'prompt_sim_msp'],
        help='One or more OOD scores to evaluate in the same run'
    )

    return parser

def get_args(argv):
    parser = create_args()
    args = parser.parse_args(argv)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config.update(vars(args))

    ood_scores = config.get('ood_scores', ['msp'])
    if 'all' in ood_scores:
        config['ood_scores'] = ['msp', 'prompt_sim_msp']
    else:
        config['ood_scores'] = [s.lower() for s in ood_scores]
    return argparse.Namespace(**config)
    
# want to save everything printed to outfile
class Logger(object):
    def __init__(self, name):
        self.terminal = sys.stdout
        self.log = open(name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()

if __name__ == '__main__':
    args = get_args(sys.argv[1:])

    # determinstic backend
    torch.backends.cudnn.deterministic=True

    # duplicate output stream to output file
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    log_out = args.log_dir + '/output.log'
    sys.stdout = Logger(log_out)

    # save args
    with open(args.log_dir + '/args.yaml', 'w') as yaml_file:
        yaml.dump(vars(args), yaml_file, default_flow_style=False)
    
    metric_keys = ['acc', 'time', 'auroc', 'fpr95']
    save_keys = ['global', 'pt', 'pt-local']
    global_only = ['time']
    
    avg_metrics = {
        'acc': {skey: [] for skey in save_keys},
        'time': {skey: [] for skey in save_keys},
        'auroc': {score: {skey: [] for skey in save_keys} for score in args.ood_scores},
        'fpr95': {score: {skey: [] for skey in save_keys} for score in args.ood_scores},
    }

    # load results
    if args.overwrite:
        start_r = 0
    else:
        try:
            for mkey in metric_keys: 
                for skey in save_keys:
                    if (not (mkey in global_only)) or (skey == 'global'):
                        save_file = args.log_dir+'/results-'+mkey+'/'+skey+'.yaml'
                        if os.path.exists(save_file):
                            with open(save_file, 'r') as yaml_file:
                                yaml_result = yaml.safe_load(yaml_file)
                                avg_metrics[mkey][skey] = np.asarray(yaml_result['history'])

            # next repeat needed
            start_r = avg_metrics[metric_keys[0]][save_keys[0]].shape[-1]

            # extend if more repeats left
            if start_r < args.repeat:
                max_task = avg_metrics['acc']['global'].shape[0]
                for mkey in metric_keys: 
                    avg_metrics[mkey]['global'] = np.append(avg_metrics[mkey]['global'], np.zeros((max_task,args.repeat-start_r)), axis=-1)
                    if (not (mkey in global_only)):
                        avg_metrics[mkey]['pt'] = np.append(avg_metrics[mkey]['pt'], np.zeros((max_task,max_task,args.repeat-start_r)), axis=-1)
                        avg_metrics[mkey]['pt-local'] = np.append(avg_metrics[mkey]['pt-local'], np.zeros((max_task,max_task,args.repeat-start_r)), axis=-1)

        except:
            start_r = 0
    
    for r in range(start_r, args.repeat):

        print('************************************')
        print('* STARTING TRIAL ' + str(r+1))
        print('************************************')

        # set random seeds
        seed = r
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # set up a trainer
        trainer = Trainer(args, seed, metric_keys, save_keys)

        # init total run metrics storage
        max_task = trainer.max_task
        if r == 0:
            # acc
            avg_metrics['acc']['global'] = np.zeros((max_task, args.repeat))
            avg_metrics['acc']['pt'] = np.zeros((max_task, max_task, args.repeat))
            avg_metrics['acc']['pt-local'] = np.zeros((max_task, max_task, args.repeat))
        
            # time
            avg_metrics['time']['global'] = np.zeros((max_task, args.repeat))
        
            # auroc / fpr95 per score
            for score in args.ood_scores:
                avg_metrics['auroc'][score]['global'] = np.zeros((max_task, args.repeat))
                avg_metrics['auroc'][score]['pt'] = np.zeros((max_task, max_task, args.repeat))
                avg_metrics['auroc'][score]['pt-local'] = np.zeros((max_task, max_task, args.repeat))
        
                avg_metrics['fpr95'][score]['global'] = np.zeros((max_task, args.repeat))
                avg_metrics['fpr95'][score]['pt'] = np.zeros((max_task, max_task, args.repeat))
                avg_metrics['fpr95'][score]['pt-local'] = np.zeros((max_task, max_task, args.repeat))
        
                
        # train model
        avg_metrics = trainer.train(avg_metrics)  

        # evaluate model
        avg_metrics = trainer.evaluate(avg_metrics)    

        # save acc and time
        for mkey in ['acc', 'time']:
            m_dir = args.log_dir + '/results-' + mkey + '/'
            if not os.path.exists(m_dir):
                os.makedirs(m_dir)
        
            for skey in save_keys:
                if (mkey not in global_only) or (skey == 'global'):
                    save_file = m_dir + skey + '.yaml'
                    result = avg_metrics[mkey][skey]
                    yaml_results = {}
        
                    if len(result.shape) > 2:
                        yaml_results['mean'] = result[:, :, :r+1].mean(axis=2).tolist()
                        if r > 1:
                            yaml_results['std'] = result[:, :, :r+1].std(axis=2).tolist()
                        yaml_results['history'] = result[:, :, :r+1].tolist()
                    else:
                        yaml_results['mean'] = result[:, :r+1].mean(axis=1).tolist()
                        if r > 1:
                            yaml_results['std'] = result[:, :r+1].std(axis=1).tolist()
                        yaml_results['history'] = result[:, :r+1].tolist()
        
                    with open(save_file, 'w') as yaml_file:
                        yaml.dump(yaml_results, yaml_file, default_flow_style=False)
        
        # save auroc and fpr95 per score
        for mkey in ['auroc', 'fpr95']:
            for score in args.ood_scores:
                m_dir = args.log_dir + f'/results-{mkey}-{score}/'
                if not os.path.exists(m_dir):
                    os.makedirs(m_dir)
        
                for skey in save_keys:
                    save_file = m_dir + skey + '.yaml'
                    result = avg_metrics[mkey][score][skey]
                    yaml_results = {}
        
                    if len(result.shape) > 2:
                        yaml_results['mean'] = result[:, :, :r+1].mean(axis=2).tolist()
                        if r > 1:
                            yaml_results['std'] = result[:, :, :r+1].std(axis=2).tolist()
                        yaml_results['history'] = result[:, :, :r+1].tolist()
                    else:
                        yaml_results['mean'] = result[:, :r+1].mean(axis=1).tolist()
                        if r > 1:
                            yaml_results['std'] = result[:, :r+1].std(axis=1).tolist()
                        yaml_results['history'] = result[:, :r+1].tolist()
        
                    with open(save_file, 'w') as yaml_file:
                        yaml.dump(yaml_results, yaml_file, default_flow_style=False)


        
        print('===Summary of experiment repeats:', r+1, '/', args.repeat, '===')
        
        print('acc | mean:', avg_metrics['acc']['global'][-1, :r+1].mean(),
              'std:', avg_metrics['acc']['global'][-1, :r+1].std())
        
        print('time | mean:', avg_metrics['time']['global'][-1, :r+1].mean(),
              'std:', avg_metrics['time']['global'][-1, :r+1].std())
        
        for score in args.ood_scores:
            print(f'auroc ({score}) | mean:',
                  avg_metrics['auroc'][score]['global'][-1, :r+1].mean(),
                  'std:',
                  avg_metrics['auroc'][score]['global'][-1, :r+1].std())
        
            print(f'fpr95 ({score}) | mean:',
                  avg_metrics['fpr95'][score]['global'][-1, :r+1].mean(),
                  'std:',
                  avg_metrics['fpr95'][score]['global'][-1, :r+1].std())
    
    


