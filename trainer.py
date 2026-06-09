import os
import sys
import argparse
import torch
import numpy as np
import random
from random import shuffle
from collections import OrderedDict
import dataloaders
from dataloaders.utils import *
from torch.utils.data import DataLoader
import learners
from ood_metrics import compute_all_metrics
import torch.nn.functional as F
import math, json

class Trainer:

    def __init__(self, args, seed, metric_keys, save_keys):

        # process inputs
        self.seed = seed
        self.metric_keys = metric_keys
        self.save_keys = save_keys
        self.log_dir = args.log_dir
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.ood_scores = args.ood_scores
        
        # model load directory
        self.model_top_dir = args.log_dir

        # select dataset
        self.grayscale_vis = False
        self.top_k = 1
        if args.dataset == 'CIFAR10':
            Dataset = dataloaders.iCIFAR10
            num_classes = 10
            self.dataset_size = [32,32,3]
        elif args.dataset == 'CIFAR100':
            Dataset = dataloaders.iCIFAR100
            num_classes = 100
            self.dataset_size = [32,32,3]
        elif args.dataset == 'ImageNet_R':
            Dataset = dataloaders.iIMAGENET_R
            num_classes = 200
            self.dataset_size = [224,224,3]
            self.top_k = 1
        elif args.dataset == 'DomainNet':
            Dataset = dataloaders.iDOMAIN_NET
            num_classes = 345
            self.dataset_size = [224,224,3]
            self.top_k = 1
        elif args.dataset == 'CUB':
            Dataset = dataloaders.iCUB
            num_classes = 200
            self.dataset_size = [224,224,3]
            self.top_k = 1
        else:
            raise ValueError('Dataset not implemented!')

        # upper bound flag
        if args.upper_bound_flag:
            args.other_split_size = num_classes
            args.first_split_size = num_classes

        # load tasks

        class_order = np.arange(num_classes).tolist()
        class_order_logits = np.arange(num_classes).tolist()
        if 'dil' not in args:
            args.dil = False
        self.dil = args.dil
        if self.dil:
            self.tasks = []
            self.tasks_logits = []
            for i in range(args.domain_num):
                self.tasks.append(class_order)
                self.tasks_logits.append(class_order_logits)
            self.num_tasks = len(self.tasks)
        else:
            if self.seed > 0 and args.rand_split:
                print('=============================================')
                print('Shuffling....')
                print('pre-shuffle:' + str(class_order))
                random.seed(self.seed)
                random.shuffle(class_order)
                print('post-shuffle:' + str(class_order))
                print('=============================================')
            self.tasks = []
            self.tasks_logits = []
            p = 0
            while p < num_classes and (args.max_task == -1 or len(self.tasks) < args.max_task):
                inc = args.other_split_size if p > 0 else args.first_split_size
                self.tasks.append(class_order[p:p+inc])
                self.tasks_logits.append(class_order_logits[p:p+inc])
                p += inc
            self.num_tasks = len(self.tasks)
        self.task_names = [str(i+1) for i in range(self.num_tasks)]
        # number of tasks to perform
        if args.max_task > 0:
            self.max_task = min(args.max_task, len(self.task_names))
        else:
            self.max_task = len(self.task_names)

        # datasets and dataloaders
        k = 1 # number of transforms per image
        if args.model_name.startswith('vit'):
            resize_imnet = True
        else:
            resize_imnet = False
        train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug, resize_imnet=resize_imnet)
        test_transform  = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=args.train_aug, resize_imnet=resize_imnet)
        self.train_dataset = Dataset(args.dataroot, train=True, lab = True, tasks=self.tasks,
                            download_flag=True, transform=train_transform, 
                            seed=self.seed, rand_split=args.rand_split, validation=args.validation)
        self.test_dataset  = Dataset(args.dataroot, train=False, tasks=self.tasks,
                                download_flag=False, transform=test_transform, 
                                seed=self.seed, rand_split=args.rand_split, validation=args.validation)

        # for oracle
        self.oracle_flag = args.oracle_flag
        self.add_dim = 0

        # Prepare the self.learner (model)
        self.learner_config = {'num_classes': num_classes,
                        'lr': args.lr,
                        'debug_mode': args.debug_mode == 1,
                        'momentum': args.momentum,
                        'weight_decay': args.weight_decay,
                        'schedule': args.schedule,
                        'schedule_type': args.schedule_type,
                        'model_type': args.model_type,
                        'model_name': args.model_name,
                        'optimizer': args.optimizer,
                        'gpuid': args.gpuid,
                        'memory': args.memory,
                        'temp': args.temp,
                        'out_dim': num_classes,
                        'overwrite': args.overwrite == 1,
                        'DW': args.DW,
                        'batch_size': args.batch_size,
                        'upper_bound_flag': args.upper_bound_flag,
                        'tasks': self.tasks_logits,
                        'top_k': self.top_k,
                        'prompt_param':[self.num_tasks,args.prompt_param],
                        'dil':self.dil,
                        'clf_type': args.clf_type,      
                        }
        self.learner_type, self.learner_name = args.learner_type, args.learner_name
        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)


    
    def task_eval(self, t_index, local=False, task='acc'):

        val_name = self.task_names[t_index]
        print('validation split name:', val_name)
        
        # eval
        self.test_dataset.load_dataset(t_index, train=True)
        test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
        if local:
            return self.learner.validation(test_loader, task_in = self.tasks_logits[t_index], task_metric=task)
        else:
            return self.learner.validation(test_loader, task_metric=task)
            
            
    def _build_task_slices(self, task_in=None):
        """
        Returns list of class-index lists, one list per task, in the CURRENT logit space.
    
        If task_in is None:
            logit space is [0 ... valid_out_dim-1], grouped by learned tasks.
        If task_in is not None:
            logit space has already been restricted to task_in only, so there is only one task.
        """
        if task_in is not None:
            return [list(range(len(task_in)))]
    
        slices = []
        start = 0
        remaining = self.learner.valid_out_dim
    
        for task_classes in self.tasks_logits:
            task_len = len(task_classes)
            if remaining <= 0:
                break
    
            use_len = min(task_len, remaining)
            slices.append(list(range(start, start + use_len)))
            start += use_len
            remaining -= use_len
    
            if remaining <= 0:
                break
    
        return slices
        
        
    @torch.no_grad()
    def _score_from_cached_outputs(
        self,
        logits,
        score_name,
        prompt_sim_mean=None,
        prompt_sim2=None,
        lid_values=None,
        class_weight_norms=None,
        task_slices=None,
    ):
        """
        Return confidence where larger => more ID-like

        logits:            torch.Tensor [N, C]
        prompt_sim_mean:   torch.Tensor [N] or None
        lid_values:        optional torch.Tensor [N]
        """
        score_name = score_name.lower()

        if score_name == "msp":
            prob = torch.softmax(logits, dim=1)
            conf = prob.max(dim=1).values
      
        elif score_name == "prompt_sim_msp":
            if prompt_sim_mean is None:
                raise ValueError("prompt_sim_mean must be provided for prompt_sim_msp")
            pconf = (prompt_sim_mean + 1.0) / 2.0
            prob = torch.softmax(logits, dim=1)
            conf = prob.max(dim=1).values * pconf
    
        else:
            raise ValueError(f"Unknown ood score: {score_name}")

        return conf

    
    @torch.no_grad()
    def _collect_ood_outputs(self, loader, task_in=None):
        """
        Collect outputs once, then reuse them for all OOD scores.
        Returns:
            logits_all: torch.Tensor [N, C]
            prompt_sim_mean_all: torch.Tensor [N] or None
            prompt_sim2_all: torch.Tensor [N] or None
            class_weight_norms: torch.Tensor [C] or None
        """
        self.learner.model.eval()

        logits_list = []
        prompt_sim_mean_list = []
        prompt_sim2_list = []

        model = self.learner.model
        if hasattr(model, "module"):
            model = model.module
            

        last = model.last
        if hasattr(last, "weight"):   # standard linear classifier
            class_weight_norms = last.weight.detach().norm(dim=1).cpu()
        elif hasattr(last, "spline_linear") and hasattr(last.spline_linear, "weight"):
            # fallback for KAC-style classifier
            class_weight_norms = last.spline_linear.weight.detach().norm(dim=1).cpu()
        else:
            class_weight_norms = None
        
        # IMPORTANT: keep same class slicing as logits
        if class_weight_norms is not None:
            class_weight_norms = class_weight_norms[:self.learner.valid_out_dim]
        
            if task_in is not None:
                class_weight_norms = class_weight_norms[task_in]
                
            
        for x, y, task in loader:
            if self.learner.gpu:
                x = x.cuda(non_blocking=True)

            logits_full, prompt_sims = model.forward(
                x, train=False, return_prompt_sims=True
            )
            logits = logits_full[:, :self.learner.valid_out_dim]

            if task_in is not None:
                logits = logits[:, task_in]

            logits_list.append(logits.detach().cpu())


            # compute cached prompt-based scores once per batch
            per_layer_maxsim = []
            per_layer_topkmean = []
            
            if prompt_sims is not None:
                for l, info in prompt_sims.items():
                    if info is None:
                        continue
            
                    # prompt_sim
                    per_layer_maxsim.append(info["max_sim"])   # [B]
            
                    # prompt_sim2
                    cos_sim = info["cos_sim"]                  # [B, P]
                    k = cos_sim.size(1)              
                    topk_vals = torch.topk(cos_sim, k=k, dim=1).values   # [B, k]
                    layer_conf2 = topk_vals.mean(dim=1)                  # [B]
                    per_layer_topkmean.append(layer_conf2)
            
            if len(per_layer_maxsim) > 0:
                prompt_sim_mean = torch.stack(per_layer_maxsim, dim=0).mean(dim=0)
                prompt_sim_mean_list.append(prompt_sim_mean.detach().cpu())
            else:
                prompt_sim_mean_list.append(None)
            
            if len(per_layer_topkmean) > 0:
                prompt_sim2 = torch.stack(per_layer_topkmean, dim=0).mean(dim=0)
                prompt_sim2_list.append(prompt_sim2.detach().cpu())
            else:
                prompt_sim2_list.append(None)


        
        if len(logits_list) == 0:
            return (
                torch.empty((0, 0), dtype=torch.float32),
                None,
                None,
                class_weight_norms,
            )
            
        logits_all = torch.cat(logits_list, dim=0)

        if all(x is None for x in prompt_sim_mean_list):
            prompt_sim_mean_all = None
        else:
            prompt_sim_mean_all = torch.cat(
                [x for x in prompt_sim_mean_list if x is not None], dim=0
            )
            
        if all(x is None for x in prompt_sim2_list):
            prompt_sim2_all = None
        else:
            prompt_sim2_all = torch.cat(
                [x for x in prompt_sim2_list if x is not None], dim=0
            )
            
        return logits_all, prompt_sim_mean_all, prompt_sim2_all, class_weight_norms

    @torch.no_grad()
    def _compute_scores_from_cached_outputs(
        self,
        logits,
        prompt_sim_mean=None,
        prompt_sim2=None,
        lid_values=None,
        class_weight_norms=None,
        task_slices=None,
    ):
        """
        logits: torch.Tensor [N, C] on CPU
        prompt_sim_mean: torch.Tensor [N] on CPU or None

        returns:
            dict {score_name: np.ndarray [N]}
        """
        score_dict = {}

        logits = logits.float()
        if prompt_sim_mean is not None:
            prompt_sim_mean = prompt_sim_mean.float()

        for score_name in self.ood_scores:
            conf = self._score_from_cached_outputs(
                logits=logits,
                score_name=score_name,
                prompt_sim_mean=prompt_sim_mean,
                prompt_sim2=prompt_sim2,
                lid_values=lid_values,
                class_weight_norms=class_weight_norms,
                task_slices=task_slices,
            )
            score_dict[score_name] = conf.detach().cpu().numpy()

        return score_dict

    
    def _collect_all_scores_for_pair(self, id_task_idx, ood_task_idx, local=False):
        """
        Collect outputs once for ID and OOD, then compute all requested scores.
        """
        self.test_dataset.load_dataset(id_task_idx, train=True)
        id_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.workers
        )

        self.train_dataset.load_dataset(ood_task_idx, train=True)
        ood_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.workers
        )

        task_in = self.tasks_logits[id_task_idx] if local else None
        task_slices = self._build_task_slices(task_in=task_in)


        logits_id, promptsim_id, promptsim2_id, class_weight_norms_id = self._collect_ood_outputs(
            id_loader, task_in=task_in
        )
        logits_ood, promptsim_ood, promptsim2_ood, class_weight_norms_ood = self._collect_ood_outputs(
            ood_loader, task_in=task_in
        )        
        
        score_id = self._compute_scores_from_cached_outputs(
            logits=logits_id,
            prompt_sim_mean=promptsim_id,
            prompt_sim2=promptsim2_id,
            class_weight_norms=class_weight_norms_id,
            task_slices=task_slices,
        )
        score_ood = self._compute_scores_from_cached_outputs(
            logits=logits_ood,
            prompt_sim_mean=promptsim_ood,
            prompt_sim2=promptsim2_ood,
            class_weight_norms=class_weight_norms_ood,
            task_slices=task_slices,
        )
        
        return score_id, score_ood  

    def _metrics_from_conf(self, conf_id, conf_ood):
        y = np.concatenate([
            np.ones_like(conf_ood),
            np.zeros_like(conf_id)
        ]).astype(np.int32)  # OOD=1

        s = np.concatenate([conf_ood, conf_id]).astype(np.float64)
        return compute_all_metrics(conf=s, label=y)        


    
    def train(self, avg_metrics):
        temp_dir = self.log_dir + '/temp/'
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)

        # val-task (rows) -> train-task (cols)
        metric_tables = {
            score: {"auroc": {}, "fpr95": {}}
            for score in self.ood_scores
        }
        metric_tables_local = {
            score: {"auroc": {}, "fpr95": {}}
            for score in self.ood_scores
        }

        for score in self.ood_scores:
            for val_name in self.task_names:
                metric_tables[score]["auroc"][val_name] = OrderedDict()
                metric_tables[score]["fpr95"][val_name] = OrderedDict()
                metric_tables_local[score]["auroc"][val_name] = OrderedDict()
                metric_tables_local[score]["fpr95"][val_name] = OrderedDict()


        
        # for each task
        for i in range(self.max_task):

            # save current task index
            self.current_t_index = i

            # print name
            train_name = self.task_names[i]
            print('======================', train_name, '=======================')

            # load dataset for task
            task = self.tasks_logits[i]
            if self.oracle_flag:
                self.train_dataset.load_dataset(i, train=False)
                self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
                self.add_dim += len(task)
            else:
                self.train_dataset.load_dataset(i, train=True)
                self.add_dim = len(task)
            print(len(self.train_dataset))
            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i


            # -----------------------------
            # OOD evaluation BEFORE training task i
            # OOD = current task train (task i) before training
            # -----------------------------
            for j in range(i):
                val_name = self.task_names[j]

                # global outputs once -> all global scores
                score_id_global, score_ood_global = self._collect_all_scores_for_pair(
                    id_task_idx=j,
                    ood_task_idx=i,
                    local=False
                )

                # local outputs once -> all local scores
                score_id_local, score_ood_local = self._collect_all_scores_for_pair(
                    id_task_idx=j,
                    ood_task_idx=i,
                    local=True
                )

                for score in self.ood_scores:
                    m = self._metrics_from_conf(
                        conf_id=score_id_global[score],
                        conf_ood=score_ood_global[score]
                    )
                    metric_tables[score]["auroc"][val_name][self.task_names[i]] = m["auroc"]
                    metric_tables[score]["fpr95"][val_name][self.task_names[i]] = m["fpr95"]

                    m_loc = self._metrics_from_conf(
                        conf_id=score_id_local[score],
                        conf_ood=score_ood_local[score]
                    )
                    metric_tables_local[score]["auroc"][val_name][self.task_names[i]] = m_loc["auroc"]
                    metric_tables_local[score]["fpr95"][val_name][self.task_names[i]] = m_loc["fpr95"]

                    print(
                        f"[{score}] task {j+1} vs new task {i+1} | "
                        f"AUROC={m['auroc']:.4f}, FPR95={m['fpr95']:.4f}"
                    )
                    # exit()
    
            
            # add valid class to classifier
            if not self.dil or i == 0 :
                self.learner.add_valid_output_dim(self.add_dim)

            # load dataset with memory
            self.train_dataset.append_coreset(only=False)

            # load dataloader
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=int(self.workers))

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # learn
            self.test_dataset.load_dataset(i, train=False)
            test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
            avg_train_time = self.learner.learn_batch(train_loader, self.train_dataset, model_save_dir, test_loader)

            # save model
            self.learner.save_model(model_save_dir)

            if avg_train_time is not None: avg_metrics['time']['global'][i] = avg_train_time
                
        # summarize and store this repeat's OOD matrices + per-task means
        for score in self.ood_scores:
            avg_metrics['auroc'][score] = self.summarize_scalar(
                avg_metrics['auroc'][score],
                metric_tables[score]['auroc'],
                metric_tables_local[score]['auroc'],
                key='auroc'
            )

            avg_metrics['fpr95'][score] = self.summarize_scalar(
                avg_metrics['fpr95'][score],
                metric_tables[score]['fpr95'],
                metric_tables_local[score]['fpr95'],
                key='fpr95'
            )

        return avg_metrics 





    def summarize_scalar(self, avg_dict, table, table_local, key):
        """
        key in {"auroc","fpr95"}
        avg_dict has shape like acc_dict: global [T,R], pt [T,T,R], pt-local [T,T,R]
        """        
        avg_all = avg_dict['global']
        avg_pt = avg_dict['pt']
        avg_pt_local = avg_dict['pt-local']
    
        avg_history = [np.nan] * self.max_task
        for i in range(self.max_task):
            train_name = self.task_names[i]
            vals = []
            for j in range(i):  # only previous tasks exist as val (ID)
                val_name = self.task_names[j]
    
                if train_name not in table.get(val_name, {}):
                    continue
    
                v = table[val_name][train_name]
                vals.append(v)
    
                avg_pt[j, i, self.seed] = v
                avg_pt_local[j, i, self.seed] = table_local[val_name][train_name]
    
            avg_history[i] = float(np.nanmean(vals)) if len(vals) else np.nan
    
        avg_all[:, self.seed] = np.asarray(avg_history)
        return {'global': avg_all, 'pt': avg_pt, 'pt-local': avg_pt_local}

    
    def summarize_acc(self, acc_dict, acc_table, acc_table_pt):

        # unpack dictionary
        avg_acc_all = acc_dict['global']
        avg_acc_pt = acc_dict['pt']
        avg_acc_pt_local = acc_dict['pt-local']

        # Calculate average performance across self.tasks
        # Customize this part for a different performance metric
        avg_acc_history = [0] * self.max_task
        for i in range(self.max_task):
            train_name = self.task_names[i]
            cls_acc_sum = 0
            for j in range(i+1):
                val_name = self.task_names[j]
                cls_acc_sum += acc_table[val_name][train_name]
                avg_acc_pt[j,i,self.seed] = acc_table[val_name][train_name]
                avg_acc_pt_local[j,i,self.seed] = acc_table_pt[val_name][train_name]
            avg_acc_history[i] = cls_acc_sum / (i + 1)

        # Gather the final avg accuracy
        avg_acc_all[:,self.seed] = avg_acc_history

        # repack dictionary and return
        return {'global': avg_acc_all,'pt': avg_acc_pt,'pt-local': avg_acc_pt_local}


    

    def evaluate(self, avg_metrics):

        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

        # store results
        metric_table = {}
        metric_table_local = {}
        for mkey in self.metric_keys:
            metric_table[mkey] = {}
            metric_table_local[mkey] = {}
            
        for i in range(self.max_task):

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # load model
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            self.learner.task_count = i 
            if not self.dil or i == 0:
                self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
            self.learner.pre_steps()
            self.learner.load_model(model_save_dir)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # evaluate acc
            metric_table['acc'][self.task_names[i]] = OrderedDict()
            metric_table_local['acc'][self.task_names[i]] = OrderedDict()
            self.reset_cluster_labels = True
            for j in range(i+1):
                val_name = self.task_names[j]
                metric_table['acc'][val_name][self.task_names[i]] = self.task_eval(j)
            for j in range(i+1):
                val_name = self.task_names[j]
                metric_table_local['acc'][val_name][self.task_names[i]] = self.task_eval(j, local=True)

        # summarize metrics
        avg_metrics['acc'] = self.summarize_acc(avg_metrics['acc'], metric_table['acc'],  metric_table_local['acc'])

        return avg_metrics