import csv
from collections import defaultdict
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import pprint
import shutil
import torch

plt.style.use('ggplot')

def mkdirp(logdir):
    if '_debug' in logdir:
        # overwrite
        if not os.path.exists(logdir):
            os.mkdir(logdir)
    else:
        try:
            os.mkdir(logdir)
        except FileExistsError:
            overwrite = 'o'
            while overwrite not in ['y', 'n']:
                overwrite = input('{} exists. Overwrite? [y/n] '.format(logdir))
            if overwrite == 'y':
                shutil.rmtree(logdir)
                os.mkdir(logdir)
            else:
                raise FileExistsError

class RunningAverage(object):
    def __init__(self):
        super(RunningAverage, self).__init__()
        self.data = {}
        self.alpha = 0.01

    def update_variable(self, key, value):
        if not self.exists(key):
            self.data['running_'+key] = value
        else:
            self.data['running_'+key] = (1-self.alpha) * self.data['running_'+key] + self.alpha * value
        return copy.deepcopy(self.data['running_'+key])

    def get_value(self, key):
        if self.exists(key):
            return self.data['running_'+key]
        else:
            assert KeyError

    def exists(self, key):
        return 'running_'+key in self.data

class Logger(object):
    def __init__(self, expname, logdir, params, variables=None, resumed_from=None):
        super(Logger, self).__init__()
        self.data = {}
        self.metrics = {}

        self.expname = expname
        self.logdir = logdir
        self.params = params
        self.resumed_from = resumed_from if resumed_from else None

        self.run_avg = RunningAverage()

        if self.resumed_from:
            assert os.path.exists(self.resumed_from)
            if os.path.dirname(self.resumed_from) != self.logdir:
                mkdirp(self.logdir)
        else:
            mkdirp(self.logdir)

        if variables is not None:
            self.add_variables(variables)

    def set_expname(self, expname):
        self.expname = expname

    def set_resumed_from(self, resume_path):
        self.data['resumed_from'] = resume_path

    #############################################
    def load_params_eval(self, eval_, resume):
        """ saved_args is mutable! """
        assert self.resumed_from is not None
        saved_args = self.load_params()
        saved_args.eval = eval_
        saved_args.resume = resume
        self.set_resumed_from(self.resumed_from)
        return saved_args

    def load_params_transfer(self, transfer, resume):
        """ saved_args is mutable! """
        assert self.resumed_from is not None
        saved_args = self.load_params()
        saved_args.transfer = transfer
        saved_args.resume = resume
        self.set_resumed_from(self.resumed_from)
        return saved_args

    # should be able to combine the above
    #############################################

    def save_params(self, logdir, params, ext=''):
        pickle.dump(params, open(os.path.join(self.logdir, '{}.p'.format('params'+ext)), 'wb'))

    def set_params(self, params):
        """ params is mutable """
        self.params = params

    def set_and_save_params(self, logdir, params, ext=''):
        self.set_params(params)
        self.save_params(logdir, params, ext)

    def load_params(self):
        params = pickle.load(open(os.path.join(self.logdir, '{}.p'.format('params')), 'rb'))
        return params

    def add_variables(self, names):
        for name in names:
            self.add_variable(name)

    def add_variable(self, name, include_running_avg=False):
        self.data[name] = []
        if include_running_avg:
            self.data['running_{}'.format(name)] = []

    def update_variable(self, name, index, value, include_running_avg=False):
        if include_running_avg:
            running_name = 'running_{}'.format(name)
            self.run_avg.update_variable(name, value)
            self.data[running_name].append((index, self.run_avg.get_value(name)))
        self.data[name].append((index, value))

    def get_recent_variable_value(self, name):
        index, recent_value = copy.deepcopy(self.data[name][-1])
        return recent_value

    def has_running_avg(self, name):
        return self.run_avg.exists(name)

    def add_metric(self, name, initial_val, comparator):
        self.metrics[name] = {'value': initial_val, 'cmp': comparator}

    def save_checkpoint(self, ckpt_data, current_metric_keys, i_iter, ext):
        current_metrics = {cm_key: self.get_recent_variable_value(cm_key) for cm_key in current_metric_keys}
        old_ckpts = [x for x in os.listdir(self.logdir) if '.pth.tar' in x and 'best' in x and ext in x]
        assert len(old_ckpts) <= len(current_metrics)

        for m in self.metrics:
            if self.metrics[m]['cmp'](current_metrics[m], self.metrics[m]['value']):
                self.metrics[m]['value'] = current_metrics[m]
                if any(m in oc for oc in old_ckpts):
                    old_ckpts_with_metric = [x for x in old_ckpts if 'best{}'.format(m) in x]
                    assert len(old_ckpts_with_metric) == 1
                    old_ckpt_to_remove = os.path.join(self.logdir,old_ckpts_with_metric[0])
                    os.remove(old_ckpt_to_remove)
                    self.printf('Removing {}'.format(old_ckpt_to_remove))
                torch.save(ckpt_data, os.path.join(self.logdir, '{}_iter{:.0e}_best{}{}.pth.tar'.format(
                    self.expname, i_iter, m, ext)))
                self.printf('Saved Checkpoint for best {}'.format(m))
            else:
                self.printf('Did not save {} checkpoint at because {} was worse than the best'.format(m, m))

    def plot(self, var1_name, var2_name, fname=None):
        if not fname:
            fname = '{}_{}'.format(self.expname, var2_name)
        x, y = zip(*self.data[var2_name])
        plt.plot(x,y)
        plt.xlabel(var1_name)
        plt.ylabel(var2_name)
        plt.savefig(os.path.join(self.logdir,'{}.png'.format(fname)))
        plt.clf()

    def to_cpu(self, state_dict):
        cpu_dict = {}
        for k,v in state_dict.items():
            cpu_dict[k] = v.cpu()
        return cpu_dict

    def save_csv(self, clear_data=False):
        name = self.expname
        csv_dict = defaultdict(dict)
        for key, value in self.data.items():
            for index, e in value:
                csv_dict[index][key] = e
        filename = os.path.join(self.logdir,'{}.csv'.format(name))
        file_exists = os.path.isfile(filename)
        with open(filename, 'a+') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.data.keys())
            if not file_exists:
                writer.writeheader()
            for i in sorted(csv_dict.keys()):
                writer.writerow(csv_dict[i])
        if clear_data: self.clear_data()  # to save memory

    def load_csv(self):
        filename = os.path.join(self.logdir,'{}.csv'.format(self.expname))
        df = pd.read_csv(filename)
        return df

    def plot_from_csv(self, var_pairs):
        df = self.load_csv()
        for var1_name, var2_name in var_pairs:
            data = df[[var1_name, var2_name]].dropna()
            x = data[var1_name].tolist()
            y = data[var2_name].tolist()
            fname = '{}_{}'.format(self.expname, var2_name)
            plt.plot(x,y)
            plt.xlabel(var1_name)
            plt.ylabel(var2_name)
            plt.savefig(os.path.join(self.logdir,'{}.png'.format(fname)))
            plt.clf()

    def save_pickle(self, name):
        pickle.dump(self.data, open(os.path.join(self.logdir,'{}.p'.format(name)), 'wb'))

    def load_pickle(self, name):
        self.data = pickle.load(open(os.path.join(self.logdir,'{}.p'.format(name)), 'rb'))

    def printf(self, string):
        if self.params.printf:
            f = open(self.logdir+'.txt', 'a')
            print(string, file=f)
        else:
            print(string)

    def pprintf(self, string):
        if self.params.printf:
            f = open(self.logdir+'.txt', 'a')
            pprint.pprint(string, stream=f)
        else:
            pprint.pprint(string)

    def clear_data(self):
        for key in self.data:
            self.data[key] = []

def create_logger(build_expname, args):
    if args.resume:
        """
            - args.resume identifies the checkpoint that we will load the model
            - We will load the args from the saved checkpoint and overwrite the 
            default args.
            - The only things we will not overwrite is args.eval and args.resume,
            which have been provided by the current run
            - We will also set the resumed_from attribute of logger to point to
            the current checkpoint we just loaded up.
            - TODO
                it's an open question of whether we should resave the params
                again, which would now contain values for args.eval and 
                args.resume
        """
        if args.eval:
            logdir = os.path.dirname(args.resume)
            logger = Logger(
                expname='',  # will overwrite
                logdir=logdir,
                params={},  # will overwrite
                resumed_from=args.resume)
            args = logger.load_params_eval(args.eval, args.resume)
            expname = build_expname(args) + '_eval'
            logger.set_expname(expname)
            logger.save_params(logger.logdir, args, ext='_eval')
        elif args.transfer:
            expname = build_expname(args) + '_transfer'
            logger = Logger(
                expname=expname,
                logdir=os.path.join(args.outputdir, expname),
                params=args,
                resumed_from=args.resume)
            logger.save_params(logger.logdir, args, ext='_transfer')
        else:
            assert False, 'You tried to resume but you did not specify whether we are in eval or transfer mode'
    else:
        expname = build_expname(args)   
        logger = Logger(
            expname=expname, 
            logdir=os.path.join(args.outputdir, expname), 
            params=args, 
            resumed_from=None)
        logger.save_params(logger.logdir, args)
    return logger


def display_stats(stats):
    metrics = list(stats.keys())
    max_metric_length = max(len(x) for x in metrics)
    aggregate_keys = list(stats[metrics[0]].keys())
    num_aggregates = len(aggregate_keys)
    agg_label_length = max(len(x) for x in stats[metrics[0]]) + 3
    ###############################################################
    value_length = 7
    pad = 2
    lefter_width = pad + max_metric_length + pad
    column_width = pad + agg_label_length + value_length + pad
    border_length = lefter_width + (column_width+1)*num_aggregates + 3
    ###############################################################
    doubledash = '=' * border_length
    dash = '-' * border_length
    display_str = '{}\n'.format(doubledash)
    header_str = '|{:^{width}s} '.format('', width=lefter_width)
    for a in sorted(aggregate_keys):
        header_str += '|{:^{width}}'.format(a, width=column_width)
    display_str += header_str +'|'
    display_str += '\n{}\n'.format(dash)
    ###############################################################
    for m in sorted(stats.keys()):
        metric_str = '|{:^{width}s} '.format(m, width=lefter_width)
        for a in sorted(stats[m].keys()):
            metric_str += '|{:^{width}.4f}'.format(stats[m][a], width=column_width)
        display_str += metric_str+'|\n'
    ###############################################################
    display_str += doubledash
    return display_str

def merge_log(log_list):
    metrics = list(log_list[0].keys())
    if 'frame' in log_list[0]:
        metrics.remove('frame')
    log = defaultdict(dict)
    aggregators = {'total': np.sum, 'avg': np.mean, 'max': np.max, 'min': np.min, 'std': np.std}
    for m in metrics:
        metric_data = [x[m] for x in log_list]
        for a in aggregators:
            log[m][a] = aggregators[a](metric_data)
    return log

def visualize_parameters(model, aString=None):
    if aString:
        print(aString)
    for n, p in model.named_parameters():
        if p.grad is None:
            print(n, p.size(), p.data.norm(), "No grad")
        else:
            print(n, p.size(), p.data.norm(), p.grad.data.norm(), torch.max(p.grad.data))
