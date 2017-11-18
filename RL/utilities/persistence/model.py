# coding=utf-8
import datetime
import glob

import shutil

import os
import torch


def load_model(configuration):
  _list_of_files = glob.glob(configuration.MODEL_DIRECTORY + '/*')
  _latest_model = max(_list_of_files, key=os.path.getctime)
  print('loading previous model: ' + _latest_model)
  return torch.load(_latest_model)

def save_model(model, configuration):
  _model_date = datetime.datetime.now()
  _model_name = '{}-{}-{}.model'.format(configuration.DATA_SET,
                                        configuration.CONFIG_NAME.replace('.',
                                                                          '_'),
                                        _model_date.strftime('%y%m%d%H%M'))
  _model_path = os.path.join(configuration.MODEL_DIRECTORY, _model_name)
  torch.save(model.state_dict(), _model_path)
  save_config(_model_path, configuration)

def save_config(model_path,configuration):
  config_path = os.path.join(configuration.CONFIG_DIRECTORY,
                               configuration.CONFIG_FILE)
  shutil.copyfile(config_path, model_path+'.py')