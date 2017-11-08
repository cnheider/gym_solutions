import torch
from neodroid.models import Reaction, Motion
from torch.autograd import Variable

_use_cuda = torch.cuda.is_available()

FloatTensor = torch.cuda.FloatTensor if _use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if _use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if _use_cuda else torch.ByteTensor


def sample_action(input_state, model, steps_done):
    vari = Variable(input_state, volatile=True).type(FloatTensor)
    action_probs = model(vari)
    action = action_probs.multinomial()
    return action.data

def testing_sample_action(input_state, model):
  vari = Variable(input_state, volatile=True).type(FloatTensor)
  action_probablities = model(vari)
  action = action_probablities.multinomial()
  return action.data
