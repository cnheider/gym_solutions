import numpy as np


def update_visualiser(visualiser, phase, epoch_loss, running_loss, epoch,
                      widget_index_number_evaluation,
                      widget_index_number_training):
  if phase == 'evaluation':
    if widget_index_number_evaluation:
      visualiser.line(
          X=np.array([epoch]),
          Y=np.array([epoch_loss]),
          opts={'title': 'evaluation loss',
                'ytype': 'log'},
          win=widget_index_number_evaluation,
          update='append')

    else:
      widget_index_number_evaluation = visualiser.line(
          X=np.array([epoch]),
          Y=np.array([running_loss]),
          opts={'title': 'evaluation loss',
                'ytype': 'log'})
  else:
    if widget_index_number_training:
      visualiser.line(
          X=np.array([epoch]),
          Y=np.array([epoch_loss]),
          opts={'title': 'training loss',
                'ytype': 'log'},
          win=widget_index_number_training,
          update='append')
    else:
      widget_index_number_training = visualiser.line(
          X=np.array([epoch]),
          Y=np.array([running_loss]),
          opts={'title': 'training loss',
                'ytype': 'log'})

  return widget_index_number_evaluation, widget_index_number_training
