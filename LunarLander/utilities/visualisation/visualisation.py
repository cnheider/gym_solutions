import numpy as np

def update_visualiser(visualiser, episode, loss, episode_length, rgb_array, windows):
  loss_window = None
  if 'loss' in windows:
    loss_window = windows['loss']
    visualiser.line(
      X=np.array([episode]),
      Y=np.array([loss]),
      opts={'title': 'loss',
            'ytype': 'log'},
      win=loss_window,
      update='append')
  else:
    windows['loss'] = visualiser.line(
      X=np.array([episode]),
      Y=np.array([loss]),
      opts={'title': 'loss',
            'ytype': 'log'})

  episode_window = None
  if 'episode_length' in windows:
    episode_window = windows['episode_length']
    visualiser.line(
        X=np.array([episode]),
        Y=np.array([episode_length]),
        opts={'title': 'episode length',
              'ytype': 'log'},
        win=episode_window,
        update='append')
  else:
    windows['episode_length'] = visualiser.line(
    X=np.array([episode]),
    Y=np.array([episode_length]),
    opts={'title': 'episode length',
          'ytype': 'log'})

  rgb_array_window = None
  if 'rgb_array' in windows:
    rgb_array_window = windows['rgb_array']
    visualiser.image(rgb_array, win=rgb_array_window)
  else:
    windows['rgb_array'] = visualiser.image(rgb_array)

  return windows
