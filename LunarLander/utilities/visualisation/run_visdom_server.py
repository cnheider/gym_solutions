# http://localhost:8097
# python -m visdom.server

import visdom.server as server

def run_visdom_server():
  server.main()


if __name__ == '__main__':
  run_visdom_server()