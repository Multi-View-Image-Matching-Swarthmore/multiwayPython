import os
import options, datasets
import numpy as np

if __name__ == '__main__':
  opts = options.get_opts()
  print("Generating Pose Graphs")
  if not os.path.exists(opts.data_dir):
    os.makedirs(opts.data_dir)
  mydataset = datasets.get_dataset(opts)
  sdata = mydataset.gen_sample()
  # print(sdata)
  np.save(os.path.join(opts.data_dir, opts.dataset), sdata, allow_pickle=True)
  print(f"{opts.dataset} Dataset Generated. Saved in: {opts.data_dir}")