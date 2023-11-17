import os
import options, datasets
import numpy as np
import pickle

def run():

  print(os.getcwd())

  opts = options.get_opts()
  print("Generating Synthetic Data...")
  if not os.path.exists(opts.data_dir):
    os.makedirs(opts.data_dir)
  mydataset = datasets.get_dataset(opts)
  # sdata = mydataset.gen_sample()
  # print(type(sdata))
  # np.save(os.path.join(opts.data_dir, opts.dataset), sdata, allow_pickle=True)
  # with open(f"{opts.data_dir}/{opts.dataset}.pkl", 'wb') as fp:
  #   pickle.dump(sdata, fp)
  #   print(f"{opts.dataset} Dataset Generated. Saved in: {opts.data_dir}")
  
  # types = [
  # 'train'
  # ]
  # for t in types:
  #   dname = os.path.join(opts.data_dir,t)
  #   if not os.path.exists(dname):
  #     os.makedirs(dname)
  #   mydataset.convert_dataset(dname, t)

  dname = os.path.join(opts.data_dir,"train")
  mydataset.create_np_dataset(dname, "train")

  import pdb; pdb.set_trace();

  print("Generated Dataset.")

if __name__ == '__main__':
  run()