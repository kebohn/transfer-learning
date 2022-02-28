import os, json
import matplotlib.pyplot as plt

def plot(path):
  for datasets in sorted(os.listdir(path)): # iterate over datasets
    res = {}
    data_dir = os.fsdecode(datasets)
    data_path = F'{path}{data_dir}'
    if os.path.isfile(data_path): # only consider directories
      continue
    for metrics in sorted(os.listdir(data_path)): # iterate over metrics
      metric_dir = os.fsdecode(metrics)
      metric_path = F'{path}{data_dir}/{metric_dir}'
      if os.path.isfile(metric_path): # only consider directories
        continue
      for file_name in sorted(os.listdir(metric_path)): # iterate over files
        if not file_name.endswith('.json'): # only consider json files
          continue
        with open(F'{metric_path}/{file_name}', 'r') as f:
          data = json.load(f)
          steps = []
          accs = []
          for key, val in data.items(): # only consider total accuracies for each metric
            steps.append(key)
            accs.append(val["total_acc"])
          res[metric_dir] = {
            "steps": steps,
            "accuracy": accs,
          }

    plt.figure()
    for key, val in res.items(): # plot line for each dataset
      plt.plot(val["steps"], val["accuracy"])
    plt.xlabel('Training Size') 
    plt.ylabel('Accuracy') 
    plt.title(data_dir)
    plt.legend(list(res.keys()), loc='upper left')
    plt.savefig(F'{data_dir}_total_acc.jpg')


# plot accuracy values of all used datasets and all metrics
path = '/local/scratch/bohn/results/'
plot(path)
