# -*- coding: utf-8 -*-
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

class Preprocessing(object):
    # TODO: currently only GCN
    """Compute the mean and std value of dataset."""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.mean = torch.zeros(3)
        self.std = torch.zeros(3)
        
    def get_mean_and_std(self, datasetdir, nworkers): 
        dataset = datasets.ImageFolder(datasetdir, transforms.Compose([transforms.ToTensor()]))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers= nworkers)
        print('==> Computing mean and std..')
        for inputs, targets in dataloader:
            for i in range(3):
                # note that PyTorch always has a dummy batch dimension
                self.mean[i] += inputs[:,i,:,:].mean()
                ## TODO: improper calculation of STD
                self.std[i] += inputs[:,i,:,:].std()
        self.mean.div_(len(dataset))
        self.std.div_(len(dataset))
