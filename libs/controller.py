import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init 
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed

class LSTMGenerator(nn.Module):

    def __init__(self, input_no, hidden_no, hiddenLayer_no, training , batch_size ,dropout = 0, sequence_no = 1):
        super(LSTMGenerator, self).__init__()
        self.batch_size = batch_size
        self.sequence_no = sequence_no
        self.hidden_no = hidden_no
        self.hiddenLayer_no = hiddenLayer_no
        self.lstm = nn.LSTM(input_no, hidden_no, hiddenLayer_no, dropout = dropout)

        for layer_p in self.lstm._all_weights:
            for p in layer_p:
                # if 'weight' in p:     # uncomment to initialize weights only
                init.uniform(self.lstm.__getattr__(p), -0.08, 0.08)   # intra LSTM weights/ bias initialized
                                                                      # from U[-0.08, 0.08]
        self.hidden2output = nn.Linear(hidden_no, input_no)
        init.uniform(self.hidden2output.weight.data, -0.08, 0.08)     # weights to output from U[-0.08, 0.08]  
        init.uniform(self.hidden2output.bias.data, -0.08, 0.08)       # bias to output from U[-0.08, 0.08]
        self.hidden = self.init_hidden(self.hiddenLayer_no, self.batch_size)

    def init_hidden(self, hiddenLayer_no, batch_size):
        self.hidden = (Variable(torch.zeros(hiddenLayer_no, batch_size, self.hidden_no)),  # hidden = 0
                Variable(torch.zeros(hiddenLayer_no, batch_size, self.hidden_no)))         # tried with randN also, didn't work

    def forward(self, input_, step_no, sequence_len = 1):
        output = list()
        # sum_prob = 0.
        for _ in range(step_no):
            lstm_out, self.hidden = self.lstm(
                input_.view(sequence_len, self.batch_size, -1), self.hidden)
            input_ = self.hidden2output(lstm_out.view(sequence_len,self.batch_size, -1))
            
            temp = F.softmax(input_[0,0,:])
            temp2 = F.log_softmax(input_[0,0,:])
            input_ = temp
            output.append(input_)
            indx = 0

            for i in range(4):
                if temp2.data[i] == max(temp2.data):
                    indx = i
                    break

            if _ == 0:
                sum_prob = temp2[indx]
            else:
                sum_prob = sum_prob + temp2[indx]

        out = list()
        for i in range(step_no):
            for j in range(4):
                if (output[i].data)[j] == max(output[i].data):
                    out.append(j)
                    break

        print('out:{}'.format(out))
        return out, output[-1], sum_prob    #''' summing up all max probs '''
class Controller(object):
    def __init__(self, args, state_space_parameters):

        self.state_space_parameters = state_space_parameters
        self.input_no, self.hidden_no, self.hiddenLayer_no =  4, \
                                                            state_space_parameters.controller_hidden_no, \
                                                            state_space_parameters.controller_hiddenLayer_no 

        self.lstm_gen = LSTMGenerator(self.input_no, self.hidden_no, self.hiddenLayer_no, training = True, \
                                  batch_size = state_space_parameters.controller_batch_size)
        self.lstm_gen = torch.nn.DataParallel(self.lstm_gen)               ### Uncomment when cuda is used
        self.lstm_gen.cuda()


    def generate_net_representation(self, layer_no):
        input_ = Variable(torch.ones(4)).cuda()
        for i in range(4):                          
            (input_.data)[i] = np.random.randint(4)         # comment out to avoid random intial input
        return self.lstm_gen(input_, layer_no*3)

    def generate_hyperparams(self, out, layer_no):
        hyperparams = [1 for _ in range(0,layer_no*3,3)]
        assert(len(out) == layer_no*3)
        for i in range(0,layer_no*3,3):
            hyperparams[i/3] = ('conv', self.state_space_parameters.possible_conv_sizes[out[i]],\
                               self.state_space_parameters.possible_conv_depths[out[i+1]],
                               self.state_space_parameters.possible_conv_strides[out[i+2]] )
        return hyperparams



    def train_controller(self, reinforce_loss, val_accuracy_cube_per_step):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.lstm_gen.parameters()), # model.parameters()#uncomment if whole network being trained
                       lr = self.state_space_parameters.controller_training_lr)
                       # betas = (state_space_parameters.beta1, state_space_parameters.beta2), \ ''' default '''
                       # eps = state_space_parameters.eps                                        ''' default '''
                       # weight_decay = state_space_parameters.weight_decay_rate, ''' no weight decay '''
        criterion = nn.L1Loss(size_average = False).cuda()
        cudnn.benchmark = True
        target = torch.zeros(reinforce_loss.data.size())
        target =  target.cuda(async=True)
        target_var = Variable(target).cuda()  
        reinforce_loss.data = reinforce_loss.data*val_accuracy_cube_per_step
        loss = criterion(reinforce_loss, target_var)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
