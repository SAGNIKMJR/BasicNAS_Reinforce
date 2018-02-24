import os
import sys
import numpy as np
import pandas as pd
import torch
from libs.parser import Parser
from libs.controller import Controller
import models.mnist.hyper_parameters
import models.mnist.state_space_parameters as state_space_parameters
import libs.train_ as train_

replay_dictionary = pd.DataFrame(columns=['net',
                                     'accuracy_best_val',
                                     'accuracy_last_val',
                                     'layer_no',
                                     'train_flag'])
class Child(object):
    def __init__(self, args, layer_no, controller):
        global replay_dictionary
        self.data = args.data  #''' path to dataset '''
        self.controller = controller
        self.out, self.output,self.sum_prob = self.controller.generate_net_representation(layer_no)
        self.hyperparams = self.controller.generate_hyperparams(self.out, layer_no)
        hyperparams_string = str(self.hyperparams)
        if hyperparams_string not in replay_dictionary['net'].values:
            self.acc_best_val, self.acc_last_val, self.train_flag = train_.train_val_net(self.hyperparams, \
                                                                state_space_parameters, \
                                                                args.data)
            print('best_val_acc:{}, last_val_acc:{} and train_flag:{}'.format(self.acc_best_val, \
                                                                        self.acc_last_val, self.train_flag))
            replay_dictionary = replay_dictionary.append(pd.DataFrame([[hyperparams_string, self.acc_best_val, self.acc_last_val, \
                            layer_no, self.train_flag]], columns=['net', 'accuracy_best_val', \
                            'accuracy_last_val', 'layer_no', 'train_flag']), ignore_index = True)


        else:
            print('net already present --')
            self.acc_best_val = replay_dictionary[replay_dictionary['net']==hyperparams_string]['accuracy_best_val'].values[0]
            self.acc_last_val = replay_dictionary[replay_dictionary['net']==hyperparams_string]['accuracy_last_val'].values[0]
            self.train_flag = replay_dictionary[replay_dictionary['net']==hyperparams_string]['train_flag'].values[0]
            print('best_val_acc:{}, last_val_acc:{} and train_flag:{}'.format(self.acc_best_val, \
                                                                        self.acc_last_val, self.train_flag))

        self.controller.train_controller(self.sum_prob, (self.acc_best_val ** 3)/float(layer_no*3))   #''' should be max val accuracy of last 5 epochs cubed '''

    def get_controller(self):
        return self.controller

def init_train_child_update_controller(args, layer_no, controller):
    child_cnn = Child(args, layer_no, controller)
    controller = child_cnn.get_controller()
    return controller

def main():
    args = Parser().get_parser().parse_args()
    layer_no = state_space_parameters.layer_no
    controller = Controller(args, state_space_parameters)
    for arch_no in range(state_space_parameters.total_arch):
        print('arch no.:{}, layer np.:{}'.format(arch_no, layer_no))
        controller = init_train_child_update_controller(args, layer_no, controller)
        if (arch_no+1) % 500 == 0:
            layer_no += state_space_parameters.layer_increment
    replay_dictionary.to_csv('./replayDict1.csv')

if __name__ == '__main__' :
    main()