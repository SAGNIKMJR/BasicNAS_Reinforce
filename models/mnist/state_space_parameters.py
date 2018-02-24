''' for CIFAR, 5000 randomly from training set for validation, rest 45000 for training ''' 
output_states = 10                                                                          # Number of Classes
image_size = 28      # ''' 32 in paper, randomly cropped, whitened, randomly flipped '''     # Size of images before they enter network (or smallest dimension of image)
weight_decay_rate = 0.0001              # ''' 1e-4 in paper, 0.0005 originally'''            # Weight Decay Factor
momentum = 0.9         #''' Nesterov momentum in paper'''
acc_threshold = 0.15                                                                        # Model must achieve greater than ACC_THRESHOLD performance in NUM_ITER_TO_TRY_LR or it is killed and next initial learning rate is tried.
train_batch_size = 128    #''' not in paper '''                                              # Training Batch Size
eval_batch_size = 100    # ''' not in paper '''                                              # Validation Batch Size
beta1 = 0.9                                                                                 # Adam Optim
beta2 = 0.999                                                                               # Adam Optim
eps = 1e-08                                                                                 # Adam Optim
training_lr = 0.1         # ''' 0.1 in paper, 0.001 originally '''                           # Training Initial Learning Rate
controller_training_lr = 0.0006 #''' ADAM in paper ''' 
controller_hidden_no = 35      # ''' as in paper '''
controller_hiddenLayer_no = 2  # ''' as in paper '''
controller_batch_size = 1     # '''  not in paper, looks like 1 '''
layer_no = 6 #''' starting layer limit, as in paper '''                                      # Max number of layers
workers = 4                                                                                 # Number of data loading workers
input_channel = 3
train_sampler = None                                                                        # Train sampler
end_epoch = 20    #''' 50 in paper '''                                                        # Number of epochs to train
total_arch = 2000 #'''12,800 in paper'''
layer_increment = 2 #''' 1600 in paper '''
increment_after = 500 #'''1600 in paper i.e. 8 increments'''
# Transition Options
possible_conv_depths = [24, 36, 48, 64]  #''' as in paper '''                                  # Choices for number of filters in a convolutional layer
possible_conv_sizes = [1,3,5,7]     #''' as in paper '''                                       # Choices for kernel size (square), stride always = 1
''' uncomment for using non-1 strides '''
# possible_conv_strides = [1, 2, 3, 4] ''' [1,2,3] in paper, 4 added for easier implementation ''' 
''' comment for using stride = 1 only '''
possible_conv_strides = [1, 1, 1, 1] #''' same as above ''' 
# possible_pool_sizes = [[5,3], [3,2], [2,2]]                                                 # Choices for [kernel size, stride] for a max pooling layer
possible_pool_sizes = [5, 3, 2]
possible_pool_strides = [3, 2, 2]
max_fc = 2                                                                                  # Maximum number of fully connected layers (excluding final FC layer for softmax output)
possible_fc_sizes = [i for i in [512, 256, 128] if i >= output_states]                      # Possible number of neurons in a fully connected layer

allow_initial_pooling = False                                                               # Allow pooling as the first layer
init_utility = 0.5                                                                          # Set this to around the performance of an average model. It is better to undershoot this
allow_consecutive_pooling = False                                                           # Allow a pooling layer to follow a pooling layer

conv_padding = 'SAME'                                                                       # set to 'SAME' (recommended) to pad convolutions so input and output dimension are the same
# conv_padding = 'VALID'                                                                                    # set to 'VALID' to not pad convolutions

batch_norm = False                                                                          # Add batchnorm after convolution before activation


# Epislon schedule for q learning agent.
# Format : [[epsilon, # unique models]]
# Epsilon = 1.0 corresponds to fully random, 0.0 to fully greedy
# epsilon_schedule = [[1.0, 1500],
#                     [0.9, 100],
#                     [0.8, 100],
#                     [0.7, 100],
#                     [0.6, 150],
#                     [0.5, 150],
#                     [0.4, 150],
#                     [0.3, 150],
#                     [0.2, 150],
#                     [0.1, 150]]

epsilon_schedule = [[1.0, 1],
                    [0.9, 1]]
                    # [0.8, 1],
                    # [0.7, 1],
                    # [0.6, 1],
                    # [0.5, 1],                 #Kindly ignore this, was there for testing code
                    # [0.4, 1],
                    # [0.3, 1],
                    # [0.2, 1],
                    # [0.1, 1]]
# Q-Learning Hyper parameters
learning_rate = 0.01                                                                        # Q Learning learning rate (alpha from Equation 3)
discount_factor = 1.0                                                                       # Q Learning discount factor (gamma from Equation 3)
replay_number = 128                                                                         # Number trajectories to sample for replay at each iteration

# Set up the representation size buckets (see paper Appendix Section B)
def image_size_bucket(image_size):
    if image_size > 7: # image size between 8 and infinity
        return 8
    elif image_size > 3: # image size between 4 and 7
        return 4
    else:
        return 1         # image size between 1 and 3
    # return image_size

# Condition to allow a transition to fully connected layer based on the current representation size

# def allow_fully_connected(representation_size):
#     return representation_size <= 4

def allow_fully_connected(image_size):
    return image_size <= 7


