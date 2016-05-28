import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class MyConvNet1(object):
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=[32], filter_size=[7],
               hidden_dim=[100], num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    if len(num_filters) != len(filter_size):
        raise Exception('Number filters does not match number filter sizes')
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    i = 1
    for num_filters_, filter_size_ in zip(num_filters, filter_size):
        self.params['W%d' % i] = np.random.normal(scale=weight_scale,
                                                  size=(num_filters_,
                                                        C, 
                                                        filter_size_,
                                                        filter_size_))
        self.params['b%d' % i] = np.zeros(num_filters)
        i += 1
        C, H, W = num_filters_, H/2, W/2

    prev_dim = C * H * H
    for hidden_dim_ in hidden_dim:
        self.params['W%d' % i] = np.random.normal(scale=weight_scale,
                                                  size=(prev_dim, hidden_dim_))
        self.params['b%d' % i] = np.zeros(hidden_dim_)
        i += 1
        prev_dim = hidden_dim_

    self.params['W%d' % i] = np.random.normal(scale=weight_scale,
                                              size=(prev_dim, num_classes))
    self.params['b%i' % i] = np.zeros(num_classes)

    self.num_conv = len(num_filters)
    self.num_full = len(hidden_dim)
    self.num_layers = self.num_conv + self.num_full + 1
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    cache = []
    a = X
    for i in xrange(1, self.num_conv+1):
        w_name = 'W{}'.format(i)
        b_name = 'b{}'.format(i)
        a, a_cache = conv_relu_pool_forward(a, self.params[w_name], 
                                            self.params[b_name], 
                                            conv_param, pool_param)
        cache.append(a_cache)

    for i in xrange(self.num_conv+1, self.num_layers):
        w_name = 'W{}'.format(i)
        b_name = 'b{}'.format(i)
        a, a_cache = affine_relu_forward(a, 
                                         self.params[w_name], 
                                         self.params[b_name])
        cache.append(a_cache)

    w_name = 'W{}'.format(self.num_layers)
    b_name = 'b{}'.format(self.num_layers)
    a, a_cache = affine_forward(a, self.params[w_name], self.params[b_name])

    cache.append(a_cache)
    scores = a
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, da = softmax_loss(a, y)
    for i in xrange(self.num_conv+1, self.num_layers+1):
        w_name = 'W{}'.format(i)
        loss += 0.5 * self.reg * np.sum(self.params[w_name] ** 2)

    a_cache = cache.pop()
    da, dw, db = affine_backward(da, a_cache)
    w_name = 'W{}'.format(self.num_layers)
    b_name = 'b{}'.format(self.num_layers)
    grads[w_name] = dw + self.reg * self.params[w_name]
    grads[b_name] = db

    # affine layers
    for i in reversed(xrange(self.num_conv+1, self.num_layers)):
        w_name = 'W{}'.format(i)
        b_name = 'b{}'.format(i)
        a_cache = cache.pop()
        da, dw, db = affine_relu_backward(da, a_cache)
        grads[w_name] = dw + self.reg * self.params[w_name]
        grads[b_name] = db
        
    # conv layers
    for i in reversed(xrange(1, self.num_conv+1)):
        w_name = 'W{}'.format(i)
        b_name = 'b{}'.format(i)
        a_cache = cache.pop()
        da, dw, db = conv_relu_pool_backward(da, a_cache)
        grads[w_name] = dw + self.reg * self.params[w_name]
        grads[b_name] = db

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
