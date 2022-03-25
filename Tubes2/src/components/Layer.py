import numpy as np
from components.Activation import Activation 



class Layer:
  
  def __init__(self, nodes, act_type=Activation.SIGMOID_ACTIVATION):
    self.nodes = nodes
    self.outputs = np.zeros(nodes)
    self.inputs = np.zeros(nodes)
    self.deltas = np.zeros(nodes)
    self.activation = act_type

  
  def activationFunction(self, x):
    return Activation.active(x, self.activation)
    
  def compute(self, inputArr):
    if len(inputArr) > self.nodes:
      raise ValueError("Input Error!!!")
    else:
      self.inputs = np.array(inputArr)
      self.outputs = np.array([self.activationFunction(y) for y in inputArr])
      return self.outputs