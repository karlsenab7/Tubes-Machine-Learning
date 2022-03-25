import math
import numpy as np
from components.Activation import Activation 



class Layer:
  
  def __init__(self, nbNodes, act_type=Activation.SIGMOID_ACTIVATION):
    self.nbNodes = nbNodes
    self.outputs = np.zeros(nbNodes) #+1 untuk node bias
    self.inputs = np.zeros(nbNodes)
    self.deltas = np.zeros(nbNodes)
    self.activation = act_type

  
  def activationFunction(self, x):
    return Activation.active(x, self.activation)
    
  
  # Return sebuah numpy array yang sizenya sama dengan nbNodes
  def compute(self, inputArr):
    if len(inputArr) > self.nbNodes:
      raise ValueError("Shape of input ({:d}) does not match number of nodes ({:d})".format(len(inputArr), self.nbNodes))
    else:
      self.inputs = np.array(inputArr)
      self.outputs = np.array([self.activationFunction(y) for y in inputArr])
      return self.outputs