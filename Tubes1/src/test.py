import math
from FFNNModel import FFNNModel

fnn = FFNNModel.getModelFromFile('model.json')
fnn.setInitialInput([1, 1])
res = fnn.solve()
print(res)