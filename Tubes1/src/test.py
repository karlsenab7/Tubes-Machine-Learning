from ntpath import join
import os
from components.FFNNModel import FFNNModel

MODELS_DIR = join(os.path.dirname(os.path.abspath(__file__)), "models")
TEST_DIR = join(os.path.dirname(os.path.abspath(__file__)), "test")

MODEL_FILE = "model1.json"
TEST_FILE = "test.csv"

MODEL_PATH = join(MODELS_DIR, MODEL_FILE)
TEST_PATH = join(TEST_DIR, TEST_FILE)

fnn = FFNNModel.getModelFromFile(MODEL_PATH)
fnn.setInitialInput([1, 1])
res = fnn.solve()
print(res)


from graphviz import Digraph
f = Digraph('Feed Forward Neural Network', filename='model 1.gv')
f.attr('node', shape='circle', fixedsize='true', width='0.9')

for i in range(len(fnn.layers)) :
    if i != 0:
        if i == 1:
            for j in range(len(fnn.layers[i].weights)) :
                for k in range(len(fnn.layers[i].weights[j])) :
                    if k == 0:
                        if j == 0:
                            f.edge(f'bx', f'h{i}_{k}', str(
                            fnn.layers[i].weights[j][k]))
                        else:
                            f.edge(f'bx', f'h{i}_{k+1}', str(
                                fnn.layers[i].weights[j][k]))
                    else :
                        if j == 0:
                            if k == 2:
                                f.edge(f'x{j}', f'h{i}_{k-1}', str(
                                fnn.layers[i].weights[j+1][k]))
                            else:
                                f.edge(f'x{j}', f'h{i}_{k-1}', str(
                                fnn.layers[i].weights[j][k]))
                        else:
                            if k == 1:
                                f.edge(f'x{j}', f'h{i}_{k-1}', str(
                                fnn.layers[i].weights[j-1][k]))
                            else:
                                f.edge(f'x{j}', f'h{i}_{k-1}', str(
                                fnn.layers[i].weights[j][k]))

        elif i ==2 :
            for j in range(len(fnn.layers[i].weights)):
                for k in range(len(fnn.layers[i].weights[j])):
                    if k == 0:
                        f.edge(f'bhx{i-1}', f'h{i}_{j}', str(
                            fnn.layers[i].weights[j][k]))
                    elif k == 1 :
                        f.edge(f'h{i-1}_{j}', f'h{i}_{0}', str(
                            fnn.layers[i].weights[j][k]))
                    else:
                        f.edge(f'h{i-1}_{j+1}', f'h{i}_{0}', str(
                            fnn.layers[i].weights[j][k]))
       
f.view()