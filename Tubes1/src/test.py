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