import pickle


class FileHandler:
    FILE_NAME = "output_model"
    @staticmethod
    def saveModel(clf):
        file_model_write = open(FileHandler.FILE_NAME, "wb")
        pickle.dump(clf, file_model_write)

    @staticmethod
    def loadModel():
        file_model_open = open(FileHandler.FILE_NAME, "rb")
        return pickle.load(file_model_open)