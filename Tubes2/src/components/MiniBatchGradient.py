import math
import numpy as np
from components.Activation import Activation
from components.Layer import Layer


class MiniBatchGradient:
    def __init__(self, nodesArray, batchSize):
        self.batchSize = batchSize
        self.layers = [Layer(nodes, Activation.SIGMOID_ACTIVATION) for nodes in nodesArray]
        self.weights = [np.random.rand(
            nodesArray[i]+1, nodesArray[i+1])-0.5 for i in range(len(nodesArray)-1)]

    def trainModel(self, trainingData, trainingTarget, epochs, minCumulativeError, learningRate):
        trainResult = {
            "epochs": epochs,
            "minErr": minCumulativeError,
            "learnRate": learningRate,
            "err": [],
            "acc": []
        }

        numOfData = len(trainingData)

        for i in range(epochs):
            accuracy = 0
            cumulativeError = 0
            count = 0
            globalDeltaW = [np.zeros(
                (self.layers[i].nodes+1, self.layers[i+1].nodes)) for i in range(len(self.layers)-1)]

            for j in range(numOfData):
                inputs = trainingData[j]
                target = trainingTarget[j]

                output = self.feedForward(inputs)
                totalError = self.sumOfError(output, target)
                cumulativeError += totalError

                localDeltaW = self.backwardProp(target, learningRate)

                for k in range(len(localDeltaW)):
                    globalDeltaW[k] += localDeltaW[k]

                if np.argmax(output) == np.argmax(target):
                    accuracy += 1

                if count == self.batchSize or i == epochs-1:
                    count = 0
                    for k in range(len(self.weights)):
                        self.weights[k] += globalDeltaW[k]
                    globalDeltaW = [np.zeros(
                        (self.layers[i].nodes+1, self.layers[i+1].nodes)) for i in range(len(self.layers)-1)]

                count += 1

            trainResult["err"].append(cumulativeError)
            trainResult["acc"].append(accuracy/numOfData)

            if (i % 100 == 0 or i == epochs-1):
                print(f"e{i}(err={cumulativeError}, acc={round(accuracy/numOfData, 2)})")

            # threshold
            if cumulativeError <= minCumulativeError:
                break

        return trainResult

    def predict(self, instances):
        results = []
        for instance in instances:
            res = self.feedForward(instance)
            idx = np.argmax(res)
            results.append(idx)

        return results

    def sumOfError(self, outputs, targets):
        errors = 0
        for i in range(len(targets)):
            errors += math.pow((targets[i]-outputs[i]), 2)

        return errors*0.5

    def feedForward(self, inputs, pr=False):
        if len(inputs) != self.layers[0].nodes:
            raise ValueError("Input error!!!")
        else:
            self.layers[0].outputs = np.array(inputs)
            instance = inputs
            for i, layer in enumerate(self.layers[1:]):
                instance = np.append(instance, 1)  # biasnya
                val = np.dot(instance, self.weights[i])
                if pr:
                    print(val, i)
                instance = layer.compute(val)

        return instance

    def backwardProp(self, targets, learningRate):
        deltaWeights = [np.zeros((self.layers[i].nodes+1, self.layers[i+1].nodes))
                        for i in range(len(self.layers)-1)]

        for i in range(len(self.layers)-1, 0, -1):
            curLayer = self.layers[i]

            if i == len(self.layers)-1:
                for j in range(curLayer.nodes):
                    outputK = curLayer.outputs[j]
                    self.layers[i].deltas[j] = outputK * \
                        (1-outputK)*(targets[j]-outputK)

            else:
                for j in range(curLayer.nodes):
                    outputH = curLayer.outputs[j]
                    self.layers[i].deltas[j] = outputH * \
                        (1-outputH) * \
                        np.dot(self.weights[i][j], self.layers[i+1].deltas)

        for n in range(len(deltaWeights)):
            outputs = self.layers[n].outputs
            deltas = self.layers[n+1].deltas

            tempResult = []
            for output in outputs:
                for delta in deltas:
                    tempResult.append(output*delta)

            for delta in deltas:
                tempResult.append(delta)

            i = 0
            for row in range(len(deltaWeights[n])):
                for col in range(len(deltaWeights[n][row])):
                    deltaWeights[n][row][col] = tempResult[i]*learningRate
                    i += 1

        return deltaWeights
