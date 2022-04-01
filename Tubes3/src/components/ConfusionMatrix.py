from sklearn import metrics


class ConfusionMatrix:
    def __init__(self, yTrue, yPred):
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for i in range (len(yPred)):
            if(yPred[i] == 1 and yTrue[i] == 1):
                tp += 1 # true positive
            elif(yPred[i] == 1 and yTrue[i] == 0):
                fp += 1 # false positive
            elif(yPred[i] == 0 and yTrue[i] == 1):
                fn += 1 # false negative
            elif(yPred[i] == 0 and yTrue[i] == 0):
                tn += 1 # true negative
        
        self.truePositif = tp
        self.trueNegatif = tn
        self.falseNegatif = fn
        self.falsePositif = fp
    
    def getEvaluation(self):
        return self.truePositif, self.falsePositif, self.falseNegatif, self.trueNegatif

    def getAccuray(self):
        return (self.truePositif + self.trueNegatif) / (self.trueNegatif + self.truePositif + self.falseNegatif + self.falsePositif)

    def getRecall(self):
        return self.truePositif / (self.truePositif + self.falseNegatif)

    def getPrecision(self):
        return(self.truePositif/(self.truePositif+ self.falseNegatif))
    
    def getF1(self):
        return((2*self.truePositif) / (2*self.truePositif + self.falsePositif + self.falseNegatif))
    
    @staticmethod
    def execConfusion(yTrue, yPred):
        conf = ConfusionMatrix(yTrue, yPred)

        truePositif, falsePositif, falseNegatif, trueNegatif = conf.getEvaluation()
        print("Result\n=========================================")
        print(f"True Positif   : {truePositif}")
        print(f"False Positif  : {falsePositif}")
        print(f"False Negatif  : {falseNegatif}")
        print(f"True Negatif   : {trueNegatif}")
        print(f"Accuracy       : {conf.getAccuray()}")
        print(f"Precision      : {conf.getPrecision()}")
        print(f"Recall         : {conf.getRecall()}")
        print(f"F1             : {conf.getF1()}")
        print()

        trueNegatif1, falsePositif1, falseNegatif1, truePositif1 = metrics.confusion_matrix(yTrue, yPred).ravel()
        print("Result (Sklearn)\n=========================================")
        print(f"True Positif   : {truePositif1}")
        print(f"False Positif  : {falsePositif1}")
        print(f"False Negatif  : {falseNegatif1}")
        print(f"True Negatif   : {trueNegatif1}")
        print(f"Accuracy       : {metrics.accuracy_score(yTrue,yPred)}")
        print(f"Precision      : {metrics.precision_score(yTrue,yPred)}")
        print(f"Recall         : {metrics.recall_score(yTrue,yPred)}")
        print(f"F1             : {metrics.f1_score(yTrue,yPred)}")
        print()