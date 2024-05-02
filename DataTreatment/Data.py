import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import pickle
from sklearn import tree

class CreditRiskModel:
    def __init__(self, filename='DataTreatment/risco_credito.pkl'):
        self.loadData(filename)
        self.buildTree()

    def loadData(self, filename):
        #leitor do arquivo .pkl, classificando em abscissa e ordenada
        with open(filename, 'rb') as file:
            self.xCreditRisk, self.yCreditRisk = pickle.load(file)

    def buildTree(self):
        #Define o tipo de classificador da árvore de decisão a ser utilizada
        self.riskCreditTree = DecisionTreeClassifier(criterion='entropy')
        self.riskCreditTree.fit(self.xCreditRisk, self.yCreditRisk)

#definindo as classificações para a árvore
    def DecisionTreePlot(self):
        features = ['histórico', 'dívida', 'garantias', 'renda']
        figure, axis = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        tree.plot_tree(self.riskCreditTree, feature_names=features, class_names=self.riskCreditTree.classes_, filled=True)

        plt.savefig('DecisionTree.png')

        plt.show()

