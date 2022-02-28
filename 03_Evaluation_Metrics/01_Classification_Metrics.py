import pandas as pd
import numpy as np

class ClassificationMetrics:
    def accuracy(self, y_true, y_pred):
        """ Check for correctness of the prediction and count"""
        correct_counter = 0
        for idx in range(len(y_true)):
            if y_true[idx] == y_pred[idx]:
                correct_counter += 1

        return correct_counter / len(y_true)


    def tp(self, y_true, y_pred):
        tp_counter = 0
        for idx in range(len(y_true)):
            if y_true[idx] == 1 and y_pred[idx] == 1:
                tp_counter += 1

        return tp_counter

    def fp(self, y_true, y_pred):
        fp_counter = 0
        for idx in range(len(y_true)):
            if y_true[idx] == 0 and y_pred[idx] == 1:
                fp_counter += 1

        return fp_counter

    def tn(self, y_true, y_pred):
        tn_counter = 0
        for idx in range(len(y_true)):
            if y_true[idx] == 0 and y_pred[idx] == 0:
                tn_counter += 1

        return tn_counter

    def fn(self, y_true, y_pred):
        fn_counter = 0
        for idx in range(len(y_true)):
            if y_true[idx] == 1 and y_pred[idx] == 0:
                fn_counter += 1

        return fn_counter


    def accuracy_v2(self, y_true, y_pred):
        tp = self.tp(y_true, y_pred)
        fp = self.fp(y_true, y_pred)
        tn = self.tn(y_true, y_pred)
        fn = self.fn(y_true, y_pred)

        return (tp + tn) / (tp + tn + fn + fp)

    def precision(self, y_true, y_pred):
        tp = self.tp(y_true, y_pred)
        fp = self.fp(y_true, y_pred)

        return tp / (tp + fp)

    def recall(self, y_true, y_pred):
        tp = self.tp(y_true, y_pred)
        fn = self.fn(y_true, y_pred)

        return tp / (tp + fn)
    

if __name__ == "__main__":
    y_true = [0, 1, 1, 1, 0, 0, 0, 1, 0, 1]
    y_pred = [1, 1, 1, 1, 0, 0, 0, 1, 0, 1]
    cm = ClassificationMetrics()
    print(f"accuracy: {cm.accuracy(y_true, y_pred)}")

    l1 = [0, 1, 1, 1, 0, 0, 0, 1]
    l2 = [0, 1, 0, 1, 0, 1, 0, 0]

    print(f'tp: {cm.tp(l1, l2)}')
    print(f'tn: {cm.tn(l1, l2)}')
    print(f'fn: {cm.fn(l1, l2)}')
    print(f'fp: {cm.fp(l1, l2)}')

    print(f'accuracy_v2 : {cm.accuracy_v2(y_true, y_pred)}')
