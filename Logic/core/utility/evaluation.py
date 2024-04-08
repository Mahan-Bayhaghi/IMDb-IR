from typing import List
import numpy as np


# *** NOTE ***
# each list is a list of strings
# we will assume first index is query and the rest are relevant movie ids retrieved

class Evaluation:

    def __init__(self, name: str):
        self.name = name

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The precision of the predicted results
        """
        # TODO: Calculate precision here
        predict_precisions = []
        for index in range(len(predicted)):
            predict = predicted[index]
            query = predict[0]
            predicted_ids = predict[1:]
            predict_precision = len([set(predicted_ids).intersection(set(actual[index]))]) / len(predicted_ids)
            predict_precisions.append(predict_precision)
        precision = 0.0
        if len(predict_precisions) != 0:
            precision = sum(predict_precisions) / len(predict_precisions)
        return precision

    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the recall of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The recall of the predicted results
        """
        # TODO: Calculate recall here
        predict_recalls = []
        for index in range(len(predicted)):
            predict = predicted[index]
            query = predict[0]
            predicted_ids = predict[1:]
            predict_recall = len([set(predicted_ids).intersection(set(actual[index]))]) / len(actual[index])
            predict_recalls.append(predict_recall)
        recall = 0.0
        if len(predict_recalls) != 0:
            recall = sum(predict_recalls) / len(predict_recalls)
        return recall

    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the F1 score of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The F1 score of the predicted results
        """
        # TODO: Calculate F1 here
        predict_f1s = []
        for index in range(len(predicted)):
            recall = self.calculate_recall(actual, predicted)
            precision = self.calculate_precision(actual, predicted)
            predict_f1 = 0.0
            if (precision + recall) > 0:
                predict_f1 = (2 * recall * precision) / (recall + precision)
            predict_f1s.append(predict_f1)
        f1 = 0.0
        if len(predict_f1s) > 0:
            f1 = sum(predict_f1s) / len(predict_f1s)
        return f1

    def calculate_AP(self, actual: List[List[str]], predicted: List[List[str]]) -> list[float]:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        list[float]
            The Average Precision of the predicted results
        """
        AP = []

        # TODO: Calculate AP here
        for index, predict in enumerate(predicted):
            query = predict[0]
            predicted_ids = predict[1:]
            actual_ids = actual[index]
            p_at_ks = []
            for i, predicted_id in enumerate(predicted_ids):
                if predicted_id in actual_ids:
                    p_at_ks.append(self.calculate_precision(actual, predicted[:i]))
            average_AP = 0.0
            if len(p_at_ks) > 0:
                average_AP = sum(p_at_ks) / len(p_at_ks)
            AP.append(average_AP)
        return AP

    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Average Precision of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The Mean Average Precision of the predicted results
        """
        # TODO: Calculate MAP here
        AP = self.calculate_AP(actual, predicted)
        MAP = 0.0
        if len(AP) > 0:
            MAP = sum(AP) / len(AP)
        return MAP

    def calculate_DCG(self, actual: List[List[(str, int)]], predicted: List[List[(str, int)]]) -> list[list[float]]:
        """
        Calculates the Discounted Cumulative Gain (DCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        list[list[float]]
            The DCG of the predicted results
        """
        DCG = []

        # TODO: Calculate DCG here
        for index, predict in enumerate(predicted):
            query = predict[0][0]
            predicted_movies = predict[1:]
            predicted_ids = [p[0] for p in predicted_movies]
            predicted_rels = [p[1] for p in predicted_movies]
            actual_ids = [a[0] for a in actual[index]]
            actual_rels = [a[1] for a in actual[index]]
            for i, predicted_id in enumerate(predicted_ids):
                if predicted_id in actual_ids:
                    DG = actual_rels[actual_ids.index(predicted_ids)] / np.log(i)
                    DCG.append(DG)
        for i in range(1, len(DCG)):
            DCG[i] += DCG[i - 1]
        return DCG

    def calculate_NDCG(self, actual: List[List[(str, int)]], predicted: List[List[(str, int)]]) -> list[list[float]]:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG) of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
            The actual results is in fact the perfect scoring for each query
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The NDCG of the predicted results
        """
        NDCG = []

        # TODO: Calculate NDCG here
        DCG = self.calculate_DCG(actual, predicted)

        # sort each actual DCG list to ideal ranking
        actual.sort(key=lambda x: x[1], reverse=True)
        for index, DCG_list in enumerate(DCG):
            NDCG_list = []
            for i in range(len(DCG_list)):
                NDCG_list.append(DCG_list[i] / actual[index][i][1])
            NDCG.append(NDCG_list)

        return NDCG

    def calculate_RR(self, actual: List[List[str]], predicted: List[List[str]]) -> list[float]:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        list[float]
            The Reciprocal Rank of the predicted results
        """
        RR = []
        # TODO: Calculate RR here
        for index, predict in enumerate(predicted):
            query = predict[0]
            predicted_ids = predict[1:]
            actual_ids = actual[index]
            for i, predicted_id in enumerate(predicted_ids):
                if predicted_id in actual_ids:
                    RR.append(1 / (i + 1))
        return RR

    def calculate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates the Mean Reciprocal Rank of the predicted results

        Parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        Returns
        -------
        float
            The MRR of the predicted results
        """
        # TODO: Calculate MRR here
        RR = self.calculate_RR(actual, predicted)
        MRR = 0.0
        if len(RR) > 0:
            MRR = sum(RR) / len(RR)
        return MRR

    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Prints the evaluation metrics

        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results

        """
        print(f"name = {self.name}")

        # TODO: Print the evaluation metrics

    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Use Wandb to log the evaluation metrics

        parameters
        ----------
        precision : float
            The precision of the predicted results
        recall : float
            The recall of the predicted results
        f1 : float
            The F1 score of the predicted results
        ap : float
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: float
            The Discounted Cumulative Gain of the predicted results
        ndcg : float
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: float
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results

        """

        # TODO: Log the evaluation metrics using Wandb

    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        call all functions to calculate evaluation metrics

        parameters
        ----------
        actual : List[List[str]]
            The actual results
        predicted : List[List[str]]
            The predicted results

        """

        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        dcg = self.calculate_DCG(actual, predicted)
        ndcg = self.calculate_NDCG(actual, predicted)
        rr = self.calculate_RR(actual, predicted)
        mrr = self.calculate_MRR(actual, predicted)

        # call print and viualize functions
        self.print_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
        self.log_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
