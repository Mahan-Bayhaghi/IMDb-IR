from typing import List
import numpy as np
import wandb


# *** NOTE ***
# each list is a list of strings
# we will assume first index is query and the rest are relevant movie ids retrieved

class Evaluation:

    def __init__(self, name: str):
        self.name = name

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> list[float]:
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
        list[float]
            The precision of the predicted results
        """
        # TODO: Calculate precision here
        predict_precisions = []
        for index in range(len(predicted)):
            predict = predicted[index]
            query = predict[0]
            predicted_ids = predict[1:]
            intersec = set(predicted_ids).intersection(set(actual[index][1:]))
            predict_precision = 0.0
            if len(predicted_ids) > 0:
                predict_precision = len(intersec) / len(predicted_ids)
            predict_precisions.append(predict_precision)
        return predict_precisions

    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> list[float]:
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
        list[float]
            The recall of the predicted results
        """
        # TODO: Calculate recall here
        predict_recalls = []
        for index in range(len(predicted)):
            predict = predicted[index]
            query = predict[0]
            predicted_ids = predict[1:]
            intersec = set(predicted_ids).intersection(set(actual[index][1:]))
            predict_recall = 0.0
            if len(predicted_ids) > 0:
                predict_recall = len(intersec) / len(actual[index][1:])
            predict_recalls.append(predict_recall)
        return predict_recalls

    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> list[float]:
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
        float[float]
            The F1 score of the predicted results
        """
        # TODO: Calculate F1 here
        predict_f1s = []
        recall = self.calculate_recall(actual, predicted)
        precision = self.calculate_precision(actual, predicted)
        for index in range(len(predicted)):
            predict_f1 = 0.0
            if (precision[index] * recall[index]) > 0.0001:
                predict_f1 = (2 * recall[index] * precision[index]) / (recall[index] + precision[index])
            predict_f1s.append(predict_f1)
        return predict_f1s

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

        # print("predicted is ", predicted)
        # TODO: Calculate AP here
        for index, predict in enumerate(predicted):
            query = predict[0]
            predicted_ids = predict[1:]
            actual_ids = actual[index][1:]
            p_at_ks = []
            for idx in range(len(predicted_ids)):
                if predicted_ids[idx] in actual_ids:
                    shit = self.calculate_precision([actual[index]], [predict[1:][:idx + 1]])[0]
                    p_at_ks.append(shit)
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

    def calculate_DCG(self, actual: list[list[(str, int)]], predicted: list[list[(str, int)]]) -> list[list[float]]:
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
            actual_ids = [a[0] for a in actual[index][1:]]
            actual_rels = [a[1] for a in actual[index][1:]]
            query_DCG = []
            for i, predicted_id in enumerate(predicted_ids):
                if predicted_id in actual_ids:
                    if i == 0:
                        query_DCG.append(actual_rels[actual_ids.index(predicted_id)])
                    else:
                        DG = actual_rels[actual_ids.index(predicted_id)] / np.log2(i)
                        query_DCG.append(DG)
            for i in range(1, len(query_DCG)):
                query_DCG[i] += query_DCG[i - 1]
            DCG.append(query_DCG)
        return DCG

    def calculate_NDCG(self, actual: list[list[(str, int)]], predicted: list[list[(str, int)]]) -> list[list[float]]:
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

        for index, predict in enumerate(predicted):
            query = predict[0][0]
            predicted_movies = predict[1:]
            predicted_ids = [p[0] for p in predicted_movies]
            predicted_rels = [p[1] for p in predicted_movies]
            actual_ids = [a[0] for a in actual[index][1:]]
            actual_rels = [a[1] for a in actual[index][1:]]
            query_DCG = []
            ideal_DCG_ids = []
            for i, predicted_id in enumerate(predicted_ids):
                if predicted_id in actual_ids:
                    if i == 0:
                        ideal_DCG_ids.append(predicted_id)
                        query_DCG.append(actual_rels[actual_ids.index(predicted_id)])
                    else:
                        DG = actual_rels[actual_ids.index(predicted_id)] / np.log2(i)
                        query_DCG.append(DG)
                        ideal_DCG_ids.append(predicted_id)
            ideal_DCG_rels = [actual_rels[actual_ids.index(ideal_DCG_id)] for ideal_DCG_id in ideal_DCG_ids]
            ideal_DCG_rels.sort(reverse=True)
            ideal_CG = [ideal_DCG_rels[0]] + [(ideal_DCG_rels[idx] / np.log2(idx + 1)) for idx in
                                              range(1, len(ideal_DCG_rels))]
            for i in range(1, len(query_DCG)):
                ideal_CG[i] += ideal_CG[i - 1]
                query_DCG[i] += query_DCG[i - 1]
            query_DCG = [(query_DCG[idx] / ideal_CG[idx]) for idx in range(len(ideal_CG))]
            NDCG.append(query_DCG)
        return NDCG

        # for query_index, predict in enumerate(predicted):
        #     DCG = self.calculate_DCG([actual[query_index]], [predict])[0]
        #     print(f"DCG ---> {DCG}")
        #     intersect = set([a[0] for a in predict[1:]]).intersection([a[0] for a in actual[query_index][1:]])
        #     print(f"intersect ---> {intersect}")
        #     if len(intersect) != len(DCG):
        #         raise ValueError("Incorrect DCG !")
        #     else:
        #         print("all good")
        #
        #     ideal_ranking = []
        #     for actual_value in actual[query_index]:
        #         if actual_value[0] in intersect:
        #             ideal_ranking.append(actual_value[1])
        #     ideal_ranking.sort(reverse=True)
        #     ideal_ranking = ideal_ranking[0] + [(rank / np.log2(idx+1)) for idx, rank in enumerate(ideal_ranking)]
        #     print(f"ideal ranking : {ideal_ranking}")
        #     for i in range(1, len(ideal_ranking)):
        #         ideal_ranking[i] += ideal_ranking[i-1]
        #     for i in range(len(ideal_ranking)):
        #         DCG.append(DCG / ideal_ranking[i])
        #     NDCG.append(DCG)

        # TODO: Calculate NDCG here
        # DCG = self.calculate_DCG(actual, predicted)
        # print(f"DCG : {DCG}")
        # ideal_rankings = []
        # for actual_ranking in actual:
        #     tmp = actual_ranking[1:].copy()
        #     tmp.sort(key=lambda x: x[1], reverse=True)
        #     ideal_rankings.append(tmp)
        #
        # ideal_DCGs = []
        # for ranking_index, ranking in enumerate(ideal_rankings):
        #     intersected_ranking = []
        #     actual_rels = [a[1] for a in actual[ranking_index][1:]]
        #     print(f"acutual rels : {actual_rels}")
        #     ranking_DCG = [ranking[0][1]] + [(actual_rels[idx] / np.log2(idx + 1)) for idx in range(1, len(ranking))]
        #     for i in range(1, len(ranking_DCG)):
        #         ranking_DCG[i] += ranking_DCG[i - 1]
        #     ideal_DCGs.append(ranking_DCG)
        #
        # for index, DCG_list in enumerate(DCG):
        #     NDCG_list = []
        #     for i in range(len(DCG_list)):
        #         NDCG_list.append(DCG_list[i] / ideal_DCGs[index][i])
        #     NDCG.append(NDCG_list)
        # return NDCG

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
            actual_ids = actual[index][1:]
            for i, predicted_id in enumerate(predicted_ids):
                if predicted_id in actual_ids:
                    RR.append(1 / (i + 1))
                    break
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

    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr, queries):
        """
        Prints the evaluation metrics

        parameters
        ----------
        precision : list[float]
            The precision of the predicted results
        recall : list[float]
            The recall of the predicted results
        f1 : list[float]
            The F1 score of the predicted results
        ap : list[float]
            The Average Precision of the predicted results
        map : float
            The Mean Average Precision of the predicted results
        dcg: list[list[float]]
            The Discounted Cumulative Gain of the predicted results
        ndcg : list[list[float]]
            The Normalized Discounted Cumulative Gain of the predicted results
        rr: list[float]
            The Reciprocal Rank of the predicted results
        mrr : float
            The Mean Reciprocal Rank of the predicted results

        """
        print(f"name = {self.name}")
        # TODO: Print the evaluation metrics
        num_queries = len(queries)
        print("----" * 20)
        print(f" MAP measure over {num_queries} queries \t: {MAP} ")
        print(f" MRR measure over {num_queries} queries \t: {MRR} ")
        for idx in range(num_queries):
            print(f" Evaluation over query {queries[idx]}")
            print(f" precision  \t: {precision[idx]}")
            print(f" recall     \t: {recall[idx]}")
            print(f" f1 measure \t: {f1[idx]}")
            print(f" AP measure \t: {AP[idx]}")
            print(f" RR measure \t: {RR[idx]}")
            print(f" DCG measure \t: {DCG[idx]}")
            print(f" NDCG measure \t: {NDCG[idx]}")
            print("----" * 25)

    # def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
    def log_evaluation(self):
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
        # wandb.login()
        # wandb.init()
        # wandb.log({"acc": 0.5, "vel": 1.8})

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


# ------------ all searches done by Okapi25 with all weights set to 1 ------------
# test query 1 :
#
# predicted = [("spiderman", 0), ('tt16360004', 12.759454761359082), ('tt13904644', 10.944147360880505),
# ('tt1872181', 10.908264454821978), ('tt4633694', 10.013595766313964), ('tt2250912', 9.95070121054384),
# ('tt10872600', 9.497616884933521), ('tt0316654', 9.11635489253859), ('tt6320628', 8.71744313614053), ('tt0413300',
# 7.483820844565398), ('tt0145487', 7.199364442221147)]
#
# actual = [(("spiderman", 0), ("tt0145487", ?), ("tt10872600", ?), ("tt0948470", ?), ("tt1872181", ?), ("tt2705436",
# ?), ("tt0112175", ?), ("tt12122034", ?), ("tt0413300", ?), ("tt4633694", ?), ("tt2250912", ?), ("tt6320628", ?),
# ("tt9362722", ?), ("tt0316654", ?), ("tt0076975", ?), ("tt16360004", ?), ("tt6135682", ?)]


if __name__ == "__main__":
    evaluation = Evaluation("my query evaluation over 2 queries")
    query_predict_1 = [("spiderman", 0), ('tt16360004', 12.759454761359082), ('tt13904644', 10.944147360880505),
                       ('tt1872181', 10.908264454821978), ('tt4633694', 10.013595766313964),
                       ('tt2250912', 9.95070121054384), ('tt10872600', 9.497616884933521),
                       ('tt0316654', 9.11635489253859), ('tt6320628', 8.71744313614053),
                       ('tt0413300', 7.483820844565398), ('tt0145487', 7.199364442221147)]
    predicted_1 = [[q[0] for q in query_predict_1]]
    query_predict_2 = [("cholops", 0), ('tt16360004', 12.759454761359082), ('tt13904644', 10.944147360880505),
                       ('tt1872181', 10.908264454821978), ('tt4633694', 10.013595766313964),
                       ('tt2250912', 9.95070121054384), ('tt10872600', 9.497616884933521),
                       ('tt0316654', 9.11635489253859), ('tt6320628', 8.71744313614053),
                       ('tt0413300', 7.483820844565398), ('tt0145487', 7.199364442221147)]
    predicted_2 = [[q[0] for q in query_predict_2]]

    query_actual_1 = [("spiderman", 26), ("tt0145487", 25), ("tt10872600", 24), ("tt0948470", 23), ("tt1872181", 22),
                      ("tt2705436", 21), ("tt0112175", 20), ("tt12122034", 19), ("tt0413300", 18), ("tt4633694", 17),
                      ("tt2250912", 16), ("tt6320628", 15), ("tt9362722", 14), ("tt0316654", 13), ("tt0076975", 12),
                      ("tt16360004", 11), ("tt6135682", 10)]
    actual_1 = [[a[0] for a in query_actual_1]]
    query_actual_2 = [("cholops", 20), ("tt0145487", 19), ("tt10872600", 19), ("tt0948470", 19), ("tt1872181", 18),
                      ("tt2705436", 15), ("tt0112175", 15), ("tt12122034", 15), ("tt0413300", 15), ("tt4633694", 14),
                      ("tt2250912", 14), ("tt6320628", 14), ("tt9362722", 14), ("tt0316654", 14), ("tt0076975", 14),
                      ("tt16360004", 13), ("tt6135682", 10)]
    actual_2 = [[a[0] for a in query_actual_2]]

    total_actual = actual_1 + actual_2
    total_predict = predicted_1 + predicted_2
    query_total_actual = [query_actual_1, query_actual_2]
    query_total_predict = [query_predict_1, query_predict_2]

    precision = evaluation.calculate_precision(total_actual, total_predict)
    recall = evaluation.calculate_recall(total_actual, total_predict)
    f1 = evaluation.calculate_F1(total_actual, total_predict)
    AP = evaluation.calculate_AP(total_actual, total_predict)
    MAP = evaluation.calculate_MAP(total_actual, total_predict)
    RR = evaluation.calculate_RR(total_actual, total_predict)
    MRR = evaluation.calculate_MRR(total_actual, total_predict)
    DCG = evaluation.calculate_DCG(query_total_actual, query_total_predict)
    NDCG = evaluation.calculate_NDCG(query_total_actual, query_total_predict)

    evaluation.print_evaluation(precision, recall, f1, AP, MAP, DCG, NDCG, RR, MRR, [query_predict_1[0][0], query_predict_2[0][0]])
