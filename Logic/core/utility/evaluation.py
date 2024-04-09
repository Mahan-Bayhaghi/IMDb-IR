from typing import List
import numpy as np
import wandb
from statistics import mean

# *** NOTE ***
# each list is a list of strings
# we will assume first index is query and the rest are relevant movie ids retrieved
# please don't forget to log in to your wandb account :)

class Evaluation:

    def __init__(self, name: str):
        self.name = name
        # wandb.login()
        # wandb.init()

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
        print("*****" * 20)
        print(f" MAP measure over {num_queries} queries \t: {MAP} ")
        print(f" MRR measure over {num_queries} queries \t: {MRR} ")
        print("*****" * 20)
        for idx in range(num_queries):
            print(f" Evaluation over query <{queries[idx]}>")
            print(f" precision  \t: {precision[idx]}")
            print(f" recall     \t: {recall[idx]}")
            print(f" f1 measure \t: {f1[idx]}")
            print(f" AP measure \t: {AP[idx]}")
            print(f" RR measure \t: {RR[idx]}")
            print(f" DCG measure \t: {DCG[idx]}")
            print(f" NDCG measure \t: {NDCG[idx]}")
            print("----" * 25)

    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Use Wandb to log the evaluation metrics

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
        # TODO: Log the evaluation metrics using Wandb
        wandb.log({"precision": mean(precision),
                   "recall": mean(recall),
                   "f1": mean(f1),
                   "MAP": map,
                   "MRR": mrr
                   })

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

if __name__ == "__main__":
    evaluation = Evaluation("my query evaluation over 5 queries")
    # # top 15 search results from https://www.imdb.com/find/?q=spiderman&ref_=nv_sr_sm
    # # relevance scores are my prediction of relevance and they may not be correct !
    query_total_actual = [[("spiderman", 0),
                           ("tt0145487", 20), ("tt10872600", 19), ("tt0948470", 18), ("tt1872181", 17),
                           ("tt2705436", 16), ("tt0112175", 15), ("tt12122034", 14), ("tt0413300", 13),
                           ("tt4633694", 12), ("tt2250912", 11), ("tt6320628", 10), ("tt9362722", 9),
                           ("tt0316654", 8), ("tt0076975", 7), ("tt16360004", 6)],
                          [("batman", 0),
                           ("tt1877830", 19), ("tt0059968", 18), ("tt0372784", 17),
                           ("tt0103359", 16), ("tt0118688", 15), ("tt0103776", 14), ("tt0112462", 13),
                           ("tt2975590", 12), ("tt19850008", 11), ("tt0147746", 10), ("tt0398417", 9),
                           ("tt0035665", 8), ("tt4116284", 7), ("tt0060153", 6)],
                          [("matrix", 0),
                           ("tt10838180", 19), ("tt0234215", 18), ("tt0242653", 17),
                           ("tt0106062", 16), ("tt30849138", 15), ("tt0410519", 14), ("tt9847360", 13),
                           ("tt31998838", 12), ("tt30749809", 11), ("tt0365467", 10), ("tt0364888", 9),
                           ("tt11749868", 8), ("tt0303678", 7), ("tt0274085", 6)],
                          [("dune", 0),
                           ('tt15239678', 15), ('tt0087182', 14), ('tt1160419', 13), ('tt0142032', 12),
                           ('tt31378509', 11), ('tt0287839', 10), ('tt10466872', 9), ('tt1935156', 8),
                           ('tt15331462', 7), ('tt11835714', 6), ('tt12451788', 5), ('tt14450978', 4),
                           ('tt31613341', 3), ('tt0099474', 2), ('tt31613353', 1)],
                          [("harry potter", 0),
                           ('tt0241527', 15), ('tt0330373', 14), ('tt0304141', 13), ('tt0295297', 12),
                           ('tt1201607', 10), ('tt0373889', 10), ('tt0417741', 9), ('tt0926084', 8),
                           ('tt13918446', 7), ('tt16116174', 6), ('tt1756545', 5), ('tt15431326', 4),
                           ('tt3731688', 3), ('tt2335590', 2), ('tt7467820', 1)]
                          ]
    total_actual = []
    for i in range(5):
        total_actual.append([a[0] for a in query_total_actual[i]])
    print(f"query_total_actual is : {query_total_actual}")
    print(f"total actual is : {total_actual}")

    # 10 most related results using our search engine for each query
    query_predict_1 = [("spiderman", 0), ('tt16360004', 12.759454761359082), ('tt13904644', 10.944147360880505),
                       ('tt1872181', 10.908264454821978), ('tt4633694', 10.013595766313964),
                       ('tt2250912', 9.95070121054384), ('tt10872600', 9.497616884933521),
                       ('tt0316654', 9.11635489253859), ('tt6320628', 8.71744313614053),
                       ('tt0413300', 7.483820844565398), ('tt0145487', 7.199364442221147)]
    predicted_1 = [[q[0] for q in query_predict_1]]
    query_predict_2 = [('batman', 0), ('tt29010341', 12.979699754748001), ('tt0468569', 11.566983515144269),
                       ('tt0103776', 11.259678020678123), ('tt1345836', 11.100198112896305),
                       ('tt0112462', 10.675471656691693), ('tt1877830', 10.313280045511773),
                       ('tt0372784', 9.882344471537353), ('tt0118688', 9.622734651374413),
                       ('tt0096895', 8.432042457354264), ('tt0439572', 7.590291091179248)]
    predicted_2 = [[q[0] for q in query_predict_2]]
    query_predict_3 = [('matrix', 0), ('tt30749809', 12.86639177089274), ('tt0410519', 12.493512229172518),
                       ('tt0365467', 12.371862140186417), ('tt0088944', 12.237474507732829),
                       ('tt0234215', 10.562381464975013), ('tt10838180', 9.861749253520264),
                       ('tt0242653', 8.52053913157536), ('tt0133093', 6.489062382051385),
                       ('tt0121765', 3.1576633751632053), ('tt0121765', 3.1576633751632053)]
    predicted_3 = [[q[0] for q in query_predict_3]]
    query_predict_4 = [('dune', 0), ('tt0142032', 30.235850711154583), ('tt1160419', 28.287954355987345),
                       ('tt0287839', 22.67758698788), ('tt0069697', 13.634478833118132),
                       ('tt2251648', 13.049730660632648), ('tt1910516', 12.962026664436529),
                       ('tt4355704', 8.436244447170337), ('tt5523718', 8.42997931219993),
                       ('tt0120689', 8.03167119285517), ('tt0293069', 7.9748124187843334)]
    predicted_4 = [[q[0] for q in query_predict_4]]
    query_predict_5 = [('harry potter', 0), ('tt0304141', 0.9999962302078496), ('tt0330373', 0.9999339370586332),
                       ('tt0241527', 0.9996542561337151), ('tt0417741', 0.9919742957649751),
                       ('tt7467820', 0.9826791405668385), ('tt3183660', 0.9826791405668385),
                       ('tt7467858', 0.9826791405668385), ('tt0295297', 0.9778848085225866),
                       ('tt0373889', 0.9103419146333283), ('tt0086541', 0.7071067811865476)]
    predicted_5 = [[q[0] for q in query_predict_5]]
    query_total_predict = [query_predict_1, query_predict_2, query_predict_3, query_predict_4, query_predict_5]
    print(f"query_total_predict is : {query_total_predict}")
    total_predict = []
    for i in range(5):
        total_predict.append([a[0] for a in query_total_predict[i]])
    print(f"total predict is : {total_predict}")


    # query_actual_1 = [("spiderman", 26), ("tt0145487", 25), ("tt10872600", 24), ("tt0948470", 23), ("tt1872181", 22),
    #                   ("tt2705436", 21), ("tt0112175", 20), ("tt12122034", 19), ("tt0413300", 18), ("tt4633694", 17),
    #                   ("tt2250912", 16), ("tt6320628", 15), ("tt9362722", 14), ("tt0316654", 13), ("tt0076975", 12)]
    # actual_1 = [[a[0] for a in query_actual_1]]
    # # top 15 search results from https://www.imdb.com/find/?q=batman&ref_=nv_sr_sm
    # # relevance scores are my prediction of relevance and they may not be correct !
    # query_actual_2 = [("batman", 0), ("tt1877830", 19), ("tt0059968", 18), ("tt0372784", 17),
    #                   ("tt0103359", 16), ("tt0118688", 15), ("tt0103776", 14), ("tt0112462", 13),
    #                   ("tt2975590", 12), ("tt19850008", 11), ("tt0147746", 10), ("tt0398417", 9),
    #                   ("tt0035665", 8), ("tt4116284", 7), ("tt0060153", 6)]
    # actual_2 = [[a[0] for a in query_actual_2]]

    precision = evaluation.calculate_precision(total_actual, total_predict)
    recall = evaluation.calculate_recall(total_actual, total_predict)
    f1 = evaluation.calculate_F1(total_actual, total_predict)
    AP = evaluation.calculate_AP(total_actual, total_predict)
    MAP = evaluation.calculate_MAP(total_actual, total_predict)
    RR = evaluation.calculate_RR(total_actual, total_predict)
    MRR = evaluation.calculate_MRR(total_actual, total_predict)
    DCG = evaluation.calculate_DCG(query_total_actual, query_total_predict)
    NDCG = evaluation.calculate_NDCG(query_total_actual, query_total_predict)
    queries = [query_predict[0][0] for query_predict in query_total_predict]
    evaluation.print_evaluation(precision, recall, f1, AP, MAP, DCG, NDCG, RR, MRR, queries)
    # evaluation.log_evaluation(precision, recall, f1, AP, MAP, DCG, NDCG, RR, MRR)