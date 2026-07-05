from nltk.translate.bleu_score import sentence_bleu
from w3lib.html import remove_tags
class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1  # 1-gram weights
        self.w2 = w2  # 2-grams weights
        self.w3 = w3  # 3-grams weights
        self.w4 = w4  # 4-grams weights

    def compute_score(self, real, predicted):
        score1 = []
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            score1.append(sentence_bleu([sent1], sent2, weights=(self.w1, self.w2, self.w3, self.w4)))
        return score1