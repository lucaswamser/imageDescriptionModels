import nltk


class Comparator(object):

    def __init__(self,ori,ref) -> None:
        self.ori = ori.split(" ")
        self.ref = ref.split(" ")
        self._calculate()

    def _calculate(self):
        BLEUscore = nltk.translate.bleu_score.sentence_bleu([self.ori], self.ref)
        self.score = BLEUscore

if __name__ == "__main__":

    print(Comparator("oii tudo", "oii tudo").score)