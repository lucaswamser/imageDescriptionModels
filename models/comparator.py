import nltk
from rouge import Rouge 
from statistics import mean

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



class ComparatorAll(object):

    def __init__(self) -> None:
        self.meteor_scores=[]
        self.bleu_scores=[]
        self.rouge1_scores=[]
        self.rouge2_scores=[]
        self.rougel_scores=[]
        self.rouge = Rouge()
        nltk.download('wordnet')
        nltk.download('omw-1.4')

    def _calculate_rouge(self,ori,ref):
        rouge1_score = 1000
        rouge2_score = 1000
        rougel_score = 1000
        for o in ori:
            scores = self.rouge.get_scores(o, ref)
            if (scores[0]["rouge-1"]["r"] < rouge1_score):
                rouge1_score = (scores[0]["rouge-1"]["r"])
            if (scores[0]["rouge-2"]["r"] < rouge2_score):
                rouge2_score = (scores[0]["rouge-2"]["r"])
            if (scores[0]["rouge-l"]["r"] < rougel_score):
                rougel_score = (scores[0]["rouge-l"]["r"])

        #scores = self.rouge.get_scores(ori, ref)
        self.rouge1_scores.append(rouge1_score)
        self.rouge2_scores.append(rouge2_score)
        self.rougel_scores.append(rougel_score)

    def _calculate_meteor(self,ori,ref):
        meteor_score = nltk.translate.meteor_score.meteor_score(ori,ref)
        self.meteor_scores.append(meteor_score)

    def _calculate_bleu(self,ori,ref):
        BLEUscore = nltk.translate.bleu_score.sentence_bleu(ori, ref)
        self.bleu_scores.append(BLEUscore)

    def add_comparation(self,ori,ref):
       ori_split = [o.split(" ") for o in ori]
       ref_split = ref.split(" ")
       self._calculate_bleu(ori_split,ref_split)
       self._calculate_meteor(ori_split,ref_split)
       self._calculate_rouge(ori,ref)

    def print_summary(self):
        print(f"Meteor Score {mean(self.meteor_scores)}")
        print(f"Bleu Score {mean(self.bleu_scores)}")
        print(f"Rouge-1 Score {mean(self.rouge1_scores)}")
        print(f"Route-2 Score {mean(self.rouge2_scores)}")
        print(f"Route-l Score {mean(self.rougel_scores)}")





if __name__ == "__main__":

    comparator_all = ComparatorAll()
    comparator_all.add_comparation(["hello i do something"], "hello i am lucas")
    comparator_all.add_comparation(["hello","sdasds"], "hello i am lucas")
    comparator_all.print_summary()


    #print(Comparator("oii tudo", "oii tudo").score)