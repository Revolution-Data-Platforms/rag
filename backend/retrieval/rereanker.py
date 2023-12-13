
from sentence_transformers.cross_encoder import CrossEncoder
from scipy.special import expit

class Reranker:

    def __init__(self, rereanker_model='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.rereanker_model = rereanker_model
    
    def rerank(self, query, docs, k=8):
        
        cross_encoder = CrossEncoder(self.rereanker_model)
        ranked_Res = []
        for doc in docs:
            score = cross_encoder.predict([query, doc.page_content])
            if expit(score) > 0.5:
                ranked_Res.append(doc)

        if len(ranked_Res) > k:
            ranked_Res = ranked_Res[:k]

        return ranked_Res