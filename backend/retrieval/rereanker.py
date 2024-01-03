
from sentence_transformers.cross_encoder import CrossEncoder
from scipy.special import expit

class Reranker:

    def __init__(self, rereanker_model='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.rereanker_model = rereanker_model
    
    def rerank(self, query, docs, k=8):
        
        cross_encoder = CrossEncoder(self.rereanker_model)
        ranked_Res = {}
        for doc in docs:
            if doc.metadata['header'] not in ranked_Res:
                ranked_Res[doc.metadata['header']] = 0
                
            score = cross_encoder.predict([query, doc.page_content])
            if ranked_Res[doc.metadata['header']] < expit(score):
                ranked_Res[doc.metadata['header']] = expit(score)
                
        if 'Table of Contents ' in ranked_Res.keys():
            ranked_Res.pop('Table of Contents ')
            
        ranked_Res = dict(sorted(ranked_Res.items(), key=lambda item: item[1], reverse=True))

        max_ele = max(ranked_Res.values())
        
        res = []
        for doc in docs:
            
            if doc.metadata['header'] != 'Table of Contents ' and \
            ranked_Res[doc.metadata['header']] == max_ele:
                res.append(doc)
        
        return res