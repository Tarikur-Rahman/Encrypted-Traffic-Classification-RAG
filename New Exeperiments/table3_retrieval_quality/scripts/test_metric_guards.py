"""Synthetic in-memory arithmetic/validation checks; never label real annotation rows."""
import math,unittest
from calculate_table3_metrics import HEADERS,METHODS,validate,query_metrics

class MetricChecks(unittest.TestCase):
    def test_known_rank_weights(self):
        value=query_metrics([0,1,0,0,1])
        expected=(1/math.log2(3)+1/math.log2(6))/(1+1/math.log2(3))
        self.assertEqual(value['precision_at_5'],.4)
        self.assertAlmostEqual(value['ndcg_at_5'],expected)
    def test_zero_and_ideal_conventions(self):
        self.assertEqual(query_metrics([0]*5)['ndcg_at_5'],0)
        self.assertEqual(query_metrics([1]*5)['ndcg_at_5'],1)
        self.assertEqual(query_metrics([1,1,0,0,0])['ndcg_at_5'],1)
    def setUp(self):
        self.expected=[]
        for query in range(150):
            for method in METHODS:
                for rank in range(1,6):self.expected.append(dict(query_id=f'SYNTHETIC-{query}',dataset='test-fixture',predicted_class='fictional',structured_query='Synthetic test only',retrieval_method=method,rank=str(rank),chunk_id=f'fixture-{rank}',document_id='fixture',chunk_text='Synthetic unit-test text',relevance=''))
        self.labeled=[dict(r,relevance='0') for r in self.expected]
        self.mapping=dict(zip(['Method A','Method B','Method C'],METHODS))
    def test_valid_unblinded_and_blinded_order(self):
        self.assertEqual(len(validate(HEADERS,self.labeled,self.expected,self.mapping)),2250)
        inverse={v:k for k,v in self.mapping.items()}
        blind=[dict(r,retrieval_method=inverse[r['retrieval_method']]) for r in reversed(self.labeled)]
        self.assertEqual(len(validate(HEADERS,blind,self.expected,self.mapping)),2250)
    def test_blank_invalid_and_missing(self):
        for label in ['',None,'2','-1','0.5','yes','=1','1.0']:
            bad=[dict(r) for r in self.labeled];bad[0]['relevance']=label
            with self.assertRaises(ValueError):validate(HEADERS,bad,self.expected,self.mapping)
        with self.assertRaises(ValueError):validate(HEADERS,self.labeled[:-1],self.expected,self.mapping)
    def test_tampering_duplicate_and_wrong_rank(self):
        for field,value in [('chunk_text','changed'),('query_id','foreign'),('rank','6'),('retrieval_method','other')]:
            bad=[dict(r) for r in self.labeled];bad[0][field]=value
            with self.assertRaises(ValueError):validate(HEADERS,bad,self.expected,self.mapping)
        bad=[dict(r) for r in self.labeled];bad[1]=bad[0]
        with self.assertRaises(ValueError):validate(HEADERS,bad,self.expected,self.mapping)

if __name__=='__main__':unittest.main(verbosity=2)
