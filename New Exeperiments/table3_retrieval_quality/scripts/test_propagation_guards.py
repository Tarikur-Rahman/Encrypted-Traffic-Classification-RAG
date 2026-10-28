"""Synthetic in-memory fixtures only; no experiment relevance labels are assigned."""
import unittest
from propagate_human_labels import derive_unique,propagate,UNIQUE_HEADERS

class PropagationChecks(unittest.TestCase):
    def setUp(self):
        base=dict(query_id='SYNTHETIC-Q1',dataset='fixture',predicted_class='fixture',structured_query='Synthetic query',chunk_id='SYNTHETIC-C1',document_id='fixture',chunk_text='Synthetic text',retrieval_method='Method A',rank='1',relevance='')
        self.original=[base,dict(base,retrieval_method='Method B',rank='3'),dict(base,query_id='SYNTHETIC-Q2')]
        self.unique,self.mapping=derive_unique(self.original)
    def test_key_and_manual_copy(self):
        self.assertEqual(len(self.unique),2)
        self.assertEqual([r['occurrence_count'] for r in self.unique],[2,1])
        fixture=[{k:str(v) for k,v in r.items()} for r in self.unique]
        fixture[0]['relevance']='0';fixture[1]['relevance']='1'
        result=propagate(self.original,UNIQUE_HEADERS,list(reversed(fixture)),self.mapping)
        self.assertEqual([r['relevance'] for r in result],['0','0','1'])
        self.assertEqual([r['rank'] for r in result],['1','3','1'])
    def test_blank_invalid_missing_and_tampering(self):
        for value in ['',None,'2','0.5','=1']:
            fixture=[dict(r,relevance='0') for r in self.unique];fixture[0]['relevance']=value
            with self.assertRaises(ValueError):propagate(self.original,UNIQUE_HEADERS,fixture,self.mapping)
        fixture=[dict(r,relevance='0') for r in self.unique]
        with self.assertRaises(ValueError):propagate(self.original,UNIQUE_HEADERS,fixture[:-1],self.mapping)
        with self.assertRaises(ValueError):propagate(self.original,UNIQUE_HEADERS,[fixture[0],fixture[0]],self.mapping)
        fixture[0]['chunk_text']='changed'
        with self.assertRaises(ValueError):propagate(self.original,UNIQUE_HEADERS,fixture,self.mapping)
    def test_map_and_source_conflicts(self):
        fixture=[dict(r,relevance='0') for r in self.unique]
        with self.assertRaises(ValueError):propagate(self.original,UNIQUE_HEADERS,fixture,list(reversed(self.mapping)))
        bad=[dict(r) for r in self.original];bad[1]['chunk_text']='conflicting duplicate'
        with self.assertRaises(ValueError):derive_unique(bad)

if __name__=='__main__':unittest.main(verbosity=2)
