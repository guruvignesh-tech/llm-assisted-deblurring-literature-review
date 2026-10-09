"""Offline regression tests: run with python -m unittest discover -s tests -v."""
import ast
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
import hashlib
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from checkpoint_guard import bind_checkpoint
from prepare_clustering_input import prepare

def functions(name, names, constants=()):
    tree=ast.parse((ROOT/'scripts'/name).read_text(encoding='utf-8-sig'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names or isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in constants for t in n.targets)]
    namespace={'pd':pd,'re':re,'json':json}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),name,'exec'),namespace)
    return namespace

class OfflineTests(unittest.TestCase):
    def test_handoff_excludes_errors_and_requires_matching_resolution_source(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            tmp=Path(tmp)
            source=tmp/'screen.xlsx'
            pd.DataFrame({
                'screen_id':['S1','S2','S3'], 'Title':['Relevant','Error','Excluded'],
                'Abstract':['Evidence',None,'Unrelated'], 'decision':['KEEP',None,'REJECT'],
                'relevance':[8,None,2], 'theme':['Survey/Review',None,'Other Relevant'],
                'api_status':['SUCCESS','PARSE_ERROR','SUCCESS']
            }).to_excel(source,index=False)
            self.assertEqual(prepare(source,tmp/'base.xlsx'),1)
            ledger=tmp/'resolutions.csv'
            pd.DataFrame([{'screen_id':'S2','decision':'MAYBE','relevance':6,
                           'source_screening_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}]).to_csv(ledger,index=False)
            self.assertEqual(prepare(source,tmp/'resolved.xlsx',ledger),2)
            rows=pd.read_excel(tmp/'resolved.xlsx',sheet_name='Clustering Input')
            self.assertEqual(rows.screen_id.tolist(),['S1','S2'])
            self.assertFalse(rows.embedding_text.str.contains('nan').any())
            with self.assertRaises(FileExistsError): prepare(source,tmp/'resolved.xlsx',ledger)
            source.write_bytes(source.read_bytes()+b'changed')
            with self.assertRaises(ValueError): prepare(source,tmp/'bad.xlsx',ledger)

    def test_score_boundaries_and_bad_scores(self):
        f=functions('gemma4_26b_RESUME_V2_UNICODE_FIX.py',{'clean','parse_obj','normalize'},{'ALLOWED_THEMES'})
        for score,decision in [(0,'REJECT'),(4,'REJECT'),(5,'MAYBE'),(6,'MAYBE'),(7,'KEEP'),(10,'KEEP')]:
            self.assertEqual(f['normalize']({'relevance':score,'theme':'Survey/Review','reason':'Evidence','methods':[]})[1],decision)
        for bad in [-1,11,7.9,True,None,'8']:
            with self.assertRaises((ValueError,TypeError)):
                f['normalize']({'relevance':bad})
        parsed=f['parse_obj']('```json\n{"relevance":8,"reason":"α"}\n```')
        self.assertEqual(parsed['reason'],'α')
        self.assertIsNone(f['parse_obj']('not json'))

    def test_theme_json_nested_braces_and_escaped_quotes(self):
        f=functions('theme_summary_FINAL_V2_SAFE.py',{'extract_json_object','validate_result'},{'REQUIRED_KEYS'})
        value={'theme':'quoted " { }','thesis_relevance':'Evidence'}
        value.update({k:[] for k in f['REQUIRED_KEYS'] if k not in value})
        text='Preface\n```json\n'+json.dumps(value)+'\n```\nTrailing commentary'
        self.assertEqual(f['validate_result'](f['extract_json_object'](text)),value)
        for text in ['', '{"incomplete":', '[]']:
            with self.assertRaises(ValueError): f['extract_json_object'](text)
        for bad in [{},dict(value,theme=[]),dict(value,research_gaps=[4])]:
            with self.assertRaises(ValueError): f['validate_result'](bad)

    def test_representatives_prioritize_probability_then_relevance(self):
        f=functions('theme_summary_FINAL_V2_SAFE.py',{'representative_subset'})
        table=pd.DataFrame({'screen_id':['a','b','c'],'cluster_probability':[.9,.9,.2],'relevance':[5,9,10]})
        self.assertEqual(f['representative_subset'](table,2).screen_id.tolist(),['b','a'])

    def test_checkpoint_resume_rejects_changed_inputs_or_scripts(self):
        before=Path.cwd()
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            try:
                os.chdir(tmp)
                Path('input.xlsx').write_bytes(b'original input')
                Path('stage.py').write_text('original script')
                bind_checkpoint('test','input.xlsx','stage.py',['output.xlsx'])
                Path('output.xlsx').write_bytes(b'checkpoint')
                bind_checkpoint('test','input.xlsx','stage.py',['output.xlsx'])
                Path('input.xlsx').write_bytes(b'reordered input')
                with self.assertRaises(ValueError): bind_checkpoint('test','input.xlsx','stage.py',['output.xlsx'])
                Path('input.xlsx').write_bytes(b'original input')
                Path('stage.py').write_text('changed prompt/configuration')
                with self.assertRaises(ValueError): bind_checkpoint('test','input.xlsx','stage.py',['output.xlsx'])
                with self.assertRaises(ValueError): bind_checkpoint('unverified','input.xlsx','stage.py',['output.xlsx'])
            finally: os.chdir(before)

if __name__=='__main__': unittest.main()
