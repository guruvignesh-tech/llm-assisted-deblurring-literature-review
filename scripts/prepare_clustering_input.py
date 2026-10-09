"""Prepare retained screening records; optionally replay explicit recorded resolutions."""
import argparse
import hashlib
from pathlib import Path
import pandas as pd

def prepare(input_path, output_path, resolutions=None):
    frame=pd.read_excel(input_path)
    required=['screen_id','Title','Abstract','decision','relevance','theme','api_status']
    missing=[c for c in required if c not in frame]
    if missing: raise ValueError(f'Missing screening columns: {missing}')
    if frame.screen_id.isna().any() or not frame.screen_id.is_unique:
        raise ValueError('screen_id must be unique and nonempty')
    if resolutions:
        ledger=pd.read_csv(resolutions)
        expected=hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
        required_ledger=['screen_id','decision','relevance','source_screening_sha256']
        if any(c not in ledger for c in required_ledger):
            raise ValueError('Resolution ledger missing required fields')
        if not ledger.screen_id.is_unique or ledger.screen_id.isna().any():
            raise ValueError('Resolution IDs must be unique and nonempty')
        if not ledger.source_screening_sha256.eq(expected).all():
            raise ValueError('Resolution ledger does not match this screening workbook')
        for row in ledger.itertuples(index=False):
            if isinstance(row.relevance,bool) or not float(row.relevance).is_integer() or not 0<=row.relevance<=10:
                raise ValueError('Invalid resolution relevance score')
            decision='KEEP' if row.relevance>=7 else ('MAYBE' if row.relevance>=5 else 'REJECT')
            if row.decision!=decision: raise ValueError('Resolution decision conflicts with score')
            mask=frame.screen_id.eq(row.screen_id)
            if mask.sum()!=1: raise ValueError('Resolution ID not present in screening workbook')
            if frame.loc[mask,'api_status'].eq('SUCCESS').any():
                raise ValueError('Resolution must not overwrite a successful API judgment')
            frame.loc[mask,['decision','relevance','api_status']]=[row.decision,int(row.relevance),'RECORDED_RESOLUTION']
            if frame.loc[mask,'theme'].isna().any(): frame.loc[mask,'theme']='Other Relevant'
    keep=frame.loc[frame.api_status.isin(['SUCCESS','RECORDED_RESOLUTION']) & frame.decision.isin(['KEEP','MAYBE'])].copy()
    if keep.empty: raise ValueError('No retained records')
    columns=['Title','Abstract','Author Keywords','Index Keywords']
    def text(row):
        return '\n'.join(f'{col}: '+('' if pd.isna(row.get(col)) else ' '.join(str(row[col]).split())) for col in columns)
    keep['embedding_text']=keep.apply(text,axis=1)
    target=Path(output_path)
    if target.exists(): raise FileExistsError(f'Refusing to overwrite {target}')
    temporary=target.with_suffix('.tmp.xlsx')
    with pd.ExcelWriter(temporary,engine='openpyxl') as writer:
        keep.to_excel(writer,sheet_name='Clustering Input',index=False)
    temporary.replace(target)
    return len(keep)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default='Gemma4_26B_FULL_3799_Results.xlsx')
    parser.add_argument('--output',default='Deblurring_Clustering_Input_1947.xlsx')
    parser.add_argument('--resolutions',help='Optional CSV of recorded decisions, bound to source workbook SHA256')
    args=parser.parse_args()
    print(f'Prepared {prepare(args.input,args.output,args.resolutions)} retained records')
