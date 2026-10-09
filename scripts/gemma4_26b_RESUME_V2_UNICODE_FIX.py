import os, re, json, time, random, subprocess, sys, shutil
from pathlib import Path
import pandas as pd
from google import genai
from google.genai.errors import APIError

MODEL = 'gemma-4-26b-a4b-it'
INPUT_FILE = 'Deblurring_Literature_Master_Deduplicated.xlsx'
PILOT_FILE = 'Gemma4_26B_10_Pilot_Selected.xlsx'
OUTPUT_FILE = 'Gemma4_26B_10_Pilot_Results.xlsx'
ERROR_FILE = 'Gemma4_26B_10_Pilot_Errors.csv'
N_PILOT = 10
MAX_RETRIES = 3
SUCCESS_DELAY = 10
RANDOM_SEED = 42

TITLE_COL='Title'; ABSTRACT_COL='Abstract'; KEYWORDS_COL='Author Keywords'; INDEX_KW_COL='Index Keywords'; QUERY_COL='Queries'
ALLOWED_THEMES = {
 'Foundational/Classical Deblurring','Deep-Learning Deblurring','Motion-Aware/Sensor-Assisted Deblurring',
 'Task-Driven/Perception','Robotics/UAV Application','Dataset/Benchmark','Survey/Review','Other Relevant'
}

def clean(x):
    if pd.isna(x): return ''
    return re.sub(r'\s+', ' ', str(x)).strip()

def load_master():
    p=Path(INPUT_FILE)
    if not p.exists():
        m=list(Path('.').glob('Deblurring_Literature_Master_Deduplicated*.xlsx'))
        if not m: raise FileNotFoundError(f'{INPUT_FILE} not found in this Jupyter folder.')
        p=m[0]
    xls=pd.ExcelFile(p)
    for sh in xls.sheet_names:
        d=pd.read_excel(p, sheet_name=sh)
        if TITLE_COL in d.columns and ABSTRACT_COL in d.columns:
            return p, sh, d
    raise ValueError('No worksheet containing both Title and Abstract was found.')

def has_q(s, q):
    s=clean(s)
    return bool(re.search(rf'(?<!\d)Q{q}(?!\d)', s, flags=re.I))

def choose_pilot(df):
    # Deliberately mix central and noisy provenance. One unique paper per slot.
    targets=[5,5,2,2,6,6,1,3,4,4]
    used=set(); rows=[]
    rng=random.Random(RANDOM_SEED)
    for q in targets:
        cand=[i for i in df.index if i not in used and has_q(df.at[i,QUERY_COL] if QUERY_COL in df.columns else '', q)]
        if cand:
            idx=rng.choice(cand); used.add(idx); rows.append(idx)
    if len(rows)<N_PILOT:
        rem=[i for i in df.index if i not in used]
        rng.shuffle(rem); rows += rem[:N_PILOT-len(rows)]
    out=df.loc[rows[:N_PILOT]].copy().reset_index(drop=True)
    out.insert(0,'pilot_id',[f'P{i+1:03d}' for i in range(len(out))])
    return out

def make_prompt(r):
    return f'''You are screening ONE scientific paper for an M.Tech thesis literature review.

THESIS TOPIC:
Motion-Aware, Task-Driven Deblurring for Robust Visual Perception in Robotic Systems.

PRIMARY QUESTION:
Can robotic motion information such as gyro/IMU/kinematics improve motion deblurring and task-relevant visual perception compared with image-only blind deblurring, and what restoration-perception-computational trade-offs arise?

PAPER METADATA:
Title: {clean(r.get(TITLE_COL,''))}
Abstract: {clean(r.get(ABSTRACT_COL,''))}
Author Keywords: {clean(r.get(KEYWORDS_COL,''))}
Index Keywords: {clean(r.get(INDEX_KW_COL,''))}
Scopus Query Provenance: {clean(r.get(QUERY_COL,''))}

Evaluate this paper independently. Query provenance is only a clue; never use it as proof of relevance.

RELEVANT STREAMS:
1. Motion/image deblurring methods, foundational/classical or deep-learning.
2. Gyro/IMU/inertial/motion-prior/sensor-assisted deblurring.
3. Deblurring evaluated through downstream perception: detection, recognition, localization, tracking, OCR, feature detection/matching, segmentation, etc.
4. Deblurring for robotics, UAVs, manipulators, autonomous/mobile systems, or robotic vision.
5. Motion-deblurring datasets, benchmarks, evaluation protocols, surveys/reviews.
A strong blind-deblurring baseline can be relevant even without robotics or sensors.

SCORE LOW/REJECT when deblurring is incidental; a generic SLAM/VIO/navigation/perception paper merely mentions blur; motion blur is only augmentation; or the work is mainly another restoration/application domain without a meaningful deblurring contribution.

SCORING: 0-2 unrelated; 3-4 weak/incidental; 5-6 potentially useful/adjacent/insufficient information; 7-8 directly relevant; 9-10 highly central.
DECISION: KEEP=7-10; MAYBE=5-6; REJECT=0-4.
THEME choose exactly one: Foundational/Classical Deblurring; Deep-Learning Deblurring; Motion-Aware/Sensor-Assisted Deblurring; Task-Driven/Perception; Robotics/UAV Application; Dataset/Benchmark; Survey/Review; Other Relevant.

Return ONLY valid JSON, no markdown and no extra prose:
{{"relevance":8,"decision":"KEEP","theme":"Motion-Aware/Sensor-Assisted Deblurring","methods":["IMU"],"reason":"One concise sentence."}}'''

def parse_obj(txt):
    txt=(txt or '').strip()
    txt=re.sub(r'^```(?:json)?\s*','',txt,flags=re.I); txt=re.sub(r'\s*```$','',txt)
    try: return json.loads(txt)
    except Exception:
        m=re.search(r'\{.*\}',txt,flags=re.S)
        if not m: return None
        try: return json.loads(m.group(0))
        except Exception: return None

def normalize(obj):
    if not isinstance(obj,dict): raise ValueError('JSON object not returned')
    value=obj.get('relevance')
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not float(value).is_integer():
        raise ValueError('relevance must be an integer from 0 to 10')
    score=int(value)
    if not 0<=score<=10: raise ValueError('relevance outside 0-10')
    decision='KEEP' if score>=7 else ('MAYBE' if score>=5 else 'REJECT')
    theme=clean(obj.get('theme'))
    if theme not in ALLOWED_THEMES: theme='Other Relevant'
    methods=obj.get('methods',[])
    if not isinstance(methods,list): methods=[clean(methods)] if clean(methods) else ['None']
    reason=clean(obj.get('reason'))
    return score,decision,theme,methods,reason

def call(client_unused, prompt):
    """
    Hard-timeout API call.
    Each request runs in a child Python process so a hung network/API call can
    be forcibly terminated without freezing the Jupyter kernel.
    """
    child_code = r'''
import os, sys, json, base64
from google import genai
payload = json.loads(base64.b64decode(sys.stdin.read().strip()).decode('utf-8'))
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
try:
    response = client.models.generate_content(
        model=payload["model"],
        contents=payload["prompt"]
    )
    print(json.dumps({"ok": True, "text": response.text or ""}, ensure_ascii=True))
except Exception as e:
    code = getattr(e, "code", None)
    print(json.dumps({"ok": False, "code": code, "error": f"{type(e).__name__}: {e}"}, ensure_ascii=True))
'''

    last = ''
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            import base64
            payload_ascii = base64.b64encode(
                json.dumps({'model': MODEL, 'prompt': prompt}, ensure_ascii=False).encode('utf-8')
            ).decode('ascii')
            child_env = os.environ.copy()
            child_env['PYTHONUTF8'] = '1'
            child_env['PYTHONIOENCODING'] = 'utf-8'
            cp = subprocess.run(
                [sys.executable, '-c', child_code],
                input=payload_ascii,
                text=True,
                encoding='utf-8',
                errors='replace',
                capture_output=True,
                timeout=120,
                env=child_env,
            )
            out = (cp.stdout or '').strip().splitlines()
            if not out:
                last = f'Child process returned no JSON. stderr: {(cp.stderr or "")[:300]}'
                print(f'    {last}', flush=True)
                time.sleep(5 * attempt)
                continue
            try:
                msg = json.loads(out[-1])
            except Exception:
                last = f'Invalid child response: {(cp.stdout or "")[-500:]}'
                print(f'    {last}', flush=True)
                time.sleep(5 * attempt)
                continue

            if msg.get('ok'):
                return msg.get('text', ''), 'SUCCESS'

            code = msg.get('code')
            last = f'APIError {code}: {msg.get("error", "unknown API error")}'
            print(f'    API {code} attempt {attempt}/{MAX_RETRIES}: {last[:240]}', flush=True)
            if code in (500, 502, 503, 504):
                time.sleep(5 * attempt)
            elif code == 429:
                time.sleep(30 * attempt)
            else:
                break

        except subprocess.TimeoutExpired:
            last = 'TIMEOUT: API request exceeded 120 seconds and was forcibly terminated.'
            print(f'    TIMEOUT attempt {attempt}/{MAX_RETRIES}: killed after 120s.', flush=True)
            # One retry only after a timeout; do not let one paper consume many minutes.
            if attempt >= 2:
                break
            time.sleep(10)
        except Exception as e:
            last = f'{type(e).__name__}: {e}'
            print(f'    {last[:240]}', flush=True)
            time.sleep(5 * attempt)

    return last, 'API_ERROR'

def atomic_excel(df,path):
    tmp=Path(str(path)+'.tmp.xlsx'); df.to_excel(tmp,index=False); os.replace(tmp,path)

p,sh,master=load_master()
from checkpoint_guard import bind_checkpoint
bind_checkpoint('screening', p, __file__, ['Gemma4_26B_FULL_3799_Results.xlsx'])
print('=== GEMMA 4 26B — SAFE RESUME V2 (UNICODE FIX + HARD TIMEOUT) ===',flush=True)
print(f'Model: {MODEL}',flush=True)
print(f'Input: {p} | sheet: {sh} | master papers: {len(master):,}',flush=True)
print('Hard timeout: 120s/request; max 2 timeout attempts; checkpoint after every paper.', flush=True)
if len(master)!=3799:
    print(f'WARNING: expected 3,799 papers; actual file has {len(master):,}.',flush=True)
if 'GEMINI_API_KEY' not in os.environ:
    raise RuntimeError('GEMINI_API_KEY is not loaded in this kernel.')

OUTPUT_FILE = 'Gemma4_26B_FULL_3799_Results.xlsx'
ERROR_FILE = 'Gemma4_26B_FULL_3799_Errors.csv'
SUCCESS_DELAY = 10

# Stable IDs tied to master row order. Never remove source rows.
df=master.copy().reset_index(drop=True)
df.insert(0,'screen_id',[f'S{i+1:04d}' for i in range(len(df))])
result_cols=['relevance','decision','theme','methods','reason','api_status','api_error']
for c in result_cols:
    df[c]=pd.Series([None]*len(df),dtype='object')

# Resume successful rows exactly as saved. API_ERROR/PARSE_ERROR rows are intentionally retried, including any Windows Unicode transport failures.
if Path(OUTPUT_FILE).exists():
    backup = Path('Gemma4_26B_FULL_3799_Results_BACKUP_before_unicode_fix.xlsx')
    if not backup.exists():
        shutil.copy2(OUTPUT_FILE, backup)
        print(f'Safety backup created: {backup}', flush=True)

    old=pd.read_excel(OUTPUT_FILE)
    if 'screen_id' in old.columns:
        if old['screen_id'].isna().any() or old['screen_id'].duplicated().any():
            raise ValueError('Checkpoint screen_id must be unique and nonempty')
        old=old.set_index('screen_id')
        loaded_attempted=0
        for i,r in df.iterrows():
            sid=r['screen_id']
            if sid in old.index and clean(old.at[sid,'api_status']) == 'SUCCESS':
                for c in result_cols:
                    if c in old.columns: df.at[i,c]=old.at[sid,c]
                loaded_attempted+=1
        print(f'Resume: {loaded_attempted:,} previously attempted papers loaded from checkpoint.',flush=True)
        print(df['api_status'].fillna('NOT_RUN').value_counts().to_string(), flush=True)

client=None  # API calls run in timeout-protected child processes
errors=[]
N=len(df)
for i,r in df.iterrows():
    sid=r['screen_id']
    if clean(r.get('api_status')) == 'SUCCESS':
        continue
    print(f'[{i+1:04d}/{N}] {sid}: {clean(r.get(TITLE_COL))[:105]}',flush=True)
    raw,status=call(client,make_prompt(r))
    if status=='SUCCESS':
        try:
            score,decision,theme,methods,reason=normalize(parse_obj(raw))
            df.at[i,'relevance']=int(score); df.at[i,'decision']=str(decision); df.at[i,'theme']=str(theme)
            df.at[i,'methods']='; '.join(map(str,methods)); df.at[i,'reason']=reason
            df.at[i,'api_status']='SUCCESS'; df.at[i,'api_error']=''
            print(f'    -> {decision} | {score}/10 | {theme}',flush=True)
        except Exception as e:
            df.at[i,'api_status']='PARSE_ERROR'; df.at[i,'api_error']=f'{type(e).__name__}: {e}'
            errors.append({'screen_id':sid,'status':'PARSE_ERROR','error':df.at[i,'api_error']})
            print(f'    -> PARSE_ERROR: {e}; saved and moving on.',flush=True)
    else:
        df.at[i,'api_status']='API_ERROR'; df.at[i,'api_error']=raw
        errors.append({'screen_id':sid,'status':'API_ERROR','error':raw})
        print('    -> API_ERROR; saved and moving on.',flush=True)

    # Durable checkpoint after every attempted paper.
    atomic_excel(df,OUTPUT_FILE)
    if errors: pd.DataFrame(errors).to_csv(ERROR_FILE,index=False,encoding='utf-8-sig')
    if df.at[i,'api_status']=='SUCCESS': time.sleep(SUCCESS_DELAY)

atomic_excel(df,OUTPUT_FILE)
failed=df[df.api_status!='SUCCESS'].copy()
if len(failed):
    failed.to_excel('Gemma4_26B_FULL_3799_Needs_Retry.xlsx',index=False)

print('\n=== FULL SCREENING PASS COMPLETE ===',flush=True)
print(df['api_status'].fillna('NOT_RUN').value_counts().to_string(),flush=True)
print('\nDecisions among successful papers:',flush=True)
print(df.loc[df.api_status=='SUCCESS','decision'].value_counts().to_string(),flush=True)
print(f'\nResults: {OUTPUT_FILE}',flush=True)
if len(failed): print(f'Needs retry: {len(failed):,} -> Gemma4_26B_FULL_3799_Needs_Retry.xlsx',flush=True)
else: print('All papers screened successfully.',flush=True)
