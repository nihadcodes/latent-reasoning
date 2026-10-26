import argparse,hashlib,json
from pathlib import Path
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--model',choices=['coconut','huginn'],required=True);args=p.parse_args()
DATA=Path('relevance_control_20261011_171002/diagnostic.json')
data=json.loads(DATA.read_text());assert len(data)==400
folder=Path('../gpt2_model') if args.model=='coconut' else Path('/mnt/home/user16/nihad/huginn_0125_torch24_compat')
tok=AutoTokenizer.from_pretrained(folder,local_files_only=True)
if args.model=='coconut':
 tok.add_tokens(['<|start-latent|>','<|end-latent|>','<|latent|>'],special_tokens=True)
 suffix=[tok.convert_tokens_to_ids('<|start-latent|>')]+[tok.convert_tokens_to_ids('<|latent|>')]*6+[tok.convert_tokens_to_ids('<|end-latent|>')]
else:assert tok.chat_template

def encode(r):
 if args.model=='coconut':return tok.encode(r['question']+'\n',add_special_tokens=True)+suffix
 prompt=tok.apply_chat_template([{'role':'user','content':r['question']}],tokenize=False,add_generation_prompt=True)
 return tok.encode(prompt,add_special_tokens=False)
lookup={(r['base_id'],r['condition'],r['query_index']):r for r in data};assert len(lookup)==400
encoded={k:encode(r) for k,r in lookup.items()};rows=[]
for b in sorted({r['base_id'] for r in data}):
 for q in (0,1):
  original=encoded[b,'original',q]
  for edit,n in [('path',2),('endpoint',1)]:
   details=[]
   for chain in (0,1):
    condition=f'{edit}_chain_{chain}';v=encoded[b,condition,q];same=len(original)==len(v)
    changes=sum(a!=c for a,c in zip(original,v)) if same else None
    details.append({'condition':condition,'length':len(v),'same_length':same,'changed_token_count':changes})
   rows.append({'base_id':b,'query_index':q,'edit_type':edit,'passed':all(r['same_length'] and r['changed_token_count']==n for r in details),'details':details})
summary={'model':args.model,'dataset':str(DATA.resolve()),'dataset_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'records':400,'comparisons':len(rows),'matched_comparisons':sum(r['passed'] for r in rows),'all_matched':all(r['passed'] for r in rows),'expected_changed_tokens':{'path':2,'endpoint':1},'format':'Same tokenizer and prompt construction as uploaded original tracing script'}
out=DATA.parent/f'{args.model}_token_audit.json';out.write_text(json.dumps({'summary':summary,'comparisons':rows},indent=2));print(json.dumps(summary,indent=2))
for r in [r for r in rows if not r['passed']][:3]:print('MISMATCH:',json.dumps(r))
print('Saved:',out.resolve())
