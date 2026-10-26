#!/usr/bin/env python3
"""Trace original SIM-CoT checkpoint on existing paired graph prompts.
No training, generation, or correctness filtering. Native uploaded evaluation
recipe implies 10 slots, but checkpoint-specific training metadata is absent.
"""
import argparse
import hashlib
import importlib.util
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--groups',type=int,default=2)
    p.add_argument('--root',type=Path,default=Path('/mnt/home/user16/nihad'))
    p.add_argument('--dataset',type=Path)
    p.add_argument('--latents',type=int,default=10)
    args=p.parse_args()
    assert args.groups>0 and args.latents>0
    root=args.root
    dataset=args.dataset or root/'coconut/mentor_experiment/relevance_control_20261011_171002/diagnostic.json'
    checkpoint=root/'simcot_gpt2_coconut/checkpoint_28'
    wrapper=root/'SIM-CoT_review/Coconut/coconut.py'
    eval_config=root/'SIM-CoT_review/Coconut/args/gsm_simcot_eval.yaml'
    base_dir=root/'coconut/gpt2_model'
    assert sha(wrapper)=='5843656e7feaa03c5f1ec8a6ba843ac220565844394b12cc9df36faa50c64c01', 'Wrapper changed since review'
    manifest=json.loads((checkpoint.parent/'download_revision.json').read_text())
    assert manifest['revision']=='676c8a8aeb4afcef3a4fb98da9d539588d969340'
    assert checkpoint.stat().st_size==1458850422
    data=json.loads(dataset.read_text())
    groups=defaultdict(list)
    for row in data:groups[row['base_id']].append(row)
    ids=sorted(groups)[:args.groups]
    assert len(ids)==args.groups
    selected=[]
    expected={(r['condition'],r['query_index']) for r in groups[ids[0]]}
    assert len(expected)==10, 'Expected five conditions and two queries'
    assert all({q for c,q in expected if c==condition}=={0,1} for condition in {c for c,q in expected})
    for group in ids:
        rows=groups[group]
        assert len(rows)==10 and {(r['condition'],r['query_index']) for r in rows}==expected
        for r in rows:
            edges=dict(r['edges']);assert len(edges)==len(r['edges'])
            node=r['start']
            for _ in range(r['hops']):node=edges[node]
            assert node==r['answer']
        selected.extend(rows)
    import torch
    import transformers
    from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
    assert torch.cuda.is_available()
    torch.manual_seed(2026);torch.cuda.manual_seed_all(2026)
    spec=importlib.util.spec_from_file_location('reviewed_simcot_wrapper',wrapper)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tokenizer=AutoTokenizer.from_pretrained(str(base_dir),local_files_only=True)
    tokenizer.pad_token=tokenizer.eos_token
    for token in ('<|start-latent|>','<|end-latent|>','<|latent|>'):tokenizer.add_tokens(token)
    start,end,latent=[tokenizer.convert_tokens_to_ids(t) for t in ('<|start-latent|>','<|end-latent|>','<|latent|>')]
    assert (start,end,latent)==(50257,50258,50259)
    cfg=AutoConfig.from_pretrained(str(base_dir),local_files_only=True)
    assert cfg.n_layer==12 and cfg.n_embd==768 and cfg.vocab_size==50257
    base=AutoModelForCausalLM.from_config(cfg,attn_implementation='eager').float()
    # Auxiliary decoder retains original vocabulary, as in run.py.
    auxcfg=AutoConfig.from_pretrained(str(base_dir),local_files_only=True)
    auxiliary=AutoModelForCausalLM.from_config(auxcfg,attn_implementation='eager').float()
    base.resize_token_embeddings(len(tokenizer))
    options=SimpleNamespace(training_method='full',visualize=False)
    model=module.CoconutGPT_Same_Word_Embedding(base,auxiliary,tokenizer,latent,start,end,
        tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<<'),2,options)
    weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert set(k.split('.')[0] for k in weights)=={'base_causallm','expainable_llm','embedding'}
    assert torch.equal(weights['embedding.weight'],weights['base_causallm.transformer.wte.weight'])
    assert torch.equal(weights['base_causallm.lm_head.weight'],weights['base_causallm.transformer.wte.weight']), 'Unexpected tied weights'
    loaded=model.load_state_dict(weights,strict=True);del weights
    model.to('cuda');model.eval();auxiliary.eval()
    # model.eval() has a custom return value: do not assign it.
    print('STRICT FULL CHECKPOINT LOADING PASSED',flush=True)
    out=root/'coconut/mentor_experiment/mentor_state_comparison'/('simcot_relevance_control_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True,exist_ok=False)
    traces=[];max_feedback=max_block=max_instrumentation=0.0
    for index,row in enumerate(selected):
        tokens=tokenizer.encode(row['question']+'\n',add_special_tokens=True)+[start]+[latent]*args.latents+[end]
        inputs=torch.tensor([tokens],device='cuda');positions=[i for i,t in enumerate(tokens) if t==latent]
        assert len(positions)==args.latents and len(tokens)<=cfg.n_positions
        kw=dict(input_ids=inputs,attention_mask=torch.ones_like(inputs),labels=inputs.clone(),
                position_ids=torch.arange(len(tokens),device='cuda').unsqueeze(0))
        baseline=None
        if index==0:
            with torch.no_grad():
                plain=model(**kw)
                baseline=(plain.logits.detach().cpu(),plain.inputs_embeds.detach().cpu())
            del plain
        calls=[];block_outputs=[];handles=[]
        def before(mod,pargs,kwargs):
            calls.append({'positions':kwargs['position_ids'].detach().cpu(),
                          'inputs':kwargs['inputs_embeds'].detach().cpu()})
        def after(mod,pargs,kwargs,output):
            calls[-1]['hidden']=output.hidden_states[-1].detach().cpu()
        handles.extend([base.register_forward_pre_hook(before,with_kwargs=True),
                        base.register_forward_hook(after,with_kwargs=True)])
        def block_hook(layer):
            def capture(mod,pargs,output):
                tensor=output[0] if isinstance(output,tuple) else output
                block_outputs.append((len(calls)-1,layer,tensor.detach().cpu()))
            return capture
        for layer,block in enumerate(base.transformer.h):handles.append(block.register_forward_hook(block_hook(layer)))
        try:
            with torch.no_grad():output=model(**kw)
        finally:
            for handle in handles:handle.remove()
        assert torch.isfinite(output.logits).all()
        if baseline:
            err=max((baseline[0]-output.logits.detach().cpu()).abs().max().item(),
                    (baseline[1]-output.inputs_embeds.detach().cpu()).abs().max().item())
            max_instrumentation=max(max_instrumentation,err);assert err<=1e-6
        assert len(calls)==args.latents+1
        by_layer={(call,layer):tensor for call,layer,tensor in block_outputs}
        vectors=output.inputs_embeds[0,positions].detach().cpu();raw=[]
        for k,pos in enumerate(positions):
            src=(calls[k]['positions'][0]==pos-1).nonzero().flatten()
            dst=(calls[k+1]['positions'][0]==pos).nonzero().flatten()
            assert src.numel()==dst.numel()==1
            source=calls[k]['hidden'][0,src.item()]
            err=max((source-vectors[k]).abs().max().item(),
                    (source-calls[k+1]['inputs'][0,dst.item()]).abs().max().item())
            max_feedback=max(max_feedback,err);assert err<=1e-6
            blocks=torch.stack([by_layer[k,l][0,src.item()] for l in range(cfg.n_layer)])
            with torch.no_grad():normalized=base.transformer.ln_f(blocks[-1].to('cuda')).cpu()
            err=(normalized-source).abs().max().item()
            max_block=max(max_block,err);assert err<=1e-5
            raw.append(blocks)
        raw=torch.stack(raw)
        assert torch.isfinite(raw).all() and torch.isfinite(vectors).all()
        traces.append({'base_id':row['base_id'],'condition':row['condition'],'query_index':row['query_index'],
                       'vectors':vectors,'raw_block_states':raw,'input_ids':tokens})
        if (index+1)%8==0:print(f'Traced {index+1}/{len(selected)}',flush=True)
    summary={'model':'Original SIM-CoT GPT2 Coconut','checkpoint':str(checkpoint),
        'revision':manifest['revision'],'checkpoint_sha256':sha(checkpoint),
        'checkpoint_variant':'released SIM-CoT weights; no synthetic adaptation',
        'checkpoint_loaded_strictly':True,'missing_keys':len(loaded.missing_keys),'unexpected_keys':len(loaded.unexpected_keys),
        'base_groups':len(ids),'records':len(selected),'base_ids':ids,'dataset_sha256':sha(dataset),
        'latent_steps':args.latents,'transformer_blocks':cfg.n_layer,'raw_block_tensor_shape':list(raw.shape),
        'maximum_feedback_error':max_feedback,'maximum_final_state_error':max_block,
        'instrumentation_error':max_instrumentation,'instrumentation_check_records':1,
        'precision':'torch.float32','torch_version':torch.__version__,'transformers_version':transformers.__version__,
        'attention_backend':'eager','correctness_filter':False,'answer_generation':False,
        'auxiliary_decoder':'Strictly loaded; not called for explanation generation or auxiliary loss',
        'state_definition':'Raw GPT-2 block outputs at each injection source; final normalized injected vectors stored separately',
        'slot_setting_source':'Uploaded eval YAML: 2 tokens per stage, 5 maximum stages; checkpoint-specific training metadata unverified',
        'wrapper_sha256':sha(wrapper),'eval_yaml_sha256':sha(eval_config),'script_sha256':sha(__file__),
        'limitations':['Different original training histories; not an architecture-only or matched-compute experiment.',
                       'Dataset is a controlled synthetic diagnostic, outside the checkpoint arithmetic training domain.']}
    torch.save(traces,out/'latent_traces.pt')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'prompts.json').write_text(json.dumps(selected,indent=2))
    print(json.dumps(summary,indent=2));print('Saved:',out)

if __name__=='__main__':main()
