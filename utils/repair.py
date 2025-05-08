import json

def update_keys(a, b):
    updated_a = {}
    for key_a, value_a in a.items():
        found_match = False
        for key_b in b.keys():
            if key_b.startswith(key_a) or key_a.startswith(key_b):
                updated_a[key_b] = value_a
                found_match = True
                break
        
        if not found_match:
            updated_a[key_a] = value_a
    
    return updated_a

sdpath='/data/TextCenGen_evaluation/SD1.5/prompt_'
p='/data/TextCenGen_evaluation/dalle/prompt_'
dataset=['ptp','llm','ddb']
for d in dataset:
    path=p+d
    base_sum=json.load(open(sdpath+d+'/tmp/tmp.json'))
    base_clip_score_sum=base_sum['clip_score_sum']
    clip_score_sum=json.load(open(path+'/0_clip_score_sum.json'))
    saliency=json.load(open(path+'/0_saliency_score_sum.json'))
    variation=json.load(open(path+'/0_variation_score_sum.json'))
    newClip=update_keys(clip_score_sum,base_clip_score_sum)
    newSaliency=update_keys(saliency,base_clip_score_sum)
    newVariation=update_keys(variation,base_clip_score_sum)
    ###保存到新的文件
    with open(path+'/1_clip_score_sum.json', 'w',encoding='utf-8') as file:
        json.dump(newClip, file, indent=4)
    with open(path+'/1_saliency_score_sum.json', 'w',encoding='utf-8') as file:
        json.dump(newSaliency, file, indent=4)
    with open(path+'/1_variation_score_sum.json', 'w',encoding='utf-8') as file:
        json.dump(newVariation, file, indent=4)
