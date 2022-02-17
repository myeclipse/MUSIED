root='sum_first_two/document_level/split/'
paths=[
        # 'train_judge.json',
        # 'train_tel.json',
        # 'train_online.json',
        # 'dev_judge.json',
        # 'dev_tel.json',
        # 'dev_online.json',
        # 'test_judge.json',
        # 'test_tel.json',
        'test_online.json'
       ]
output_file='sum_first_two/sentence_level/split/test_online_sentence.json'

result=[]
for file_path in paths:
    with open(root+file_path, 'r', encoding='UTF-8') as f:
        data = json.load(f)
        for d in data:
            item=dict()
            id=d['id']

            triggers=d['events']
            sentences=d['content']
            set_ids=set()
            for idx,sentence in enumerate(sentences):
                for event in triggers:
                    e_sent_id=event['sent_id']
                    if e_sent_id==idx:
                        set_ids.add(idx)
                        trigger = {'text': event['trigger_word'],
                                   'start': event['offset'][0],
                                   'end': event['offset'][1],
                                   'event_type': event['type']}
                        if idx not in item:
                            item[idx]= dict()
                            item[idx]['id'] = id
                            item[idx]['sentence']=sentence
                            item[idx]['trigger']=[]
                        item[idx]['trigger'].append(trigger)
            for idx,sentence in enumerate(sentences):
                if idx in set_ids:
                    result.append(item[idx])
                else:
                    result.append({'id':id,'sentence':sentence,'trigger':[]})

with open(output_file,"w") as f:
    f.write(json.dumps(result,indent=4, ensure_ascii=False))
