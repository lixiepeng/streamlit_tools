import json
import random
import re

import numpy as np
import pandas as pd
import requests
from nested_dict import nested_dict
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report as sequence_report
from sklearn.metrics import confusion_matrix


def diff_pred_gold_ner(nlp, docs_golds, output):
    '''
    print FP/FN
    '''
    diff = []

    def pretty_spacy_ner_tuple(doc, spacy_ner_tuples):
        res = []
        for ent in spacy_ner_tuples:
            res.append((doc[ent[0]:ent[1]], ent[0], ent[1], ent[2]))
        return res

    def doc2spacy_ner_tuple(doc):
        res = []
        for ent in doc.ents:
            res.append((ent.start_char, ent.end_char, ent.label_))
        return res
    for doc, gold in docs_golds:
        doc_pred = nlp(doc) if isinstance(doc, str) else doc
        ner_pred = {
            'predictions': pretty_spacy_ner_tuple(doc, doc2spacy_ner_tuple(doc_pred))
        }
        gold_entities = pretty_spacy_ner_tuple(doc, gold['entities'])
        for k, v in doc_pred.user_hooks.items():
            if k.endswith('matcher'):
                ner_pred.update({
                    k: doc_pred.user_hooks[k]
                })
        if gold_entities != ner_pred['predictions']:
            diff_ = {
                'type': 'FP' if not gold_entities else 'FN',
                'text': doc,
                'entities': gold_entities,
            }
            diff_.update(ner_pred)
            diff.append(diff_)
        elif gold_entities:
            diff_ = {
                'type': 'TP',
                'text': doc,
                'entities': gold_entities,
            }
            diff_.update(ner_pred)
            diff.append(diff_)
        else:
            pass
    with open(output, 'w', encoding='utf-8') as fw:
        for diff_ in diff:
            fw.write(json.dumps(diff_, ensure_ascii=False)+'\n')


def generalize_label(y_true, y_pred, confuse_map={}):
    y_true = str(y_true)
    y_pred = str(y_pred)
    for k, v in confuse_map.items():
        y_true = y_true.replace(k, v)
        y_pred = y_pred.replace(k, v)
    return eval(y_true), eval(y_pred)


def span2seq(ner_span):
    tags = ['O']*len(ner_span['text'])
    for span in ner_span.get('entities', ner_span.get('spans', [])):
        s, e = span['start'], span['end']
        tags[s:e] = ['B-'+span['label']]+['I-'+span['label']]*(e-s-1)
    return tags


def confusion_metrics(y_true, y_pred, digits=4):
    label_dict = nested_dict(2, int)
    for x, y in zip(y_pred, y_true):
        entity_names = set([label.split('-')[1]
                            for label in y if '-' in label])
        x = [label.split('-')[-1] for label in x]
        y = [label.split('-')[-1] for label in y]
        for x1, y1 in zip(x, y):
            if x1 == y1 and y1 in entity_names:
                label_dict[y1]['TP'] += 1
                label_dict[y1]['SUP'] += 1
            elif x1 != y1:
                if x1 in entity_names:
                    label_dict[x1]['FP'] += 1
                if y1 in entity_names:
                    label_dict[y1]['FN'] += 1
                    label_dict[y1]['SUP'] += 1
    label_dict = label_dict.to_dict()
    for label, v in label_dict.items():
        if 'TP' not in v:
            label_dict[label]['TP'] = 0
        if 'FP' not in v:
            label_dict[label]['FP'] = 0
        if 'FN' not in v:
            label_dict[label]['FN'] = 0
    return label_dict


def calculate_f1(TP, FP, FN, small_value=0.00001):
    precision = TP / (TP + FP + small_value)
    recall = TP / (TP + FN + small_value)
    f1 = (2 * precision * recall) / (precision + recall + small_value)
    return precision, recall, f1


def produce_report(label_dict, digits=4):
    last_line_heading = 'avg / total'
    name_width = max([len(name) for name in label_dict.keys()])
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'
    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    sup, p, r, f1 = [], [], [], []
    for k, v in label_dict.items():
        ps, rs, f1s = calculate_f1(v['TP'], v['FP'], v['FN'])
        report += row_fmt.format(*[k, ps, rs, f1s, v['SUP']],
                                 width=width, digits=digits)
        sup.append(v['SUP'])
        p.append(ps)
        r.append(rs)
        f1.append(f1s)
    report += u'\n'
    report += row_fmt.format(last_line_heading,
                             np.average(p, weights=sup),
                             np.average(r, weights=sup),
                             np.average(f1, weights=sup),
                             np.sum(sup),
                             width=width, digits=digits)
    return report


def flat_list(l):
    l_ = []
    for x in l:
        l_.extend(x)
    return l_


def get_non_completed_entity(label_sequence):
    found = re.findall('[BSU]\-([^,]+)', ','.join(label_sequence))
    return "O" if not found else found[0]


def align_label_sequence(y_true, y_pred):
    '''
    FIFO strategy for partial overlap!
    >>> y_true = ['O', 'B-LOC', 'I-LOC', 'O', 'O', 'O', 'O']
    >>> y_pred = ['O', 'O', 'O', 'O', 'B-PER', 'I-PER', 'O']
    >>> align_label_sequence(y_true,y_pred)
    (['O', 'LOC', 'O', 'O', 'O'],['O', 'O', 'O', 'PER' 'O'])
    '''
    i = 0
    y_true_, y_pred_ = [], []
    while i < len(y_true):
        if re.match('[BSUILME]', y_true[i]):
            tmp_label = y_true[i].split('-')[1]
            span_start = i
            i += 1
            while i < len(y_true) and not re.match('[BSUO]', y_true[i]):
                i += 1
            y_true_.append(tmp_label)
            y_pred_.append(get_non_completed_entity(y_pred[span_start:i]))
        elif re.match('[BSUILME]', y_pred[i]):
            tmp_label = y_pred[i].split('-')[1]
            span_start = i
            i += 1
            while i < len(y_true) and not re.match('[BSUO]', y_pred[i]):
                i += 1
            y_pred_.append(tmp_label)
            y_true_.append(get_non_completed_entity(y_true[span_start:i]))
        else:
            y_true_.append(y_true[i])
            y_pred_.append(y_pred[i])
            i += 1
    # if re.search('I\-',','.join(y_true_+y_pred_)):
    #     print(y_true, y_pred)
    return y_true_, y_pred_


def get_confusion_matrix(y_true, y_pred):
    aligned_y_true, aligned_y_pred = zip(*[
        align_label_sequence(x, y) for x, y in zip(y_true, y_pred)])
    flat_y_true, flat_y_pred = flat_list(
        aligned_y_true), flat_list(aligned_y_pred)
    labels = list(set(flat_y_true) | set(flat_y_pred))
    labels.sort()
    cm = confusion_matrix(flat_y_true, flat_y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


def get_metrics_report(y_true, y_pred, digits=4):

    report_dict = {
        "acc": '{:2.4f}'.format(accuracy_score(y_true, y_pred)),
        "token_based": produce_report(confusion_metrics(y_true, y_pred)),
        "span_based": sequence_report(y_true, y_pred, digits=4)
    }
    y_true, y_pred = generalize_label(y_true, y_pred, confuse_map={
        'FUNCTION': 'FUN_ISY_SKL',
        'SKILL': 'FUN_ISY_SKL',
        'INDUSTRY': 'FUN_ISY_SKL'
    })
    report_dict.update({
        "confused_acc": '{:2.4f}'.format(accuracy_score(y_true, y_pred)),
        "confused_token_based": produce_report(confusion_metrics(y_true, y_pred)),
        "confused_span_based": sequence_report(y_true, y_pred, digits=4)
    })
    return report_dict


def random_color(light=True):
    if light:
        num = random.uniform(0.5, 0.8)
    else:
        num = random.random()
    return hex(int(num * 0XFFFFFF)).replace('0x', '#')


def doccano2spacy(doccano):
    return [{
        "start": ent[0],
        "end":ent[1],
        "label":ent[2].upper()
    } for ent in doccano]


def spacy2yedda(spacy):
    '''
    >>> ner = {'text': '我叫朱慈祥，今年54岁了，家住龙游县湖镇镇周家村墩头自然村，我在司炉工岗位工作已经十几年了。',
         'entities': [{'start': 32, 'end': 35, 'label': 'FUNCTION'}]}
    >>> spacy2yedda(ner)
    '我叫朱慈祥，今年54岁了，家住龙游县湖镇镇周家村墩头自然村，我在[@司炉工#FUNCTION*]岗位工作已经十几年了。'
    '''
    tokens = list(spacy['text'])
    for ent in sorted(spacy['entities'],
                      key=lambda x: x.get('start'),
                      reverse=True):
        tokens[ent['start']:ent['end']] = list('[@') + \
            tokens[ent['start']:ent['end']] + \
            list('#') + \
            list(ent['label']) + \
            list('*]')
    return ''.join(tokens)


def yedda2spacy(yedda):
    '''
    >>> text = '我叫朱慈祥，今年54岁了，家住龙游县湖镇镇周家村墩头自然村，我在[@司炉工#FUNCTION*]岗位工作已经十几年了。'
    >>> yedda2spacy(text)
    {'text': '我叫朱慈祥，今年54岁了，家住龙游县湖镇镇周家村墩头自然村，我在司炉工岗位工作已经十几年了。',
     'entities': [{'start': 32, 'end': 35, 'label': 'FUNCTION'}]}
    '''
    entities = []
    raw_text = re.sub(r'\[[\@\$](.*?)\#.*?\*\](?!\#)', r'\1', yedda)
    i, j = 0, 0
    while i < len(raw_text):
        if raw_text[i] == yedda[j]:
            i += 1
            j += 1
        else:
            if yedda[j] == '[' and yedda[j+1] in ['@', '$']:
                span_start = i
                span_type = yedda[j+1]
                j += 2
                while raw_text[i] == yedda[j]:
                    i += 1
                    j += 1
                label_start = j+1
                while yedda[j] != '*' and yedda[j+1] != ']':
                    j += 1
                tmp_label = ''.join(yedda[label_start:j])
                entities.append({
                    'start': span_start,
                    'end': i,
                    'label': tmp_label.upper()
                })
                j += 2
    return {
        'text': raw_text,
        'entities': entities
    }


class NlpModel:

    def __init__(self):
        super().__init__()

    def __call__(self, text):
        raise NotImplementedError


class AiiNerModel(NlpModel):

    def __init__(self):
        super().__init__()

    def __call__(self, text):
        raise NotImplementedError


class AiiNerHttpModel(AiiNerModel):

    def __init__(self, url):
        super().__init__()
        self.url = url if url != "http://localhost:port/api" else "http://localhost:51785/ner_cv_aii"
        self.headers = {"content-type": "application/json"}
        self.session = requests.Session()
        self.labels = ["SCHOOL", "DEGREE", "MAJOR", "CERTIFICATE",
                       "INDUSTRY", "COMPANY", "FUNCTION", "SKILL"]

    def __call__(self, text):
        data = {"traffic_paramsDict": {"text": [text]}}
        response = self.session.post(
            self.url, data=json.dumps(data), headers=self.headers)
        if response.status_code == 200:
            res_data = json.loads(response.text)
            ents = [{'start': ent['boundary'][0], 'end':ent['boundary'][1],
                     'label':ent['type'].upper()} for ent in res_data['result'][0]['entities']]
            return ents
        else:
            print(response)
            return None


class AiiNerRuleHttpModel(AiiNerHttpModel):

    def __init__(self, url):
        self.url = url if url != "http://localhost:port/api" else "http://localhost:51655/ner"
        super().__init__(self.url)

    def __call__(self, text):
        data = {
            "header": {
                "log_id": "0x666",
                "user_ip": "192.168.8.52",
                "provider": "algo_survey",
                "product_name": "algo_survey",
                "uid": "0x666"
            },
            "request": {
                "c": "tagging",
                "m": "ner",
                "p": {
                    "user_id": 1001,
                    "query_body": {
                        "text_list": [text]
                    }
                }
            }
        }
        response = self.session.post(
            self.url, data=json.dumps(data), headers=self.headers)
        if response.status_code == 200:
            res_data = json.loads(response.text)
            ents = [{'start': ent['boundary'][0], 'end':ent['boundary'][1],
                     'label':ent['type'].upper()} for ent in res_data['response']['results'][0]]
            return ents
        else:
            print(response)
            return None


def test_recommend():
    '''
    use history annotated entity dict to annotate future raw text
    '''
    from recommend import maximum_matching
    spacy_ner = {
        "text": "尊敬的公司领导，你们好，我是胡爱辉，龙游等中共党员，今年37周岁，我的性格活泼开朗，善于与人沟通，而且爱好广泛，喜欢唱歌，追求新事物，现在在浙江顺康金属制品有限公司做电工，我做事比较认真负责，受到同事与上司的认可线，现在应聘贵公司高压电工或网格多功能工。",
        "entities": [{
                "start": 83,
                "end": 85,
                "label": "FUNCTION"
        }, {
            "start": 70,
            "end": 82,
            "label": "COMPANY"
        }, {
            "start": 115,
            "end": 119,
            "label": "FUNCTION"
        }, {
            "start": 120,
            "end": 126,
            "label": "FUNCTION"
        }]
    }
    raw_text = '我在浙江顺康金属制品有限公司做电工，想应聘贵公司的高压电工或网格多功能工。'
    yedda = spacy2yedda(spacy_ner)
    print('raw: '+raw_text)
    print('anno: '+maximum_matching(yedda, raw_text))


if __name__ == '__main__':
    test_recommend()
