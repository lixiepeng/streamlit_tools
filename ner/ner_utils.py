import json
import random

import numpy as np
import requests
from seqeval.metrics import accuracy_score
from seqeval.metrics import classification_report as sequence_report

from nested_dict import nested_dict

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


def get_metrics_report(y_true, y_pred, digits=4):
    return {
        "acc": '{:2.4f}'.format(accuracy_score(y_true, y_pred)),
        "label_based": produce_report(confusion_metrics(y_true, y_pred)),
        "entity_based": sequence_report(y_true, y_pred, digits=4)
    }


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
        self.url = url
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
