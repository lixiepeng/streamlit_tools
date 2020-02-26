import random
from spacy.language import Language
import requests
import json


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
        "label":ent[2]
    } for ent in doccano]


def get_ents(model_func, text):
    if isinstance(model_func, Language):
        doc = model_func(text)
        doc_json = doc.to_json()
        ents = doc_json['ents']
    else:
        ents = model_func(text)
    return ents


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
