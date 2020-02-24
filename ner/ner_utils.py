import random


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
