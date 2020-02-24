import json
import sys

import pandas as pd
import spacy
import srsly
import streamlit as st
from spacy import displacy

from ner.ner_utils import doccano2spacy, random_color

DEFAULT_TEXT = """Donald John Trump (born June 14, 1946) is the 45th and current president of the United States."""
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.5rem; padding: 0.5rem; margin-bottom: 0.5rem">{}</div>"""
SETTINGS = {"style": "ent", "manual": True, "options": {}}


@st.cache(allow_output_mutation=True)
def load_model(name):
    try:
        return spacy.load(name)
    except:
        return None


@st.cache(allow_output_mutation=True)
def process_text(model_name, text):
    nlp = load_model(model_name)
    return nlp(text)


st.sidebar.title("Interactive spaCy visualizer")
st.sidebar.markdown("""
Process text with [spaCy](https://spacy.io) models and visualize named entities,
dependencies and more. Uses spaCy's built-in
[displaCy](http://spacy.io/usage/visualizers) visualizer under the hood.
""")

custom_model = None
if len(sys.argv) > 1:
    custom_model = sys.argv[1]
spacy_model = st.sidebar.text_input(
    "Input one model name or path:", custom_model if custom_model else "en_core_web_sm")
model_load_state = st.info(f"Loading model '{spacy_model}'...")
nlp = load_model(spacy_model)
model_load_state.empty()


def spacy_pipeline():
    text = st.text_area("Text to analyze", DEFAULT_TEXT)
    doc = process_text(spacy_model, text)

    if "parser" in nlp.pipe_names:
        st.header("Dependency Parse & Part-of-speech tags")
        st.sidebar.header("Dependency Parse")
        split_sents = st.sidebar.checkbox("Split sentences", value=True)
        collapse_punct = st.sidebar.checkbox(
            "Collapse punctuation", value=True)
        collapse_phrases = st.sidebar.checkbox("Collapse phrases")
        compact = st.sidebar.checkbox("Compact mode")
        options = {
            "collapse_punct": collapse_punct,
            "collapse_phrases": collapse_phrases,
            "compact": compact,
        }
        docs = [span.as_doc() for span in doc.sents] if split_sents else [doc]
        for sent in docs:
            html = displacy.render(sent, options=options)
            # Double newlines seem to mess with the rendering
            html = html.replace("\n\n", "\n")
            if split_sents and len(docs) > 1:
                st.markdown(f"> {sent.text}")
            st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)

    if "ner" in nlp.pipe_names:
        st.header("Named Entities")
        st.sidebar.header("Named Entities")
        label_set = list(nlp.get_pipe("ner").labels)

        labels = st.sidebar.multiselect(
            "Entity labels", label_set, label_set
        )
        SETTINGS.update({
            'options': {
                'colors': {label: random_color() for label in labels},
                'ents': labels
            }
        })
        doc_json = doc.to_json()
        html = displacy.render(
            {'text': doc_json['text'], 'ents': doc_json['ents']}, **SETTINGS)
        # Newlines seem to mess with the rendering
        html = html.replace("\n", " ")
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
        attrs = ["text", "label_", "start", "end", "start_char", "end_char"]
        if "entity_linker" in nlp.pipe_names:
            attrs.append("kb_id_")
        data = [
            [str(getattr(ent, attr)) for attr in attrs]
            for ent in doc.ents
            if ent.label_ in labels
        ]
        df = pd.DataFrame(data, columns=attrs)
        st.dataframe(df)

    if "textcat" in nlp.pipe_names:
        st.header("Text Classification")
        st.markdown(f"> {text}")
        df = pd.DataFrame(doc.cats.items(), columns=("Label", "Score"))
        st.dataframe(df)

    vector_size = nlp.meta.get("vectors", {}).get("width", 0)
    if vector_size:
        st.header("Vectors & Similarity")
        st.code(nlp.meta["vectors"])
        text1 = st.text_input("Text or word 1", "apple")
        text2 = st.text_input("Text or word 2", "orange")
        doc1 = process_text(spacy_model, text1)
        doc2 = process_text(spacy_model, text2)
        similarity = doc1.similarity(doc2)
        if similarity > 0.5:
            st.success(similarity)
        else:
            st.error(similarity)

    st.header("Token attributes")

    if st.button("Show token attributes"):
        attrs = [
            "idx",
            "text",
            "lemma_",
            "pos_",
            "tag_",
            "dep_",
            "head",
            "ent_type_",
            "ent_iob_",
            "shape_",
            "is_alpha",
            "is_ascii",
            "is_digit",
            "is_punct",
            "like_num",
        ]
        data = [[str(getattr(token, attr)) for attr in attrs] for token in doc]
        df = pd.DataFrame(data, columns=attrs)
        st.dataframe(df)

    st.header("JSON Doc")
    if st.button("Show JSON Doc"):
        st.json(doc.to_json())

    st.header("JSON model meta")
    if st.button("Show JSON model meta"):
        st.json(nlp.meta)


def ner_plus():
    def row_2_html(row):
        return displacy.render(row, **SETTINGS).replace("\n\n", "\n")

    uploaded_file = st.file_uploader(
        "Upload documents in annotated jsonl(spacy|doccano) or raw txt:",
        type=["jsonl", "txt"],
        encoding="utf-8")

    st.sidebar.header("Named Entities")
    label_set = list(nlp.get_pipe("ner").labels)
    labels = st.sidebar.multiselect(
        "Entity labels", label_set, label_set
    )
    select_all = labels == label_set
    SETTINGS.update({
        'options': {
            'colors': {label: random_color() for label in labels},
            'ents': labels
        }
    })
    entity_info = 'Entities Info'
    if uploaded_file is not None:
        lines = uploaded_file.readlines()
        try:
            data = [json.loads(line) for line in lines]
        except:
            data = [{"text": line.strip()}
                    for line in lines if line.strip()]
        count_dict = {
            "n_no_ents": 0,
            "n_total_ents": 0,
            "n_predict_ents": 0,
            "n_diff": 0
        }

        only_diff = st.sidebar.checkbox("Only Diff")

        st.header(f"{entity_info} ({len(data)})")

        predict_data = []

        for eg in data:
            ents = eg.get("entities", doccano2spacy(eg.get("labels", [])))
            original_ents = [ent for ent in ents if ent['label'] in labels]

            doc = nlp(eg["text"])
            doc_json = doc.to_json()
            predict_ents = [ent for ent in doc_json['ents']
                            if ent['label'] in labels]
            if not (only_diff and original_ents == predict_ents):
                predict_data.append({
                    "meta": eg.get("meta", {}),
                    "text": eg["text"],
                    "entities": predict_ents
                })
            if original_ents == predict_ents:
                if only_diff or (not select_all and not original_ents):
                    continue
                row = {"text": eg["text"], "ents": original_ents}
                count_dict['n_total_ents'] += len(row["ents"])
                if not row["ents"]:
                    count_dict['n_no_ents'] += 1
                count_dict['n_predict_ents'] += len(row["ents"])
                html = row_2_html(row)
                st.markdown(HTML_WRAPPER.format(html), unsafe_allow_html=True)
            else:
                count_dict['n_diff'] += 1
                # original
                row = {"text": eg["text"], "ents": original_ents}
                count_dict['n_total_ents'] += len(row["ents"])
                if not row["ents"]:
                    count_dict['n_no_ents'] += 1
                original_html = row_2_html(row)
                # predict
                row = {"text": eg["text"], "ents": predict_ents}
                count_dict['n_predict_ents'] += len(row["ents"])
                predict_html = row_2_html(row)
                st.markdown(HTML_WRAPPER.format('original: \n'+original_html+'\npredict:\n' +
                                                predict_html), unsafe_allow_html=True)
        n_no_ents = count_dict['n_no_ents']
        n_total_ents = count_dict['n_total_ents']
        n_predict_ents = count_dict['n_predict_ents']
        n_diff = count_dict['n_diff']
        st.sidebar.markdown(
            f"""
        | `{entity_info}` | |
        | --- | ---: |
        | Total examples | {len(data):,} |
        | Total entities | {n_total_ents:,} |
        | Total predict entities | {n_predict_ents:,} |
        | Examples with diff | {n_diff:,} |
        | Examples with no entities | {n_no_ents:,} |
        """
        )
        st.sidebar.header("Predict data")
        st.sidebar.json(predict_data)

# page router


usage = st.sidebar.selectbox("Usage", ['spacy_pipeline', 'ner_plus'])

if nlp is not None:
    if usage == 'spacy_pipeline':
        spacy_pipeline()
    elif usage == 'ner_plus':
        ner_plus()
    else:
        st.markdown("""
        # You need choice an Usage
        """)
else:
    st.info("Please input a valid model name or path")
