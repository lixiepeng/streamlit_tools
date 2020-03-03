### Run
```
ssh -L 51785:192.168.7.202:51785 249
ssh -L 51655:192.168.7.214:51655 249
streamlit run ner_streamlit.py [custom_model]
```

### TO DO
  - fix cache return value mutation problem for model loading & model prediction
  - text area just time prediction for every char input
  - support batch predict for boosting performence
