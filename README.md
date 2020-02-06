# Chatty
http://rest.chatty.ryanglambert.com/

- Inference
  - You probably want to start by taking a look in `chatty.api`
  - Then you'll want to take a look in `analyze`
  - `speech_act` and `extractors` are both used in analyze take a look at those next
  - `model` handles loading and saving of model which is called from within `speech_act`
  
- Training
  - Training is one time and in a notebook for now.
  - Checkout `Speech Act Classifier Final.ipynb` in `research/daily_dialogue` for more
  - Model is stored as \*.pkl in the repo


# setup
- Clone the repo
- put this pkl file in `old_chatty/chatty/` [https://drive.google.com/file/d/1PAgc5sOBUw47muXHX8LLUb6yS7xandgV/view?usp=sharing](pretrained_sklearn_estimator)
- `python3 -m venv .env`
- `source .env/bin/activate`
- `pip install -r requirements.txt`
- `pip install -e .`
- `EXPORT ENV=dev`
- `EXPORT SECRET_KEY=notasecret`
- `EXPORT CHATTY_CONF_PATH=conf.yaml`
- `py.test` (all tests should pass)

# module descriptions
```
chatty
 | api        - The api through which you can pass example slack messages or already parsed utterances
 | conf       - Configuration Parser which points to config.yaml
 | analyze    - Machine Learning module for analyzing text (uses sentiment and speech_act modules, extractors) 
 | extractors - (should be moved) Util module to handle utterances in different formats (slack vs pre-split)
 | model      - (should be moved) Util module for handling saving and loading of models
 | sentiment  - (should be moved) Util module placeholder (sentiment classification not yet implemented)
 | speech_act - (should be moved) Util module for parsing/extracting speech acts from text 
 | utils
   | tokens     - Util module for tokenizing text
   | cleaning   - Util module for cleaning text
research
 | daily_dialogue
   | Speech Act Classifier Final.ipynb - Notebook containing final model chosen for rest api endpoint
   | data (needs to be moved)          - Util module for preprocessing steps and data transformation
   | plot                              - Helper module for plotting in notebook

tests/
 | test_analyze
 | test_analyze_endpoint
 | test_extractory
```

# not yet implemented
- `sentiments` makes a random guess on sentiment for now as a placeholder

