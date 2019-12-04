# Dependencies 

The code runs with Python 3.7. It has dependency on the following libraries. 

Run the commands to install them:

```bash
pip install gitpython

pip install pydriller

pip install nltk

pip install numpy

pip install pandas

pip install python-bugzilla

pip install python-dateutil

pip install scikit-learn

pip install scipy
```

# Results

## RQ1a. Textual & Categorical:

Model M1: 

```bash
python ML_Text_Categorical/classification_text.py
```

Model M2: 

```bash
python ML_Text_Categorical/classification_text_component.py
```

Model M3: 

```bash
python ML_Text_Categorical/classification_text_component_all_devs.py
```

## RQ1b. FixerCache:

```bash
python FixerCache/prediction_fixer_cache.py
```

## RQ2. Crash traces: 

```bash
python stacktrace_processing/prediction_by_msr.py
```

## RQ3. Combined Model:

```bash
python ensemble/classification_ensemble.py
```

## RQ4. High Confidence Predictions:

```bash
python high_confidence_prediction/classification_text_component_high_confidence.py
```
