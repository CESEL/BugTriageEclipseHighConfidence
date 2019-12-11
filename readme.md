# BugTriageEclipseHighConfidence

Add abstract

Link to ICSME version and to thesis

## Install and Dependencies 

To setup python and install the libraries and prepare the dataset, please refer to [install](https://github.com/CESEL/BugTriageEclipseHighConfidence/blob/master/install.md)

## Generating the Results

To run the models and get the results, please refer to [run models](https://github.com/CESEL/BugTriageEclipseHighConfidence/blob/master/run_models.md)

### RQ1a. Textual & Categorical

How accurately do models containing textual and categorical features triage bugs?

1. Model M1: Logistic RegressionWith Textual Features - To get the results of Model M1, run [classification_text](https://github.com/CESEL/BugTriageEclipseHighConfidence/blob/master/ML_Text_Categorical/classification_text.py)

2. Model M3: Textual and Component - To get the results of Model M3, run [classification_text_component](https://github.com/CESEL/BugTriageEclipseHighConfidence/blob/master/ML_Text_Categorical/classification_text_component.py)

3. Model M4: Textual and Component (All developers)- To get the results of Model M3, run [classification_text_component_all_devs](https://github.com/CESEL/BugTriageEclipseHighConfidence/blob/master/ML_Text_Categorical/classification_text_component_all_devs.py)

### RQ1b. FixerCache

Does a developer’s affinity to working on specific components improve the accuracy of bug triaging?

1. Refer to the paper of Wang et al. [Fixer Cache](https://dl.acm.org/citation.cfm?id=2652536)

2. Run [prediction_fixer_cache](https://github.com/CESEL/BugTriageEclipseHighConfidence/blob/master/FixerCache/prediction_fixer_cache.py) to get the results of FixerCache Model

### RQ2. Crash traces
Does the information contained in alarm logs, crash dumps, and stack traces help in bug triaging?

1. Run the [prediction_by_msr](https://github.com/CESEL/BugTriageEclipseHighConfidence/blob/master/stacktrace_processing/prediction_by_msr.py) on existing data to get the results of the stack trace & commit score based model.

2. To Run the model on new data set, please follow the steps mentioned below.

    1. For setup of infozilla tool, please follow the steps from [infozilla](https://github.com/kuyio/infozilla)
    
    2. Download the comments of the bug reports. The structure of the location where the text file of comments should be kept is - path/bug_id/bug_id_comments.txt
    
    3. Run the [infozilla](https://github.com/CESEL/BugTriageEclipseHighConfidence/blob/master/stacktrace_processing/infozilla.py) after replacing the original path of the resources
    
    4. Run the [xml_parser](https://github.com/CESEL/BugTriageEclipseHighConfidence/blob/master/stacktrace_processing/xml_parser.py). This script runs with the xml output generated by infozilla creates a json where call depth is the key and the corresponding value is the source code file name
    
    5. Clone the eclipse code repositories locally and put their location in jdt_dict.json and platform_dict.json.
    
    6. Run the [source_code_path_in_repo_collector](https://github.com/CESEL/BugTriageEclipseHighConfidence/blob/master/stacktrace_processing/source_code_path_in_repo_collector.py) to extract the path of the source code file from the cloned Eclipse repository.
    
    7. Run the [prediction_by_msr](https://github.com/CESEL/BugTriageEclipseHighConfidence/blob/master/stacktrace_processing/prediction_by_msr.py) to get the final results.

### RQ3. Combined Model
Does the model trained with text, categorical and log features improve accuracy of bug triaging?

1. Run [classification_ensemble](https://github.com/CESEL/BugTriageEclipseHighConfidence/blob/master/ensemble/classification_ensemble.py) to get the result of Ensemble Model M6

### RQ4. High Confidence Predictions
What is the impact of high confidence prediction on the accuracy of triaging?

1. Run [classification_text_component_high_confidence](https://github.com/CESEL/BugTriageEclipseHighConfidence/blob/master/high_confidence_prediction/classification_text_component_high_confidence.py) by setting appropraite cut_off confidence in [config.json](https://github.com/CESEL/BugTriageEclipseHighConfidence/blob/master/resources/config.json)
