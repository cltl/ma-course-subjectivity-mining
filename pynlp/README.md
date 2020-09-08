Code for NB/SVM
===========================

The  code is organized in four folders

- tasks: contains classes to extract data related to specific tasks. You can use the model of 'vua-format' to extract data from other tasks/data sets
- ml_pipeline: code for ML pipeline
    * preprocessing: tokenize, lowercase, etc.
    * representation: format data for input to classifiers. Currently allows for count vectors and word embeddings
    * utils: utility functions for data splitting and grid search
    * pipelines: defines pipelines with a given preprocessing, representation and classification step. Current classifiers are Naive Bayes and SVM
    * experiment: contains the main method to run pipelines on a given task
- tests: contains a basic test suite (to be run with pytest), showing usage examples 
- resources: scripts and resources for processing the TRAC2018 and hate-speech-vicom datasets (the datasets are not included).

Requirements
============
Environment
------------
You can create a Python (3) environment (for instance 'myenv') with pip or conda, and load required packages as follows.

With venv and pip:

```
   $ python -m venv myenv
   $ source activate myenv
```

With conda:

```
   $ conda create --name myenv python=3.7
   $ conda activate myenv
```

Dependencies
-------------
Install the following libraries, as well as the spacy model 'en_core_web_sm':

```
   $ pip install Keras tensorflow gensim scikit-learn spacy nltk numpy pandas pytest
   $ python -m spacy download en_core_web_sm
```
Even if you use conda to create your environment, it is recommended to install Keras with `pip`.

Data
----
The following data should be stored under a `data` folder under the project root `pynlp/`:

* Offenseval data: `offenseval-training-v1.tsv` and `offenseval-trial.txt`
* Word embeddings. The code currently runs with Glove twitter embeddings (`data/glove.twitter.27B.100d.vec`) or wiki-news embeddings (`data/wiki-news-300d-1M.vec`). You can also modify the code `pynlp/ml_pipeline/representation.py` to use with other embeddings.

To run some of the tests, you should also have the following files:

* `resources/TRAC2018/VUA_format/devData.csv` and `.../trainData.csv`
* `resources/hate-speech-dataset-vicom/VUA_format/testData.csv` and `.../trainData.csv`

Usage
=======
Test suite
-----------
You can use pytest to run the test suite, from the command line or from Pycharm.

* from the command line (under 'pynlp'): call 'pytest'; this will run all test suites under 'tests'
* from PyCharm: create a pytest configuration to run `test_suite.py`. Use the absolute path to 'pynlp' as working directory.

Follow these steps if you have difficulties creating a pytest configuration:

1. set pytest as the default test runner in Pycharm: `Preferences (CMD+,) > Tools > Python Integrated Tools` (see also the [Pycharm documentation](https://www.jetbrains.com/help/pycharm/pytest.html))
2. create a test run configuration: in the Pycharm menu, select `Run > Edit configurations`, then click on the `+` to add a new `Python tests / pytest` configuration. Enter a module name (`test_suite`) or a script path in the `target` field. The working directory for tests should point to `pynlp`.

Note also that the tests will not run out of the box, as they depend on data files being present.

Running experiments
--------------------
Call the 'ml_pipeline' module to run experiments:

   * ```pynlp$ python -m ml_pipeline```  
   * from PyCharm: edit a run configuration, setting the working directory to 'pynlp'
   * the main function takes three arguments:
       * a task name (default is 'vua_format') specifying data file names
       * the path to the data (default is 'data/test')
       * a pipeline (default is 'naive_bayes')   

Authors 
=========
Sophie Arnoult, Pia Sommerauer

