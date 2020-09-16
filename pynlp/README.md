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
- resources: scripts and polarity lexicons

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
Install libraries, as well as the spacy model 'en_core_web_sm':

```
   $ pip install -r requirements.txt
   $ python -m spacy download en_core_web_sm
```

Data
----
The following data should be stored under a `data` folder under the project root `pynlp/`:

* Word embeddings. The code currently runs with Glove twitter embeddings (`data/glove.twitter.27B.100d.vec`) or wiki-news embeddings (`data/wiki-news-300d-1M.vec`). You can also modify the code `pynlp/ml_pipeline/representation.py` to use with other embeddings.


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


Running experiments
--------------------
Call the 'ml_pipeline' module to run experiments:

   * ```pynlp$ python -m ml_pipeline```  
   * from PyCharm: edit a run configuration, setting the working directory to 'pynlp'
   * the main function takes three arguments:
       * a task name (default is 'vua_format') specifying data file names
       * the path to the data (default is 'data/[task-name]')
       * a pipeline (default is 'naive_bayes')   

Authors 
=========
Sophie Arnoult, Pia Sommerauer, Isa Maks

