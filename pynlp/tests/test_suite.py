from sklearn.pipeline import FeatureUnion, Pipeline

from ml_pipeline.cnn import CNN
from tasks import vua_format as vf
from ml_pipeline import utils
from ml_pipeline import preprocessing
from ml_pipeline import representation
from ml_pipeline import pipelines
import numpy as np

test_data_dir = 'tests/data/'


def test_data_load():
    task = vf.VuaFormat()
    task.load(test_data_dir)
    train_X, train_y, test_X, test_y = utils.get_instances(task, split_train_dev=False)
    assert len(train_X) == 199
    assert len(test_X) == 99


def train_test_data():
    train_X = ["They twats all deserve an ass kicking .", "Hope to talk to you later ."]
    train_y = ["hate", "noHate"]
    test_X = ["Lying Marxists !", "Now, young man !"]
    test_y = ["hate", "noHate"]
    return train_X, train_y, test_X, test_y


def test_preprocessing():
    train_X, train_y, test_X, test_y = train_test_data()
    preprocessor = preprocessing.std_prep()
    X = preprocessor.fit_transform(train_X)
    assert X[0] == "they twats all deserve an ass kicking ."

    preprocessor = preprocessing.lex_prep()
    X = preprocessor.fit_transform(train_X)
    assert X[0] == "NEUTRAL HATE NEUTRAL NEUTRAL NEUTRAL NEUTRAL NEUTRAL NEUTRAL"


def test_count_vectorizer():
    train_X, train_y, test_X, test_y = train_test_data()
    cv = representation.count_vectorizer()
    X = cv.fit_transform(train_X)
    assert cv.get_feature_names() == ['all', 'an', 'ass', 'deserve', 'hope', 'kicking', 'later', 'talk', 'they', 'to', 'twats', 'you']
    result = X.toarray()
    expected = np.array([[1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 1, 0, 1, 1, 0, 2, 0, 1]], np.int64)
    assert (result == expected).all()


def test_combined_features():
    train_X, train_y, test_X, test_y = train_test_data()
    token_features = Pipeline([('prep', preprocessing.std_prep()), ('frm', representation.count_vectorizer({'min_df': 1}))])
    X = token_features.fit_transform(train_X)
    expected = np.array([[1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0], [0, 0, 0, 0, 1, 0, 1, 1, 0, 2, 0, 1]], np.int64)
    assert (X.toarray() == expected).all()

    polarity_features = Pipeline(
        [('prep', preprocessing.lex_prep()), ('frm', representation.count_vectorizer({'min_df': 1}))])
    X = polarity_features.fit_transform(train_X)
    expected = np.array([[1, 7], [0, 7]], np.int64)
    assert (X.toarray() == expected).all()

    combined_features = FeatureUnion([
        ('token_features', token_features),
        ('polarity_features', polarity_features)])
    X = combined_features.fit_transform(train_X, train_y)
    actual = X.toarray()
    expected = np.array([[1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 7], [0, 0, 0, 0, 1, 0, 1, 1, 0, 2, 0, 1, 0, 7]], np.int64)
    assert (actual == expected).all()

    tokens_from_lexicon = combined_features.transformer_list[1][1].steps[0][1].tokens_from_lexicon
    assert tokens_from_lexicon == 1




def test_full_pipelines():
    train_X, train_y, test_X, test_y = train_test_data()
    pipes = [pipelines.naive_bayes_counts, pipelines.svm_libsvc_embed(), pipelines.naive_bayes_counts_lex()]

    for pipe in pipes:
        pipe = pipelines.naive_bayes_counts()
        pipe.fit(train_X, train_y)
        sys_y = pipe.predict(test_X)
        assert len(sys_y) == len(test_y)


def test_cnn_raw():
    pipe = CNN()
    train_X, train_y, test_X, test_y = train_test_data()
    train_X, train_y, test_X, test_y = pipe.encode(train_X, train_y, test_X, test_y)

    pipe.fit(train_X, train_y)
    sys_y = pipe.predict(test_X)
    assert len(sys_y) == len(test_y)