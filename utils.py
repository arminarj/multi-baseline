import gzip
import torch
import torch
import numpy as np
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth
    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def make_tensor(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    data_x = torch.tensor(data_x)
    data_y = np.asarray(data_y, dtype='int32')
    return data_x, data_y


def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret

def save_load_name(args, name=''):
    if args.aligned:
        name = name if len(name) > 0 else 'aligned_model'
    elif not args.aligned:
        name = name if len(name) > 0 else 'nonaligned_model'

    return name + '_' + args.model


def save_model(args, model, name=''):
    name = save_load_name(args, name)
    torch.save(model, f'pre_trained_models/{name}.pt')


def load_model(args, name=''):
    name = save_load_name(args, name)
    model = torch.load(f'pre_trained_models/{name}.pt')
    return model

def eval_mosei_senti(results, truths, exclude_zero=False):

    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()
    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0 or (not exclude_zero)])
    thr = 0.4
    mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
    # corr = np.corrcoef(test_preds, test_truth)[0][1]
    f_score = f1_score((test_preds > thr), (test_truth > thr), average='weighted')
    binary_truth = (test_truth[non_zeros] > thr)
    binary_preds = (test_preds[non_zeros] > thr)

    for i in range(6):
        test_preds = results.view(-1, 6).cpu().detach().numpy()[:, i].reshape(-1, 1)
        test_truth = truths.view(-1, 6).cpu().detach().numpy()[:, i].reshape(-1, 1)
        test_preds = np.array(test_preds)
        test_truth = np.array(test_truth)
        thr = 0.4
        f_e_score = f1_score(test_truth>thr, test_preds>thr, average='weighted')
        print(f'F1 emotion {i+1}th score : {f_e_score}')

    print("MAE: ", mae)
    # print("Correlation Coefficient: ", corr)
    print("F1 score: ", f_score)
    print("Accuracy: ", accuracy_score(binary_truth, binary_preds))

    print("-" * 50)
