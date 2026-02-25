import pickle
import warnings


def loadDictionaryFromPickle(picklePath):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*align.*", category=DeprecationWarning)
        with open(picklePath, 'rb') as handle:
            data = pickle.load(handle)
    return data


def writeDictionaryToPickle(data, picklePath):
    with open(picklePath, 'wb') as handle:
        pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)
