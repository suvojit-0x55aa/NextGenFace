import pickle


def loadDictionaryFromPickle(picklePath):
    with open(picklePath, 'rb') as handle:
        data = pickle.load(handle)
    return data


def writeDictionaryToPickle(data, picklePath):
    with open(picklePath, 'wb') as handle:
        pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)
