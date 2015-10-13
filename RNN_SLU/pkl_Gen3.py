__author__ = 'Ahmad'
"""third version: direct logical symbol acquaintance, no duplicate tokens"""
"""inputs are train.process, test.process files and train and test partial folders"""
"""functions and entities are separated"""
import os
import json
from numpy import array
from collections import defaultdict
import cPickle

func = []
ent = []
map_dic = defaultdict(dict)


def mapping_generator():
    indexes_entity = ["N/E"]
    indexes_function = ["N/F"]
    global func
    files = [os.path.join("C:\\Users\\Ahmad\\Desktop\\freebase\\partial", fn) for fn in
             next(os.walk("C:\\Users\\Ahmad\\Desktop\\freebase\\partial"))[2]]
    for file in files:
        mappings = json.loads(open(file).read())
        for key in mappings:
            for symb in mappings[key]:
                if 'lambda' in symb:
                    if symb not in indexes_function:
                        indexes_function.append(symb)
                else:
                    if symb not in indexes_entity:
                        indexes_entity.append(symb)

    """aggregate train and test logical symbols in dict and assign a unique ids to each"""
    for sym in indexes_function:
        map_dic["labels2idx"][sym] = indexes_function.index(sym)

    for sym in indexes_entity:
        map_dic["tables2idx"][sym] = indexes_entity.index(sym)

    ####################################################################################################################

    """extract the words from train and test set and assign an ids to each"""
    dataset = json.loads(open("free917_process.json").read())
    chunks = []
    for line in dataset:
        for chunk in line['utterance'].split():
            chunks.append(chunk)

    ids = 0
    for chunk in set(chunks):
        map_dic["words2idx"][chunk] = ids

        ids += 1
    return map_dic
    ####################################################################################################################


def data_pickling(ds, limit1, limit2):  # "free917train_process.json"
    """replace each words in train utterances with their corresponding ids:"a" arrays:(sym, chun)"""
    dataset = json.loads(open(ds).read())

    chunk_list = []
    sym_list = []
    func_list = []

    for index, utter in enumerate(dataset):
        map_file = json.loads(open("partial\\mapping_Partial" + str(index + 1) + ".json").read())
        if index in range(limit1, limit2):
            fn = []
            sm = []
            chun = []

            for token in utter['utterance'].split():
                if token in map_file:
                    if len(map_file[token]) == 1:
                        for k in map_file[token]:
                            if "lambda" in k:
                                fn.append(map_dic["labels2idx"][k])
                            else:
                                fn.append(map_dic["labels2idx"]["N/F"])
                    if len(map_file[token]) == 2:
                        for k in map_file[token]:
                            if "lambda" in k:
                                fn.append(map_dic["labels2idx"][k])
                else:
                    fn.append(map_dic["labels2idx"]["N/F"])

            for token in utter['utterance'].split():
                if token in map_file:
                    if len(map_file[token]) == 1:
                        for k in map_file[token]:
                            if "lambda" not in k:
                                sm.append(map_dic["tables2idx"][k])
                            else:
                                sm.append(map_dic["tables2idx"]["N/E"])
                    if len(map_file[token]) == 2:
                        for k in map_file[token]:
                            if "lambda" not in k: sm.append(map_dic["tables2idx"][k])
                else:
                    sm.append(map_dic["tables2idx"]["N/E"])

                chun.append(map_dic["words2idx"][token])

            chunk_list.append(array(chun))
            sym_list.append(array(sm))
            func_list.append(array(fn))
            if len(array(chun)) != len(array(sm)) != len(array(fn)):
                print array(chun)
                print array(sm)
                print array(fn)
                print utter
                print '**' * 100
    return (chunk_list, sym_list, func_list)

#######################################################################################################################
map_dictionary = mapping_generator()
training = data_pickling("free917_process.json", 0, 513)
validation = data_pickling("free917_process.json", 513, 641)
testing = data_pickling("free917_process.json", 641, 918)
myObject = (training, validation, testing, map_dictionary)
f = open('free917.pkl', 'w')
cPickle.dump(myObject, f)
f.close()

