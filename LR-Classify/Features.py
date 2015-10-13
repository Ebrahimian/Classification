__author__ = 'Ahmad'
'''third version: no duplicate tokens'''
'''inputs are train.process, test.process files and train and test partial folders'''
'''functions and entities are separated'''
import os
import nltk  # needs more work
import json
from collections import defaultdict
import cPickle
import numpy


def mapping_generator():
    functions = ["N/F"]
    entities = ["N/E"]
    chunks = []
    vocs = []
    uni_symbol_mapping = {}
    files = [os.path.join("C:\\Users\\Ahmad\\Desktop\\freebase\\partial", fn) for fn in
             next(os.walk("C:\\Users\\Ahmad\\Desktop\\freebase\\partial"))[2]]
    for file in files:
        mappings = json.loads(open(file).read())
        for key in mappings:
            for symb in mappings[key]:
                if 'lambda' in symb:
                    if symb not in functions: functions.append(symb)
                else:
                    if symb not in entities:entities.append(symb)
                uni_symbol_mapping[key] = symb
    uni_symbol_mapping['N/S'] = 'N/S'
    ####################################################################################################################

    """extract the words from train and test set and assign an ids to each"""
    dataset = json.loads(open("free917_process.json").read())
    for line in dataset:
        for chunk in line['utterance'].split():
            if chunk not in chunks: chunks.append(chunk)

    for chunk in chunks:
        for voc in chunk.split('_'):
            if voc not in vocs:
                vocs.append(voc)  # nltk.stem.WordNetLemmatizer().lemmatize(voc)

    return functions, entities, chunks, vocs, uni_symbol_mapping
    ####################################################################################################################

def lexicalFeatures(start, end):
    functions, entities, chunks, vocs, uni_entity_mapping = mapping_generator()
    symbols = ["N/S"]
    symbols.extend(functions)
    symbols.extend(entities)
    print '# of functions',len(functions)
    print '# of entities',len(entities)
    print '# of symbols',len(symbols)
    design_matrix = numpy.zeros((0, len(vocs)), dtype=float)
    target_vector = numpy.zeros((0,), dtype=int)
    files = [os.path.join("C:\\Users\\Ahmad\\Desktop\\freebase\\partial", fn) for fn in
             next(os.walk("C:\\Users\\Ahmad\\Desktop\\freebase\\partial"))[2]]

    for n, file in enumerate(files):
        if n >= end: break
        if n < start: continue
        mappings = json.loads(open(file).read())
        for key in mappings:
            tmp_symbol = numpy.zeros(shape=(1, len(vocs)))
            for symb in mappings[key]:
                try:
                    tmp_target = [uni_entity_mapping.keys().index(key)]
                except:
                    tmp_target = [uni_entity_mapping.keys().index('N/S')]

                for x in key.split('_'):
                    tmp_symbol[0][vocs.index(x)] = 1.0

                design_matrix = numpy.append(design_matrix, tmp_symbol, axis=0)
                target_vector = numpy.append(target_vector, tmp_target, axis=0)
    return (design_matrix, target_vector)


def composition_features(start, end):
    functions, entities, chunks, vocs, uni_entity_mapping = mapping_generator()
    design_matrix = numpy.zeros((0, len(vocs)+len(entities)), dtype=float)
    target_vector = numpy.zeros((0,), dtype=int)
    files = [os.path.join("C:\\Users\\Ahmad\\Desktop\\freebase\\partial", fn) for fn in
             next(os.walk("C:\\Users\\Ahmad\\Desktop\\freebase\\partial"))[2]]

    for n, file in enumerate(files):
        if n >= end: break
        if n < start: continue
        mappings = json.loads(open(file).read())
        for key in mappings:
            tmp_symbol1 = numpy.zeros((1, len(vocs)), dtype=float)
            tmp_symbol2 = numpy.zeros((1, len(entities)), dtype=float)
            if len(mappings[key])>1:
                for symb in mappings[key]:
                    try:
                        tmp_target = [functions.index(key)]
                    except:
                        tmp_target = [functions.index('N/F')]

                    if 'lambda' in symb:
                        for x in key.split('_'):
                            tmp_symbol1[0][vocs.index(x)] = 1.0
                    else:
                        tmp_symbol2[0][entities.index(symb)] = 1.0

                tmp_symbol = numpy.append(tmp_symbol1, tmp_symbol2, axis=1)
                design_matrix = numpy.append(design_matrix, tmp_symbol, axis=0)
                target_vector = numpy.append(target_vector, tmp_target, axis=0)
    return (design_matrix, target_vector)


if __name__ == '__main__':
    lex_train_set_xy = lexicalFeatures(0, 700)
    lex_valid_set_xy = lexicalFeatures(700, 817)
    lex_test_set_xy = lexicalFeatures(817, 917)

    comp_train_set_xy = composition_features(0, 700)
    comp_valid_set_xy = composition_features(700, 817)
    comp_test_set_xy = composition_features(817, 917)


    print 'lex matrix shape, training:',lex_train_set_xy[0].shape,lex_train_set_xy[1].shape
    print 'lex matrix shape, validation:',lex_valid_set_xy[0].shape,lex_valid_set_xy[1].shape
    print 'lex matrix shape, testing:',lex_test_set_xy[0].shape,lex_test_set_xy[1].shape

    print 'comp matrix shape, training:',comp_train_set_xy[0].shape,comp_train_set_xy[1].shape
    print 'comp matrix shape, validation:',comp_valid_set_xy[0].shape,comp_valid_set_xy[1].shape
    print 'comp matrix shape, testing:',comp_test_set_xy[0].shape,comp_test_set_xy[1].shape

    lex_myObject = (lex_train_set_xy, lex_valid_set_xy, lex_test_set_xy)
    comp_myObject = (comp_train_set_xy, comp_valid_set_xy, comp_test_set_xy)

    f = open('lex_free917LR.pkl', 'w')
    cPickle.dump(lex_myObject, f)
    f.close()

    f = open('comp_free917LR.pkl', 'w')
    cPickle.dump(comp_myObject, f)
    f.close()