#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 15:12:33 2017

@author: roberto
"""

import hashlib
import numpy as np


# problema : se rodar de novo, gera outro conjunto de test_set
# mas é bom que o test_set seja um só e que não tenha dados de treino
# solucao : hashlib 

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

# calculo um hash para cada instancia do conjunto de dados
# e pego o ultimo byte do hash, e se for menor que 51 ( 20% de 256)
# coloco no conjunto teste.
#Agora mesmo que eu recarregue o conjunto de dado, o novo conjunto de teste não
# vai conter nenhum dado que antes era do conjunto de treino

def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column,hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_,test_ratio,hash))
    return data.loc[~in_test_set],data.loc[in_test_set]


housing = load_housing_data()

#o problema eh que housing nao tem uma coluna de indice, vou acrescentar:
housing_with_id = housing.reset_index()

train_set,test_set =split_train_test_by_id(housing_with_id,0.2,"index")

train_set , test_set = split_train_test(housing, 0.2)