import numpy as np
import pandas as pd
from Tree import Tree  

def representacio(node, indent, file, tipus = 0): # escriu l'arbre en un arxiu .txt
    acc = ((indent-1) * "    |")         #guiones verticales
    if indent>0:
        acc += "    |"                          #espacios + 'T' apuntando pa la derecha
    acc += 3 * "-"                          #guion lateral
    acc += "Value: " + node.etiqueta + " --> "

    if node.fulla:
        file.write(acc + "Solution:\"" + str(node.decisio) + "\"" + "\n")
    else:
        file.write(acc + "Attribute: " + str(node.atribut) + "\n")
        for fill in node.list_fills:
            representacio(fill, indent+1, file)
            
def load_dataset(path): # carrega el dataset d'arxius .csv
    dataset = pd.read_csv(path, header=None, delimiter=',')
    
    return dataset

def tractament_nulls(data): # tractament avançat de les dades NaN (decisió per majoria)
    for i in range(data.shape[1]):
        col = data[:,i]
        maxim = col[np.argmax(data[:,i])]
        data[:,i] = np.where(col==' ?', maxim, col) 
        
    return data

def kfold(dataset, n_particions, labels): # estrategia de validació creuada per provar el nostre model
    """DIVIDIR DATASET EN N PARTICIONS"""
    new_dataset = dataset[np.random.permutation(len(dataset))]
    conjunts = []
    for inc,i in enumerate(range(0,new_dataset.shape[0],int(new_dataset.shape[0]/n_particions))):
        if inc < n_particions:
            seguent = (int(new_dataset.shape[0]/n_particions))*(inc+1)
            conjunts.append(new_dataset[i:seguent,:])
    """EXECUTAR CROSS-VALIDATION"""
    mitj = 0
    for i, test in enumerate(conjunts):
        conjunts.pop(i)
        train = np.concatenate(conjunts)
        
        arbre = Tree(train, labels, test)
        arbre.generarArbre()
        #########
        pred = arbre.calcular_predict()
        mitj += pred
        print("Predicció per al conjunt", i, ":", pred)
        
        conjunts.insert(i, test)
    
    print("--------------------------")
    print("Resultat mitjà del cross-validation:", mitj / n_particions)