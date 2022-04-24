import pandas as pd
import time
from Tree import Tree
from functions import representacio, load_dataset, tractament_nulls, kfold

def main():
    #carregar dataset
    adult_data = load_dataset('../BD/adult.data')
    test_data = load_dataset('../BD/adult.test')
    adult_names = load_dataset('../labels.txt')
    
    #eliminar valors nuls (cas C)
    ##adult_data = adult_data[(adult_data.iloc[:, 1:] != ' ?').all(axis=1)]
    ##test_data = test_data[(test_data.iloc[:, 1:] != ' ?').all(axis=1)]
    adult_names = adult_names.dropna()
    
    #tractar valors nulls
    adult_data = pd.DataFrame(tractament_nulls(adult_data.values))
    test_data = pd.DataFrame(tractament_nulls(test_data.values))
    
    #Treure els punts del train perque l'atribut objectiu acaba amb . al tesi i al train no :(
    test_data.iloc[:,-1] = test_data.iloc[:,-1].replace({' <=50K.':' <=50K', ' >50K.':' >50K'}, regex=True)
    
    #discretitzar valors continus
    columnes_discr = [0 , 2, 4, 10, 11, 12] #columnes a discretitzar
    for i in columnes_discr:
        col_train = pd.qcut(adult_data[i], q=4, precision = 0, duplicates="drop")
        col_train = col_train.astype('category')
        col_train = col_train.cat.codes
        
        adult_data[i] = col_train.to_numpy()
        ######################################
        col_test = pd.qcut(test_data[i], q=4, precision = 0, duplicates="drop")
        col_test = col_test.astype('category')
        col_test = col_test.cat.codes
        
        test_data[i] = col_test.to_numpy()
        
    print("==========================")
    print("PREDICCIÓ AMB TRAIN I TEST")
    print("==========================")
    start = time.time()
    arbre = Tree(adult_data.values, adult_names.values, test_data.values)
    arbre.generarArbre()
    end = time.time()
    print("Resultat de la prediccio:", arbre.calcular_predict())
    print("--------------------------")
    print("Temps transcorregut:", end - start)
    with open('arbre.txt', 'w') as file:
        representacio(arbre.node_arrel, 0, file)
        
    print("==========================")
    print("==== CROSS-VALIDATION ====")
    print("==========================")
    start = time.time()
    kfold(adult_data.values, 5, adult_names.values)
    end = time.time()
    print("--------------------------")
    print("Temps transcorregut:", end - start)
    print("==========================")
        
    
    


if __name__ == "__main__":
    main()