import numpy as np
from Node import Node

class Tree:
    def __init__(self, data, candidats, test):
        self.data = data # matriu amb la base de dades
        self.candidats = candidats # cadascun dels atributs de la base de dades
        self.test = test # conjunt de test
        self.node_arrel = Node("Arrel", None, self.data, self.candidats[:-1])
    
    def generarArbre(self, tipus = 0):
        self.expand_tree(self.node_arrel, tipus)
    

    def calcular_predict(self):
        # classifica les dades del conjunt de test i suma els encerts de cada classe,
        # i calcula l'accuracy
        return self.predict_rec(self.test, self.node_arrel) / self.test.shape[0]

    def predict_rec(self, x_test, node):
        if node.fulla:
            #calcular els encerts
            y = x_test[:,-1]
            x = np.count_nonzero(y == node.decisio)
            return x
        else:
            suma = 0
            atrib = np.where(self.candidats == node.atribut)[0][0]
            for i, fill in enumerate(node.etiquetes_fills):
                nou_data = x_test[x_test[:,atrib] == fill]
                suma += self.predict_rec(nou_data, node.list_fills[i])
            return suma
    
    def expand_tree(self, node_arrel, tipus = 0): #expandeix segons l'atribut escollit; s'ha d'escollir el tipus d'expansió
                                                  #de l'arbre, que correspon a un dels 3 algorismes implementats
        pila = [node_arrel] # realment es una cua (cerca en profunditat)
    
        while (len(pila) != 0): #quan no quedin més nodes per expandir parem
            llista_aux = []
            node = pila[0]
            del pila[0]
            
            best_atrib = node.calculate_best_atribute(self.candidats, tipus)
            if best_atrib == -1: # cas que les files siguin totes iguals llavors prenem decisió per majoria
                valors_unics, count_valors = np.unique(node.data[:,-1], return_counts=True)
                node.decisio = valors_unics[np.argmax(count_valors)]
                node.fulla = True
                continue
            
            node.atribut = self.candidats[best_atrib][0]
            node.etiquetes_fills = np.unique(node.data[:,best_atrib])
            for fill in node.etiquetes_fills: #expandim el node actual creant els nodes fills i posant-los a la cua
                nou_data = node.data[node.data[:,best_atrib] == fill] #deixem les files que tinguin com a valor del atribut actual el valor del fill
                cand = np.delete(node.candidats, np.where(node.candidats == self.candidats[best_atrib])[0]) # eliminem el atribut actual, ja l'hem processat
                nou_node = Node(str(fill), node, nou_data, cand)
                node.list_fills.append(nou_node)
                
                if (nou_node.entropy < 0.2 or nou_node.data.shape[0] < 20 or nou_node.candidats.shape[0] == 0): # tallem la expansió per no crear un arbre massa gran
                    nou_node.fulla = True
                    valors_unics, count_valors = np.unique(node.data[:,-1], return_counts=True)
                    nou_node.decisio = valors_unics[np.argmax(count_valors)]
                else:
                    llista_aux.append(nou_node)
            pila = llista_aux + pila