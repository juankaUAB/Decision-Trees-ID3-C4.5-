import numpy as np
import math


class Node:
    def __init__(self, etiqueta, pare, data, candidats, fulla = False):
        self.etiqueta = etiqueta # valor del atribut que representa el pare
        self.nodePare = pare
        self.candidats = candidats # atributs que encara es poden provar en aquest node
        self.atribut = None # atribut que representa el node
        self.decisio = None # decisio final
        self.etiquetes_fills = []
        self.list_fills = []
        
        self.data = data # matriu del dataset que es té en compte en aquest node
        self.entropy = self.calculate_base_entropy()
        self.gini = self.calculate_base_gini()
        self.fulla = fulla # si es fulla o no
    
    
    def sub_gini(self, dupla):
        # funcio que calcula el index gini per cada valor possible d'un node (per cada fill)
        if (dupla[0] == 0 or dupla[1] == 0):
            return 0
        
        total = dupla[0] + dupla [1]
        a = dupla[0] / total
        b = dupla[1] / total
        
        total = 1 - (pow(a,2) + pow(b,2))
        
        return total
    
    def sub_entropy(self, dupla):
        # funcio que calcula la entropia per cada valor possible d'un node (per cada fill)
        if (dupla[0] == 0 or dupla[1] == 0):
            return 0
        
        total = dupla[0] + dupla [1]
        a = dupla[0] / total
        a = -a * math.log2(a)
        
        b = dupla[1] / total
        b = -b * math.log2(b)
        
        return a + b
        
    
    def calculate_entropy_SA(self, atrib): #calcula l'entropia d'un atribut, tenint en compte tots els valors
                                           # possibles que pot prendre aquest atribut
        nFiles = self.data[:,(self.data.shape[1] - 1)].shape[0]
        d = {}
        for i, x in enumerate(self.data[:,atrib]):
            if (x in d.keys()):
                if (self.data[:, (self.data.shape[1] - 1) ][i] == " <=50K"):
                    d[x][0] += 1
                else:
                    d[x][1] += 1
            else:
                if (self.data[:, (self.data.shape[1] - 1) ][i] == " <=50K"):
                    d[x] = [1, 0]
                else:
                    d[x] = [0, 1]
                    
        entropia = 0
        recompte_fills = dict()
        for var in d.keys():
            total = d[var][0] + d[var][1]
            recompte_fills[var] = total / nFiles
            entropia += (total / nFiles) * self.sub_entropy(d[var])
        return entropia, recompte_fills
    
    def calculate_gini_SA(self, atrib): #calcula L'index Gini d'un atribut, tenint en compte tots els valors
                                        # possibles que pot prendre aquest atribut
        nFiles = self.data[:,(self.data.shape[1] - 1)].shape[0]
        d = {}
        for i, x in enumerate(self.data[:,atrib]):
            if (x in d.keys()):
                if (self.data[:, (self.data.shape[1] - 1) ][i] == " <=50K"):
                    d[x][0] += 1
                else:
                    d[x][1] += 1
            else:
                if (self.data[:, (self.data.shape[1] - 1) ][i] == " <=50K"):
                    d[x] = [1, 0]
                else:
                    d[x] = [0, 1]
                    
        entropia = 0
        recompte_fills = dict()
        for var in d.keys():
            total = d[var][0] + d[var][1]
            recompte_fills[var] = total / nFiles
            entropia += (total / nFiles) * self.sub_gini(d[var])
        return entropia, recompte_fills
    
    
    def calculate_base_entropy(self): # calcula la entropia d'un atribut segons la base de dades que té en aquell moment
                                      # tenint en compte quina classe té cada dada del dataset
        unique_elements, unique_counts = np.unique(self.data[:,-1], return_counts=True)
        num_elements = len(self.data[:,-1])

        probabilities = np.true_divide(unique_counts, num_elements)
        entropy = -np.sum(probabilities * np.log2(probabilities))

        return entropy
    
    def calculate_base_gini(self): # calcula l'index Gini d'un atribut segons la base de dades que té en aquell moment
                                   # tenint en compte quina classe té cada dada del dataset
        unique_elements, unique_counts = np.unique(self.data[:,-1], return_counts=True)
        num_elements = len(self.data[:,-1])

        probabilities = np.true_divide(unique_counts, num_elements)
        gini = 1 - np.sum(np.power(probabilities,2))

        return gini
            
    
    def calculate_best_atribute(self, candidats_tree, tipus = 0): #calcula el millor atribut segons el tipus d'expansió que
                                                                  #s'hagi escollit (ID3-Entropy, C4.5-GainRatio, Index Gini(CART,...))
        best = 0.0
        
        best_atrib = -1
        for atribut in self.candidats.reshape(-1):
            recompte_fills = dict()
            index_atrib = np.where(candidats_tree == atribut)[0][0]
            if tipus == 0: #gain (ID3)
                e_SA, recompte_fills = self.calculate_entropy_SA(index_atrib)
                guany = self.entropy - e_SA
            elif tipus == 1: #gain_ratio (C4.5)
                e_SA, recompte_fills = self.calculate_entropy_SA(index_atrib)
                guany = self.entropy - e_SA
                guany = guany / self.calculate_split_info(recompte_fills)
            elif tipus == 2: # gini (CART, SLIQ, SPRINT...)
                gini_SA, recompte_fills = self.calculate_gini_SA(index_atrib)
                guany = self.gini - gini_SA
            if (guany > best):
                best = guany
                best_atrib = index_atrib
        
        return best_atrib
    
    def calculate_split_info(self, recompte_fills): #calcula el split_info, el numero que divideix el gain per
                                                    #implementar el C4.5
        split_info = 0
        for fill in recompte_fills.keys():
            split_info -= recompte_fills[fill] * math.log2(recompte_fills[fill])
        if split_info == 0:
            return 1
        else:
            return split_info