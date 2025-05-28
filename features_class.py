import networkx as nx
import spacy
from janome.tokenizer import Tokenizer
import jieba
from konlpy.tag import Okt
import numpy as np

# Memory load class
class memory_load:
    def __init__(self, sentence, language,**model_dict):
        self.model_dict = model_dict
        self.sentence = sentence
        self.language = language
        self.nlp = spacy.load(self.model_dict[self.language]) 
        self.doc = self.nlp(self.sentence)
        self.feature_interference = self._feature_interference()
        self.feature_misbinding = self._feature_misbinding()
        
    def _feature_interference(self):
     
     dep_dict={}
     pos_dict={}
     dep_count = 0
     pos_count = 0

     for token in self.doc:
      if token.dep_ in dep_dict:
       dep_dict[token.dep_]+=1
      else:
       dep_dict[token.dep_] = 1

      if token.pos_ in pos_dict:
       pos_dict[token.pos_]+=1
      else:
       pos_dict[token.pos_] = 1

     for dep,count in dep_dict.items():
      if count > 1:
       dep_count+=1

     for pos,count in pos_dict.items():
      if count > 1:
       pos_count+=1

     interference_score = dep_count + pos_count
     return interference_score

    def _feature_misbinding(self):

     misbinding_score = 0
     for token in self.doc:
        if token.dep_ in ["nsubj", "dobj", "iobj", "pobj"]:  
            if token.head.dep_ not in ["ROOT", "acl", "relcl"]:  
                misbinding_score += 1
     return misbinding_score

    def value(self):
        memory_load_value =  self.feature_interference + self.feature_misbinding
        return memory_load_value



# Fixed effects class
class fixed_effects:
    def __init__(self, sentence, language, **model_dict):
        self.model_dict = model_dict
        self.sentence = sentence
        self.language = language
        self.nlp = spacy.load(self.model_dict[self.language])
        self.doc = self.nlp(self.sentence)
        self.adj_matrix = self.__adj_matrix()
        self.coord = self.__coord()

    def __adj_matrix(self):
        edges = [(f"{token.head.text}_{token.head.i}", f"{token.text}_{token.i}") for token in self.doc]

        # Create the graph
        G = nx.DiGraph()
        G.add_edges_from(edges)

        # Generate the adjacency matrix
        nodelist = [f"{token.text}_{token.i}" for token in self.doc]
        adj_matrix = nx.to_numpy_array(G, nodelist=nodelist)
        return adj_matrix

    def __coord(self):
        matrix = self.adj_matrix
        row_val = []
        for i in range(matrix.shape[0]):
            if np.sum(matrix[i]) == 0:
                pass
            else:
                row_val.append(i)
        coord = []
        for a in row_val:
            for j in range(matrix.shape[1]):
                if matrix[a][j] == 1:
                    coord.append((a, j))
        return coord

    def dep_length(self):
        dependency_length = 0
        for i, j in self.coord:
            dependency_length += np.abs(i - j)

        return dependency_length

    def __inter_val(self, start, end):
        value = 0
        if start > end:
            start, end = end, start
            sub_matrix = self.adj_matrix[start:end+1, :]
        else:
            sub_matrix = self.adj_matrix[start:end, :]
        if start == end:
            return 0
        for i in range(len(sub_matrix)):
            if np.any(sub_matrix[i] == 1):
                value += 1
        return value

    def intervener_complexity(self):
        complexity_val = 0
        for start, end in self.coord:
            distance = self.__inter_val(start, end)
            complexity_val += distance
        return complexity_val

    def sentence_length(self):
     if self.language != 'japanese' and self.language != 'chinese' and self.language != 'korean':
         ans = self.sentence.split()
         return len(ans)
     
     if self.language == 'japanese':
         tokenizer = Tokenizer()
         words_japanese = [token.surface for token in tokenizer.tokenize(self.sentence)]
         return len(words_japanese)
     
     if self.language == 'chinese':
         word_chinese = list(jieba.cut(self.sentence))
         return len(word_chinese)
     
     if self.language == 'korean':
         okt = Okt()
         words_korean = okt.morphs(self.sentence)
         return len(words_korean)