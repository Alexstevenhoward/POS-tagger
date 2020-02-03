#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 14:20:09 2017

@author: alexanderhoward
"""
from nltk import DefaultTagger,UnigramTagger,BigramTagger,TrigramTagger
from nltk.corpus.reader import TaggedCorpusReader
from nltk.probability import FreqDist
from numpy import mean

mypath = "/Users/alexanderhoward/Desktop/"
EstonianCorpus = TaggedCorpusReader(mypath,"estonianSmall.txt", encoding="latin-1")
sentences = EstonianCorpus.tagged_sents()

tags = [tag for sent in sentences[:1000] for _,tag in sent]
tagF = FreqDist(tags)
mostFreq = tagF.max()

def crossValidationPosTagger(k,sentences,mostFreq):
    length = len(sentences)
    results = []
    for i in range(k):
        startIndex = length*i//k 
        endIndex = length*(i+1)//k
        testing = sentences[startIndex:endIndex]
        training = sentences[0:startIndex] + sentences[endIndex:]
        customTagger = DefaultTagger(mostFreq)
        customTagger.tag_sents(sentences[:100])
        tagger1 = UnigramTagger(training, backoff=customTagger)
        tagger2 = BigramTagger(training, backoff=tagger1)
        tagger3 = TrigramTagger(training, backoff=tagger2)
        score = tagger3.evaluate(testing)
        results.append(score)
    return results
       
score = crossValidationPosTagger(3,sentences,mostFreq)
print score
print mean(score)