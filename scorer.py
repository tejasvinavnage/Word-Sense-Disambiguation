# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 21:47:36 2019

@author: tejasvi
"""

import sys
import re
from sklearn.metrics import confusion_matrix
import pandas as pd

#reading data
#answers = open(sys.argv[1], "r")
#gold_std = open(sys.argv[2], "r")

answers = open("file_op12.txt", "r")
gold_std = open("line-answers.txt", "r")


#extracting sense-id from answers
answers2 = []
for line in answers:
    answers2.append(line)

str1 = ''.join(answers2)
sense_ans = []
pattern = re.compile(r'senseid="(.*)"')
matches = pattern.finditer(str1)
for match in matches:
    sense_ans.append(match.group(1))

#extracting sense-id from gold standard

gold_std2 = []
for line in gold_std:
    gold_std2.append(line)

str1 = ''.join(gold_std2)
sense_gold = []
pattern = re.compile(r'senseid="(.*)"')
matches = pattern.finditer(str1)
for match in matches:
    sense_gold.append(match.group(1))

#compute Accuracy
sum = 0
for i in range(len(sense_gold)):
    if(sense_ans[i]==sense_gold[i]):
        sum+=1
accuracy = sum/len(sense_gold)
print("Accuracy = "+str(accuracy))

#create Confusion Matrix

c = confusion_matrix(sense_gold,sense_ans)
c = pd.DataFrame(c,index=["phone","product"],columns=["phone","product"])

print("\n Confusion Matrix:\n")
print(c)