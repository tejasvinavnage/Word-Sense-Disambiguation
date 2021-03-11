# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 03:58:24 2019

@author: tejasvi
"""
import sys
import re
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import nltk
from math import log
#nltk.download('wordnet')

#reading data=====================================================================
train = open('line-train.xml', "r")
test = open('line-test.xml', "r")
my_list = open('op.txt', "r+")
file_op = open('file_op4.txt', "w")

#cleaning file====================================================================
def clean_sentence(s):
    cleaned_sentences = []
    for sentence in s:
        cleaned_sentences.append(re.sub(r'(<s>|<\/s>)','',sentence))
    return cleaned_sentences

def lemmatize_sentence(s):
    lemmatized_sentences = []
    lemmatizer = WordNetLemmatizer()
    for sentence in s:
        lem_sent = lemmatizer.lemmatize(sentence)
        lemmatized_sentences.append(lem_sent)
    return lemmatized_sentences

def tokenize_sentence(s):
    tokenized_sentences = []
    tokenizer = RegexpTokenizer(r'\w+')
    for sentence in s:
        tok_sent = tokenizer.tokenize(sentence)
        tokenized_sentences.append(tok_sent)
    return tokenized_sentences

def post_words(string):
    post_words = []
    p2 = re.compile(r'<head>(.*)</head>(.*)')
    matches2 = p2.finditer(string)
    for match2 in matches2:
        post_words.append(match2.group(2))
    return post_words

def prev_words(string):
    prev_words = []
    pattern1 = re.compile(r'(.*)<head>(.*)</head>')
    matches1 = pattern1.finditer(string)
    for match1 in matches1:
        prev_words.append(match1.group(1))
    return prev_words


#reading training file============================================================
train_raw, senses, sentences = [], [], []
for line in train:
    if re.match(r'(.*)senseid="(.*)"', line):
        senses.append(line)
    elif re.match(r'<s>(.*)<\/s>', line):
        sentences.append(line)
    train_raw.append(line)


#reading test file================================================================
test_list = []
for line in test:
    test_list.append(line)

#getting sense extracted from sense===============================================
sense_extracted = []
for s in senses:
    matches = re.compile(r'senseid="(.*)"').finditer(s)
    for match in matches:
        sense_extracted.append(match.group(1))

#Compute Frequency Distribution
fdist_sense1 = FreqDist(sense_extracted)
        
#extracting ambiguous words=======================================================
train_words = []
for line in sentences:
    matches = re.compile(r'<head>(.*)</head>').finditer(line)
    for match in matches:
        train_words.append(match.group(1))
        
train_list =[(sense_extracted[i],train_words[i], sentences[i]) for i in range(0,len(sense_extracted))]

prevfeat, posfeat = [], []

prevfeat = prev_words(''.join(sentences))
clean_prevfeat = clean_sentence(prevfeat)
lem_prevfeat = lemmatize_sentence(clean_prevfeat)
tok_prevfeat = tokenize_sentence(lem_prevfeat)

posfeat =  post_words(''.join(sentences))
clean_posfeat = clean_sentence(posfeat)
lem_posfeat = lemmatize_sentence(clean_posfeat)
tok_posfeat = tokenize_sentence(lem_posfeat)

#Selecting 1 prev word============================================================
prevtrain = []
for line in tok_prevfeat:
    prevtrain.append(line[-1:])

#Selecting 1 posr word===========================================================
postrain = []
for line in tok_posfeat:
    postrain.append(line[:1])
    
tot_feat = [x+y for x,y in zip(prevtrain,postrain)]

#creating bag of words for each sense=============================================
phone_bow = []
product_bow = []

for i in range(len(sense_extracted)):
    if(sense_extracted[i]=="phone"):
        phone_bow.append(tot_feat[i])
        phone_bow.append(tot_feat[i])
    else:
        product_bow.append(tot_feat[i])
        product_bow.append(tot_feat[i])
        
temp =[]
for item in phone_bow:
    temp += item
phone_bow = temp

temp = []
for item in product_bow:
    temp += item
product_bow = temp

#Freq Distributions===============================================================
fdist_phone = FreqDist(phone_bow)
fdist_product = FreqDist(product_bow)

total = phone_bow + product_bow
fdist_total = FreqDist(total)        

#testing===========================================================================
#extracting instance id
str2 = ''.join(test_list).lower()
t_sense1 = []
pattern = re.compile(r'instance id="(.*)"')
matches = pattern.finditer(str2)
for match in matches:
    t_sense1.append(match.group(1))

#extracting ambigous word
t_sense2 = []
pattern = re.compile(r'<head>(.*)</head>')
matches = pattern.finditer(str2)
for match in matches:
    t_sense2.append(match.group(1))

#extracting future words===========================================================
tprevfeat = prev_words(str2)
clean_tprevfeat = clean_sentence(tprevfeat)
lem_tprevfeat = lemmatize_sentence(clean_tprevfeat)
tok_tprevfeat = tokenize_sentence(lem_tprevfeat)

tposfeat =  post_words(str2)
clean_tposfeat = clean_sentence(tposfeat)
lem_tposfeat = lemmatize_sentence(clean_tposfeat)
tok_tposfeat = tokenize_sentence(lem_tposfeat)

#selecting prev 1 word
t_past2 = []
for line in tok_tprevfeat:
    t_past2.append(line[-1:])

#selecting 1 future words
t_fut2 = []
for line in tok_tposfeat:
    t_fut2.append(line[:1])

#combining bag of test words
t_combined = [x+y for x,y in zip(t_past2,t_fut2)]


#calculating the decision list=====================================================
output = []
temp = 0
for j in range(len(t_combined)):
    my_list.write("line =" + str(j+1) + ":\n")
    prob_phone = []
    prob_product = []
    for i in range(len(t_combined[j])):
        #sense phone
        if t_combined[j][i] in total:
            print("sense phone==========================")
            print("word <",t_combined[j][i],"> frequency:",fdist_phone[t_combined[j][i]])
            print("prob_word = fdist_phone[t_combined[j][i]]/fdist_total[t_combined[j][i]]=",fdist_phone[t_combined[j][i]],"/",fdist_total[t_combined[j][i]])
            prob_word = (fdist_phone[t_combined[j][i]]/fdist_total[t_combined[j][i]])
            print("prob_feature= fdist_phone[t_combined[j][i]]/fdist_sense1[phone]=",fdist_phone[t_combined[j][i]],"/",fdist_sense1["phone"])
            prob_feature = (fdist_phone[t_combined[j][i]]/fdist_sense1["phone"])
            print("prob_word", prob_word,"prob_feature", prob_feature)
            prob_phone.append(prob_word * prob_feature)
#            print("."*40)
        else:
            print("else: fdist_sense1[phone]/len(sense_extracted)", fdist_sense1["phone"],"/",len(sense_extracted))
            prob_phone.append(fdist_sense1["phone"]/len(sense_extracted))
            temp+=1

        #sense product
        if t_combined[j][i] in total:
            print("sense product==========================")
            print("word <",t_combined[j][i], "> frequency:", fdist_product[t_combined[j][i]])
            print("prob_word = fdist_product[t_combined[j][i]]/fdist_total[t_combined[j][i]] =",fdist_product[t_combined[j][i]], "/", fdist_total[t_combined[j][i]])
            prob_word = fdist_product[t_combined[j][i]]/fdist_total[t_combined[j][i]]
            print("prob_feature = fdist_product[t_combined[j][i]]/fdist_sense1[product]",fdist_product[t_combined[j][i]],"/",fdist_sense1["product"])
            prob_feature = fdist_product[t_combined[j][i]]/fdist_sense1["product"]
            print("prob_word",prob_word,"prob_feature",prob_feature)
            prob_product.append(prob_word * prob_feature)
        else:
            print("fdist_sense1[product]/len(sense_extracted)", fdist_sense1["product"],"/",len(sense_extracted))
            prob_product.append(fdist_sense1["product"]/len(sense_extracted))
            temp+=1
    print("phone prob:",prob_phone)
    print("product prob",prob_product)
        
    final_phone = 1
    final_product = 1

    for item in prob_phone:
        print("final_phone:",final_phone, "*item:",item)
        final_phone = final_phone * item
        print("=final_phone", final_phone)
    for item in prob_product:
        print("final_product:",final_product, "*item:",item)
        final_product = final_product * item
        print("=final_product", final_product)
        
    if(final_phone>final_product):
        output.append("phone")
        print("phone------------------", final_phone)
    else:
        output.append("product")
        print("product----------------", final_product)
        
    print("#"*60)
    
    
#output
for i in range(len(output)):
#    print("<answer instance=\""+str(t_sense1[i])+"\" senseid=\""+str(output[i])+"\"/>")
    s = "<answer instance=\""+str(t_sense1[i])+"\" senseid=\""+str(output[i])+"\"/>"
    file_op.write(s+'\n')