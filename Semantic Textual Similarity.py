#Import Required Packages

import pandas as pd
import os
import gensim
import spacy

#Load library include 1 million word vectors 
nlp = spacy.load('en_core_web_lg')

#Enter File Path for read the document

read_file = "E:\Shruti\Study\MBA\DRIVES\Company Task\Precily Assessment(28 May)\Text_Similarity_Dataset.csv"

#Read csv file

text_data = pd.read_csv(read_file)

#Print Extracted csv File

print(text_data)

#Tokenize reference text and check text 
ref_text = [nlp(row) for row in text_data['text2']]
whole_text = [nlp(row) for row in text_data['text1']]

#Create dictionary to store similarity score
similarity_score = []

#Create dictonary to store unique_id
unique_id = []
for i in range(len(whole_text)):
    #Calculate Similarity Score of refernce and check text
    sim_score = whole_text[i].similarity(ref_text[i])
    #Append Similarity Score in respective dictionary
    similarity_score.append(sim_score)
    unique_id.append(i)
    #Create DataFrame to store Similarity Score along with Unique_ID
    sim_docs = pd.DataFrame(list(zip(unique_id, similarity_score)), columns = ['Unique_ID', 'Similarity_Score'])

#Display the DataFrame storing similarity score
print(sim_docs)

#File path for writing the similarity score document
write_file = "Similarity_Score.csv"

#Write the Dataframe into csv file
sim_docs.to_csv(write_file, index = False)
    
    
    
