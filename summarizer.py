import spacy #to build an extractive summarizer. 
import pytextrank #to use google's TextRank algorithm


# defining spacy pipelin but for that you need the install that model.
nlp = spacy.load('en_core_web_lg') 

# now let's use textrank
nlp.add_pipe("textrank")

text = input('Enter your article: ')

summary = nlp(text)

print("Summarized: ", end=' ')
for i in summary._.textrank.summary(limit_sentences = 2):
    print(i)