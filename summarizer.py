import spacy #to build an extractive summarizer. 
import pytextrank #to use google's TextRank algorithm


# defining spacy pipelin but for that you need the install that model.
nlp = spacy.load('en_core_web_lg') 

# now let's use textrank
nlp.add_pipe("textrank")

text = "We propose a lightweight real-time sign language detection model, as we identify the need for such a case in videoconferencing. We extract optical flow features based on human pose estimation and, using a linear classifier, show these features are meaningful with an accuracy of 80%, evaluated on the Public DGS Corpus. Using a recurrent model directly on the input, we see improvements of up to 91% accuracy, while still working under 4 ms. We describe a demo application to sign language detection in the browser in order to demonstrate its usage possibility in videoconferencing applications."

summary = nlp(text)

for i in summary._.textrank.summary(limit_sentences = 2):
    print(i)