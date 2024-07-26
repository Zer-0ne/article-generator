import pandas as pd
import cohere 

co = cohere.Client("T7E5YdqYVduosUnRrTAGvimDFbrSXFSdUOmk3nHA") 

df = pd.read_csv('https://raw.githubusercontent.com/cohere-ai/notebooks/main/notebooks/data/bbc_news_subset.csv', delimiter=',')
df.drop(['ArticleId'], axis=1, inplace=True)

MAX_CHARS = 300

def shorten_text(text):
    return text[:MAX_CHARS]

df['Text'] = df['Text'].apply(shorten_text)

articles = df['Text'].tolist()
categories = df['Category'].unique().tolist()

output = co.embed(model='embed-english-v3.0', input_type='search_document', texts=articles)
embeds = output.embeddings

EX_PER_CAT = 5
ex_texts = []
ex_labels = []
for category in categories:
    df_category = df[df['Category'] == category]
    samples = df_category.sample(n=EX_PER_CAT, random_state=42)
    ex_texts += samples['Text'].tolist()
    ex_labels += samples['Category'].tolist()

from cohere import ClassifyExample
examples = [ClassifyExample(text=txt, label=lbl) for txt, lbl in zip(ex_texts, ex_labels)]
