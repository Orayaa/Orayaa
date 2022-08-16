from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.corpora.dictionary import Dictionary
import itertools

from collections import defaultdict
articles = [] 


for i in range(10) : 
    #Read TXT file  C:\Users\ADMIN\Desktop\6206021610081 
    f = open(f"wiki_article_{i}.txt", "r") 
    article = f.read() 
    # Tokenize the article: tokens 
    tokens = word_tokenize(article) 
    # Convert the tokens into lowercase: lower_tokens 
    lower_tokens = [t.lower() for t in tokens] 
    # Retain alphabetic words: alpha_only 
    alpha_only = [t for t in lower_tokens if t.isalpha()] 
    # Remove all stop words: no_stops 
    no_stops = [t for t in alpha_only if t not in stopwords.words('english')] 
    # Instantiate the WordNetLemmatizer 
    wordnet_lemmatizer = WordNetLemmatizer() 
    # Lemmatize all tokens into a new list: lemmatized 
    lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops] 
    #list_article 
    articles.append(lemmatized) 

#print(articles[0]) 

dictionary = Dictionary(articles)

computer_id = dictionary.token2id.get("computer")
print(computer_id)
print(dictionary.get(computer_id))

# Create a Corpus: corpus
corpus = [dictionary.doc2bow(a) for a in articles]


# Save the second document: doc
doc = corpus[0]

# Sort the doc for frequency: bow_doc
bow_doc = sorted(doc, key=lambda w: w[1], reverse=True)
#print(corpus)

total_word_count = defaultdict(int)
for word_id, word_count in itertools.chain.from_iterable(corpus):

    total_word_count[word_id] += word_count
sorted_word_count = sorted(total_word_count.items(), key=lambda w: w[1],reverse=True)

for word_id, word_count in sorted_word_count[:20]:
    print(dictionary.get(word_id), word_count)
#for word_id, word_count in bow_doc[:10]:
#    print(dictionary.get(word_id), word_count)