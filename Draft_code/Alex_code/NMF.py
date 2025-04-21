import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np
from collections import defaultdict
# Step1: Read and Input
def load_20newsgroups_from_folder(root_dir):
    data = []
    labels = []
    label_names = []

    for label_idx, category in enumerate(sorted(os.listdir(root_dir))):
        category_path = os.path.join(root_dir, category)
        if not os.path.isdir(category_path):
            continue
        label_names.append(category)

        filepaths = glob.glob(os.path.join(category_path, '*'))
        print(f"Loading {len(filepaths)} files from category '{category}'")

        for filepath in filepaths:
            try:
                with open(filepath, encoding='latin-1') as f: #UTF-8 didn't work
                    text = f.read().strip()
                    if text:
                        data.append(text)
                        labels.append(label_idx)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")

    return data, labels, label_names
#Step 2: Set up the path and loading:
root_dir = '/Users/a86138/Desktop/PRIMES/20_newsgroups'
documents, labels, label_names = load_20newsgroups_from_folder(root_dir)
print(f"\n✅ Loaded {len(documents)} documents from {len(label_names)} categories.\n")

#Step 3: TF-IDF
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
tfidf = vectorizer.fit_transform(documents)
feature_names = vectorizer.get_feature_names_out()

#Step 4: NMF
n_topics = 10
nmf = NMF(n_components=n_topics, random_state=42)
W = nmf.fit_transform(tfidf) #Topics
H = nmf.components_#Words

# Step 5: Keywords for each topic
n_top_words = 10
print("Top Words per Topic (NMF)")
topic_top_words = []
for topic_idx, topic in enumerate(H):
    top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
    topic_top_words.append(top_words)
    print(f"Topic #{topic_idx + 1}: {', '.join(top_words)}")

# Step 6: Term-Document Map:
def build_term_doc_map(docs, vocab):
    term_doc_map = defaultdict(set)
    for doc_idx, doc in enumerate(docs):
        tokens = set(doc.lower().split())
        for token in tokens:
            if token in vocab:
                term_doc_map[token].add(doc_idx)
    return term_doc_map

# Step 7: Topic Coherence Functions:
def compute_topic_coherence(top_words, term_doc_map):
    score = 0.0
    M = len(top_words)
    for m in range(1, M):
        for l in range(0, m):
            word_m = top_words[m]
            word_l = top_words[l]
            docs_m = term_doc_map[word_m]
            docs_l = term_doc_map[word_l]
            D_vl = len(docs_l)
            D_vm_vl = len(docs_m & docs_l)
            score += np.log((D_vm_vl + 1) / (D_vl + 1e-10))  # divide 0 error
    return score

# Step 8: Coherence score for each topic:
print("\n Topic Coherence Scores:")
vocab_set = set(feature_names)
term_doc_map = build_term_doc_map(documents, vocab_set)

coherence_scores = []
for idx, top_words in enumerate(topic_top_words):
    score = compute_topic_coherence(top_words, term_doc_map)
    coherence_scores.append(score)
    print(f"Topic #{idx + 1}: Coherence = {score:.4f}")

avg_coherence = np.mean(coherence_scores)
print(f"\n Average Topic Coherence: {avg_coherence:.4f}")

#Why positive numbers for the results. (All values should be negative)