from flask import Flask, request, jsonify
import inverted_index_gcp
import nltk
from nltk.corpus import stopwords
import threading
from collections import Counter
import math
import re
import pickle

# uncomment for colab
# import gcsfs
# fs = gcsfs.GCSFileSystem(project='ir-assignment-3-370613')

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        self.N = 6348910

        # uncomment for colab
        # self.body_index_path = '318932639/postings_gcp_text'
        # self.title_index_path = '318932639/postings_gcp_title'
        # self.anchor_index_path = '318932639/postings_gcp_anchor'
        # self.dl_path = '318932639/dl/dl.pkl'
        # self.dt_path = '318932639/dt/dt.pkl'

        self.body_index_path = 'postings_gcp_text'
        self.title_index_path = 'postings_gcp_title'
        self.anchor_index_path = 'postings_gcp_anchor'
        self.dl_path = 'dl/dl.pkl'
        self.dt_path = 'dt/dt.pkl'
        
        self.index_body = inverted_index_gcp.InvertedIndex.read_index(self.body_index_path, 'index')
        self.index_title = inverted_index_gcp.InvertedIndex.read_index(self.title_index_path, 'index')
        self.index_anchor = inverted_index_gcp.InvertedIndex.read_index(self.anchor_index_path, 'index')
        # with fs.open(self.dl_path, 'rb') as f: # uncomment for colab
        with open(self.dl_path, 'rb') as f:
            self.DL = pickle.load(f)
            self.DL_LEN = len(self.DL)
        # with fs.open(self.dt_path, 'rb') as f: # uncomment for colab
        with open(self.dt_path, 'rb') as f:
            self.DT = pickle.load(f)

        self.body_res = []
        self.title_res = []
        self.anchor_res = []
        
        self.stemmer = nltk.PorterStemmer()
        
        nltk.download('stopwords')

        self.CALLED_BY = False
        
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    app.CALLED_BY = True

    body_thread = threading.Thread(search_BM25(), args=query)
    body_thread.start()
    # body_thread = threading.Thread(search_body(), args=query)
    # body_thread.start()
    # title_thread = threading.Thread(search_title(), args=query)
    # title_thread.start()
    anchor_thread = threading.Thread(search_anchor(), args=query)
    anchor_thread.start()

    id_ranking = Counter()
    body_thread.join()
    for i in range(len(app.body_res)):
        id_ranking[app.body_res[i][0]] += 30 / (i + 1) # TODO: Check what is the wanted weight

    # title_thread.join()
    # for i in range(len(app.title_res)):
    #     id_ranking[app.title_res[i][0]] += 5

    anchor_thread.join()
    for i in range(len(app.anchor_res)):
        id_ranking[app.anchor_res[i][0]] += 10
    # print('id_ranking.most_common(100)', id_ranking.most_common(100))
    for id, value in id_ranking.most_common(100):
        try:
            res.append((id, app.DT[id]))
        except:
            continue
    app.body_res = []
    app.title_res = []
    app.anchor_res = []

    app.CALLED_BY = False

    # END SOLUTION
    return jsonify(res)

@app.route("/search_BM25")
def search_BM25(query=None):
    ''' Exactly the same as search body method only the similarity method used here is BM25
        and not the classic tf-idf and cosine similarity.
        the method is a utility method for search.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    if query is None:
        query = request.args.get('query', '')
    if len(query) == 0:
        if not app.CALLED_BY:
            return jsonify(res)
        return res
    # BEGIN SOLUTION
    query = tokenize(query)
    similarities = Counter()

    b = 0.75
    k1 = 2
    avgdl = 341
    for term in query:
        posting_list = app.index_body.read_posting_list(term, app.body_index_path)
        if posting_list:
            df = app.index_body.df[term]
            idf = math.log((app.DL_LEN - df + 0.5) / (df + 0.5) + 1)
            for id, fr in posting_list:
                if id in app.DL:
                    pl = app.DL[id]
                    similarities[id] += idf * ((fr * (k1 + 1) / (fr + k1 * (1 - b + (b * (pl / avgdl))))))

    most = similarities.most_common(100)

    for id, value in most:
        try:
            res.append((id, app.DL[id]))
        except:
            continue
    # END SOLUTION
    if not app.CALLED_BY:
        return jsonify(res)

    app.body_res = res
    return

@app.route("/search_body")
def search_body(query=None):
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    if query is None:
        query = request.args.get('query', '')
    if len(query) == 0:
        if not app.CALLED_BY:
            return jsonify(res)
        return res
    # BEGIN SOLUTION
    query_word_count = Counter(query)
    similarities = Counter()

    for term in query:
        posting_list = app.index_body.read_posting_list(term, app.body_index_path)
        idf = math.log2(app.DL_LEN / app.index_body.df[term])
        for id, fr in posting_list:
            if id in app.DL:
                pl = app.DL[id]
                tf = (fr/pl)
            else:
                tf = 0
            weight = tf * idf
            similarities[id] += (weight * query_word_count[term])

    most = similarities.most_common(100)

    for id, value in most:
        try:
            res.append((id, app.DL[id]))
        except:
            continue
    # END SOLUTION
    if not app.CALLED_BY:
        return jsonify(res)

    app.body_res = res
    return

@app.route("/search_title")
def search_title(query=None):
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO 
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. For example, a document 
        with a title that matches two distinct query words will be ranked before a 
        document with a title that matches only one distinct query word, 
        regardless of the number of times the term appeared in the title (or 
        query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        if not app.CALLED_BY:
            return jsonify(res)
        return res
    # BEGIN SOLUTION
    #
    # for word in query:
    #     stemmed = app.stemmer.stem(word)
    #     if stemmed not in query:
    #         query.append(stemmed)
    #
    # posting_lists = app.index_title.get_posting_lists(query, base_dir=app.title_index_path)
    #
    # for id, value in posting_lists:
    #     try:
    #         res.append((id, app.DT[id]))
    #     except:
    #         continue

    # END SOLUTION
    if not app.CALLED_BY:
        return jsonify(res)

    app.title_res = res
    return

@app.route("/search_anchor")
def search_anchor(query=None):
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment 
        3 (GCP part) to do the tokenization and remove stopwords. For example, 
        a document with a anchor text that matches two distinct query words will 
        be ranked before a document with anchor text that matches only one 
        distinct query word, regardless of the number of times the term appeared 
        in the anchor text (or query). 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    if query is None:
        query = request.args.get('query', '')
    if len(query) == 0:
        if not app.CALLED_BY:
            return jsonify(res)
        return res
    # BEGIN SOLUTION

    posting_lists = app.index_anchor.get_posting_lists(query, base_dir=app.anchor_index_path)
    pls_words_ids = [id for pl in posting_lists for id, _ in pl]
    sorted_ids_by_count = Counter(pls_words_ids).most_common()
    for id, count in sorted_ids_by_count:
        if id in app.DT:
            res.append((id, app.DT[id]))

    # END SOLUTION
    if not app.CALLED_BY:
        return jsonify(res)

    app.anchor_res = res
    return

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    # END SOLUTION
    return jsonify(res)


def tokenize(query):
    """
    :param query: string - the query provided
    :return: list of tokens after removing stopwords and punctuation
    """
    english_stopwords = frozenset(stopwords.words('english'))
    corpus_stopwords = ["category", "references", "also", "external", "links",
                        "may", "first", "see", "history", "people", "one", "two",
                        "part", "thumb", "including", "second", "following",
                        "many", "however", "would", "became"]

    all_stopwords = english_stopwords.union(corpus_stopwords)
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

    tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
    return [token for token in tokens if token not in all_stopwords]


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
