# BGU-IR-Project
## Author:
Tali Narovlyansky

---
In the following Information Retrieval project we asked to build a search engine for English Wikipedia.

## Data structures:

In order to speed up the calculations, I used two pre-calculated data structures:
1. ***DL (Document Length):*** a dictionary that stored the length of each document, allowing the search engine to quickly retrieve the length of a document without having to go through the entire document.
2. ***DT (Document Title):*** a dictionary that stored the title of each document, which helped to display the search results in a more informative way to the end user.

And those Inverted Indexes:
3. Body Inverted Index.
4. Anchor text Inverted Index.

All data structures were created on a cluster in GCP and stored in a bucket.

## Functionality:

To issue a query navigate to a URL like: http://YOUR_SERVER_DOMAIN/search?query=hello+world where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io if you're using ngrok on Colab or your external IP on GCP.

### search:
Returns up to a 100 of the best search results for the query. This is 
        my best search engine. I used a mix of BM-25 and search_anchor.

### search_body:
Returns up to a 100 search results for the query using TFIDF OF THE BODY OF ARTICLES ONLY.

### search_anchor:
Returns ALL (not just top 100) search results that contain A QUERY WORD IN THE ANCHOR TEXT of articles, ordered in descending order of the NUMBER OF QUERY WORDS that appear in anchor text linking to the page.

