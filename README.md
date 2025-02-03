# Context
Machine Learning project building a context similarity system which uses a fine tuned BERT model in order to detect semantic academic context similarity. Integrated with text, document and database support.
There are different pages.

The first is the index page which provided with three options -
  A. Compare two pieces of text using the fine tuned model and get the semantic similarity. This wont be affected much if you change the wordings or paraphrase( an advantage of our model).
  B. Compare two documents. Upload them and compare using the same fine tuned model. Along with this generate a explanation on what the two documents have and the scores that represent them- the reasons for them and how they compare with each other. This is done using ollama, a local LLM model.
  C. Compare using database. You can either input your own database info(mongodb) and then compare with all the documents present there and it will fetch the scores along with respective documents. You can also add the papers yourself to the mongodb database( use your own db name, collections in the code) and then comapare with them. This works for local database as well as cloud(online) mongodb database.

This is an open source project. Free to use, just add credits. Thank you
