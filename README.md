# Information Retrieval Engine

## Modules used in the project

### Search Front-End
* search - the main search method of the engine, combine results from number of sub-searches.
* search_body - searches the body index.
* search_title - searches the title index.
* search_anchor - searches the anchor index.
* get_pagerank - returns the pagerank of a given wiki-id page.
* get_pageviews - returns the page views count of a given wiki article id
* search_config - searches using specific configuration.

### Backend
* Loads all the relevant files and indices saved in the storage cloud using the loader module.
* Handles all the search methods from the frontend.


### Loader
Contains all the functions for loading all needed elements from the storage.


### Helper Classes
* QueryPreprocessing - contains all functions to prepare a query and other files before the search.
* Calculator - contains calculation methods of different scores.
* ResultProcessor - contains all functions to prepare the result before sending back.

### Inverted Index GCP
All the files from assignment 3, of reading and writing inverted index.


### BM25
Modified BM25 class, which can return the BM25 score of a given document.
