# Module 1: Introduction
 
In this module, we will learn what LLM and RAG are and
implement a simple RAG pipeline to answer questions 

## 1.1 Introduction to LLM and RAG

* LLM
* RAG
* RAG architecture
* Course outcome


## 1.2 Preparing the Environment

* Installing libraries
* Alternative: installing anaconda or miniconda

```bash
pip install tqdm notebook==7.1.2 openai elasticsearch pandas scikit-learn ipywidgets
```

## 1.3 Retrieval

* We will use the search engine we build in the [build-your-own-search-engine workshop](https://github.com/alexeygrigorev/build-your-own-search-engine): [minsearch](https://github.com/alexeygrigorev/minsearch)
* Indexing the documents
* Peforming the search


## 1.4 Generation with OpenAI


* Invoking OpenAI API
* Building the prompt
* Getting the answer


If you don't want to use a service, you can run an LLM locally
refer to [module 2](../02-open-source/) for more details.

In particular, check "2.7 Ollama - Running LLMs on a CPU" - 
it can work with OpenAI API, so to make the example from 1.4 
work locally, you only need to change a few lines of code.


## 1.4.2 OpenAI API Alternatives


[Open AI Alternatives](../awesome-llms.md#openai-api-alternatives)


## 1.5 Cleaned RAG flow


* Cleaning the code we wrote so far
* Making it modular

## 1.6 Searching with ElasticSearch


* Run ElasticSearch with Docker
* Index the documents
* Replace MinSearch with ElasticSearch

Running ElasticSearch:

```bash
docker run -it \
    --rm \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3
```

If the previous command doesn't work (i.e. you see "error pulling image configuration"), try to run ElasticSearch directly from Docker Hub:

```bash
docker run -it \
    --rm \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    elasticsearch:8.4.3
```

Index settings:

```python
{
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"} 
        }
    }
}
```

Query:

```python
{
    "size": 5,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["question^3", "text", "section"],
                    "type": "best_fields"
                }
            },
            "filter": {
                "term": {
                    "course": "data-engineering-zoomcamp"
                }
            }
        }
    }
}
```

We use `"type": "best_fields"`. You can read more about 
different types of `multi_match` search in [elastic-search.md](elastic-search.md).
