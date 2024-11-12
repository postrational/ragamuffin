# Ragamuffin - Chat with your documents 🐈

Ragamuffin is a [RAG][rag]-powered chat agent which can access documents, articles from your [Zotero][zotero] library
and code from any GitHub repository.

You can ask questions, and the agent will respond using information from documents in your library.
It will also display the sources used to generate each answer.

![Zotero Chat](screenshot.png)

Ragamuffin is built with the amazing [LlamaIndex][llama-index], [SBERT][sbert], and [Transformers][transformers]
libraries and uses [Gradio][gradio] for the user interface.

## Installation

### Pre-requisites

Ragamuffin requires Python 3.10 or higher. It's recommended to use a virtual environment to install the package.

### Install Ragamuffin in a virtual environment

    $ python3 -m venv venv
    $ source venv/bin/activate
    (venv) $ pip install --upgrade git+https://github.com/postrational/ragamuffin.git

## Usage

In order to use Ragamuffin, you need to generate an [OpenAI API key][openai-key].
Please set this as an environment variable before running the `muffin` commands. 

    $ export OPENAI_API_KEY=sk-proj-XXXX........

### Create a chat agent based on a directory of documents

You can generate a RAG index based on a directory of files (e.g. TXT, PDF, EPUB, etc.).

Use the `muffin` command to generate an agent named `my_agent` based on the documents in `/path/to/my/documents/`:

    (venv) $ muffin generate from_files my_agent /path/to/my/documents/

Start the chat agent using the following command:

    (venv) $ muffin chat my_agent

### Create a chat agent based on your Zotero library

In order to use Ragamuffin with Zotero, you need to generate a [Zotero API key][zotero-key] and 
an [OpenAI API key][openai-key]. Set these as environment variables before running `muffin`. 

    $ export OPENAI_API_KEY=sk-proj-XXXX........
    $ export ZOTERO_LIBRARY_ID=1234567
    $ export ZOTERO_API_KEY=XXXX........
    (venv) $ muffin generate from_zotero zotero_agent

This will generate a RAG index based on all papers in your Zotero library.

You can also create an agent for a specific collection in your Zotero library using the `--collection` option:

    (venv) $ muffin generate from_zotero zotero_agent --collection "My Collection"

Later, you can chat with Ragamuffin using the `muffin chat` command:

    (venv) $ muffin chat zotero_agent

### Create a chat agent based on a Git repository

If you want to learn about a specific codebase, you can generate a RAG index based on a GitHub repository.

    (venv) $ muffin generate from_git my_agent https://github.com/postrational/ragamuffin/

If you want to use a specific branch, tag or commit, you can use the `--ref` option:

    (venv) $ muffin generate from_git poetry https://github.com/python-poetry/poetry --ref 1.8.4

### Chat with the agent

You can chat with the agent using the `muffin chat` command:

    (venv) $ muffin chat my_agent

### List created agents

You can list all the agents created using the `muffin` command:

    (venv) $ muffin agents

### Delete an agent

You can delete an agent using the `muffin` command:

    (venv) $ muffin delete my_agent

## Use Cassandra for agent storage

You can use [Cassandra DB][cassandra] for more efficient storage of the RAG indexes of your agents.

Install Cassandra DB and create a keyspace named `ragamuffin`.

On a Mac, you can install Cassandra using [Homebrew][brew]:

    $ brew install cassandra
    $ brew services start cassandra
    $ cqlsh
    cqlsh> CREATE KEYSPACE ragamuffin WITH replication = {'class': 'SimpleStrategy', 'replication_factor': 1};

To use Cassandra for storage, set the `RAGAMUFFIN_STORAGE_TYPE` environment variable to `cassandra`:

    $ export RAGAMUFFIN_STORAGE_TYPE=cassandra


[brew]: https://brew.sh/
[cassandra]: https://cassandra.apache.org/
[gradio]: https://www.gradio.app/
[llama-index]: https://www.llamaindex.ai/
[openai-key]: https://platform.openai.com/api-keys
[rag]: https://en.wikipedia.org/wiki/Retrieval-augmented_generation
[sbert]: https://sbert.net/
[transformers]: https://huggingface.co/transformers/
[zotero-key]: https://www.zotero.org/settings/security#applications
[zotero]: https://www.zotero.org/
