# Facilitating Observation and Classification Using Specimens (FOCUS)

This project is my Computer Science capstone project, a web service for zero-shot image classification of organisms using Contrastive Language-Image Pretraining (CLIP).

## Features
- Zero-shot image classification using OpenCLIP
- Fast inference, even with huge taxonomies 
- Configurable prompt templates for labels to improve classification accuracy
- On-disk vector index using Annoy for scalable inference
- FastAPI for web service deployment

## Installation

### Prerequisites
- Python 3.10+
- Conda (optional)
- Dependencies listed in `environment.yaml`

### Setup
1. Clone the repository
   ```sh
   git clone https://github.com/nathaniel-ribeiro/taxon-classification.git
   cd taxon-classification
   
2. Create a conda environment from the provided `environment.yaml` file
   ```sh
   conda env create -f environment.yaml

3. Activate your new conda environment
   ```sh
   conda activate UVACapstone

### Configuration
Hyperparameters can be edited in `config.yaml`. Please note that if these values are modified, `taxonomy_embeddings.py` 
must be re-run. Failure to do so will lead to unpredictable results.
* checkpoint: OpenCLIP pretrained checkpoint. See [OpenCLIP documentation](https://github.com/mlfoundations/open_clip) for help.
* index_filepath: path to save on-disk vector index to.
* num_trees: number of random projections to build vector index with. See [Annoy documentation](https://github.com/spotify/annoy) for tips.
* prompt_templates: a list of templates to append each classification label to. Acts as basic "prompt engineering" to improve performance. See <link to CLIP prompt engineering> for more examples. Note that the vector index stores len(taxonomy) * len(prompt_templates) vectors so more templates results in possibly more accurate classification at the cost of larger indices.

## Running

### Building taxonomy
A taxonomy is provided for you in this repository under `species_with_taxonomic_info.csv`. You may 
use this file as is or supply your own .csv file in the working directory. At minimum, your file should have 'vernacularName' and 'scientificName' headers.
The provided file is created from the full iNaturalist taxonomy. Note that the ID/taxonID field is unique to each platform
that supplies a taxonomy (iNaturalist, GBIF, etc.) and are NOT interchangeable.

A quirk of Annoy is that only the numeric indices (0, 1, 2, 3 ... n-1) of the items are stored in the index. Thus, the
script provided also produces a persistent mapping of these indices to the taxon information and the engineered prompt.

### Building vector index
Zero-shot classification requires comparing the image embedding vector to one or more label embedding vectors.
With a small number of output classes, it is fast enough to rerun the labels through the text encoder for each call
to the classify method. 

However, taxonomies are often massive, with the provided species-level dataset numbering in the hundreds of thousands of rows. 
Rerunning each species label through the text encoder would be prohibitively slow, and an in-memory cache may be too large to fit
in RAM. This repository precomputes all label embeddings (assumed to be static or updated infrequently) and stores them in a persistent vector database (Annoy). 
Although not as fast as libraries like FAISS, Annoy's persistent structure allows an index to be mmap'd into multiple processes 
and exploits paging to only bring required portions of the index into RAM. This allows for arbitrarily large taxonomies and 
prompt templates while being cognizant of hardware constraints.

Note that building the index is a very slow process (taking hours to possibly days) but only needs to be run once. If
any hyperparameters are changed or if the taxonomy grows, the index must be rebuilt. Here's an example of how to build
the index:

1.
   ```sh
   python3 taxonomy_embeddings.py

### Running web service
After these two steps, you're ready to run your webservice! Here's an example of how to run it locally.
<Add uvicorn script here>

### Classifying an image
Uploading an image is easy! Simply make an HTTP POST request to the `/classify` endpoint with your image file.
In this example, we will use the `curl` command to upload an image of a snipe saved under `snipe.jpg`.
Feel free to edit this command to upload whatever image you want. You can also use another command/app to make the POST
request.

1.
   ```sh
   curl -X POST -F "file=@snipe.jpg" http://127.0.0.1:8000/classify/

The returned JSON will look something like:

### Coming Soon
- Fine-tuning script
  - Options to use LoRA/QLoRA
  - Options to freeze either text or image encoder
  - Options to change hyperparameters like learning rate, betas, momentum, weight decay, epochs, batch size, and more
- Deploying to the cloud
- Benchmarked results
  - iNaturalist competition data results + scripts to replicate results
- Multi GPU support

### Contribute
As this is an individual project for a graduation requirement, I am not currently accepting contributions. I will open
this repository for contributions after December 2025 when I graduate. I welcome feedback though!

### Contact
Feel free to [email me](mailto:nathaniel.eldred.ribeiro@gmail.com) with any questions, comments, or concerns you might have and I will do my best to get back to you in a timely fashion. 
You can also open an issue on my GitHub.
