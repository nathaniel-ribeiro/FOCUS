import cloudpickle as pickle
import open_clip
import polars as pl
import torch
import yaml
from annoy import AnnoyIndex
from tqdm import tqdm
import os
import warnings
from pathlib import Path
from datetime import datetime
import subprocess
import argparse

#TODO: add the following flags
# deterministic: allows a seed to be set before building the index (maybe nix)
# build_on_disk: if True, builds the index on disk rather than in RAM
# overwrite_if_exists: if a file with name index_filepath already exists, the existing file will be IMMEDIATELY overwritten (dangerous!)
# backup_if_exists: if a file with name index_filepath already exists, first copy the old index and append a timestamp to the filename, then proceed
# abort_if_exists (default): if a file with name index_filepath already exists, log a warning and gracefully exit


#TODO: make species info an actual dtype
def build_prompts(species_info):
    '''
    Uses configured prompt_templates to conduct basic prompt engineering on species labels. (Appends the vernacular and scientific name to each prompt template)
    :param species_info: dict of information about a given species. Must at least contain keys 'vernacularName' and 'scientificName'
    :return: list of engineered prompts to store in index
    '''
    vernacular_and_scientific_name = species_info['vernacularName'] + " (" + species_info['scientificName'] + ")"
    engineered_prompts = [None] * len(prompt_templates)
    for i, template in enumerate(prompt_templates):
        engineered_prompts[i] = template.format(label=vernacular_and_scientific_name)
    return engineered_prompts

def get_embeddings(prompts):
    '''
    Encodes a given list of prompts using OpenCLIP tokenizer and text encoder.
    :param prompts: list of prompts
    :return: encoded features of each prompt
    '''
    with torch.no_grad():
        tokenized_text = tokenizer(prompts).to(device)
        text_features = model.encode_text(tokenized_text)
    return text_features

def backup_index():
    '''
    Creates a backup of the configured index_filepath. The backup will be saved as {index_filepath}_backup_{timestamp}
    to avoid name conflicts. The new index will be stored in index_filepath. Aborts without altering the existing index
    if an error occurs while backing up.
    '''
    try:
        current_time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        backup_index_filepath = Path(index_filepath).stem + "_backup_" + current_time_string + Path(index_filepath).suffix
        backup_index_to_taxon_filepath = Path(index_to_taxon_filepath).stem + "_backup_" + current_time_string + Path(index_to_taxon_filepath).suffix
        # parent process blocks until child finishes
        subprocess.run(["cp", index_filepath, backup_index_filepath], check=True)
        subprocess.run(["cp", index_to_taxon_filepath, backup_index_to_taxon_filepath], check=True)

    except subprocess.CalledProcessError:
        warnings.warn("Failed to create backup, aborting.")
        exit(1)


with open("config.yaml") as f:
    params = yaml.safe_load(f)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained = params['pretrained']
    model_name = params['model_name']
    index_filepath = params['index_filepath']
    index_to_taxon_filepath = params['index_to_taxon_filepath']
    prompt_templates = params['prompt_templates']

    model, preprocess = open_clip.create_model_from_pretrained(model_name=model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name=model_name)
    model.to(device)

    text_features_dim = model.text_projection.shape[0]
    df = pl.read_csv("species_with_taxonomic_info.csv")

    index = AnnoyIndex(text_features_dim, "angular")

    if os.path.exists(index_filepath):
        warnings.warn("An index with this name already exists. Are you sure you want to erase it? [Y/n/backup]")
        choice = input("Type Y, n, or backup: ")
        if choice == "Y":
            pass
        elif choice == "backup":
            backup_index()
        else:
            exit(1)

    index_to_species = {}
    # building on disk is slower but allows building indices that may not fit in RAM
    index.on_disk_build(index_filepath)

    items_added = 0
    for species_info in tqdm(df.rows(named=True)):
        prompts = build_prompts(species_info)
        embeddings = get_embeddings(prompts)

        for prompt, embedding in zip(prompts, embeddings):
            index.add_item(items_added, embedding)
            index_to_species[items_added] = {**species_info, **{"prompt": prompt}}
            items_added += 1

    index.build(params['num_trees'])

    with open(index_to_taxon_filepath, "wb") as f:
        pickle.dump(index_to_species, f)
