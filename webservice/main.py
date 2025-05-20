import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from io import BytesIO
import open_clip
from annoy import AnnoyIndex
import cloudpickle as pickle
import yaml

with open("config.yaml") as f:
    params = yaml.safe_load(f)

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = params['model_name']
pretrained = params['pretrained']
prompt_templates = params['prompt_templates']
index_filepath = params['index_filepath']

model, preprocess = open_clip.create_model_from_pretrained(model_name=model_name, pretrained=pretrained)
tokenizer = open_clip.get_tokenizer(model_name=model_name)
model.to(device)

embedding_dim = model.text_projection.shape[0]
vector_db = AnnoyIndex(embedding_dim, "angular")
vector_db.load(index_filepath)

with open("index_to_taxon_info.pickle", "rb") as f:
    index_to_taxon_info = pickle.load(f)

@app.post("/classify/")
async def classify(file: UploadFile = File(...), top_k: int = 10):
    # upload image and preprocess
    image_data = await file.read()
    image = Image.open(BytesIO(image_data)).convert("RGB")
    processed_image = preprocess(image).unsqueeze(0).to(device)

    # get image embedding and k * len(templates) (approximate) nearest neighbors
    with torch.no_grad():
        image_features = model.encode_image(processed_image).squeeze()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        nearest_neighbors, distances = vector_db.get_nns_by_vector(image_features, top_k * len(prompt_templates), include_distances=True)

    taxons = [{**index_to_taxon_info[i], **{"distance": distance}} for i, distance in zip(nearest_neighbors, distances)]

    # Boil down nearest neighbors from k * len(templates) -> k, taking min distance (highest similarity) for duplicated species
    aggregated_taxons = {}
    for taxon in sorted(taxons, key=lambda taxon: taxon['distance']):
        taxon_id = taxon['id']
        if taxon_id not in aggregated_taxons:
            aggregated_taxons[taxon_id] = taxon

    aggregated_taxons = sorted(aggregated_taxons.values(), key=lambda taxon: taxon['distance'])[:top_k]

    species_names = [taxon['vernacularName'] + " (" + taxon['scientificName'] + ")" for taxon in aggregated_taxons]
    prompts = [taxon['prompt'] for taxon in aggregated_taxons]

    # re-run closest k taxons through text embedding model to get similarities to image embedding
    with torch.no_grad():
        tokenized_text = tokenizer(prompts).to(device)
        text_features = model.encode_text(tokenized_text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        scores = (100.0 * text_features @ image_features.T).softmax(dim=-1)
        scores = scores.cpu().tolist()

    return zip(species_names, scores)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


