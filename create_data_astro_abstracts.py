import os
from tqdm import tqdm
import json 

def create_data_json_lm_flow(dataset):
    instances = []
    #type_of_split='train'
    for type_of_split in dataset:
      for row in tqdm(dataset[type_of_split]):
        instances.append({'text': row['text']})
  
      temp_dict = {}
      temp_dict['type'] = "text_only"
      temp_dict['instances'] = instances

      output_data_folder = f'data/astro_ph_abstracts/{type_of_split}'

      if not os.path.isdir(output_data_folder):
          os.makedirs(output_data_folder, exist_ok=True)

      output_file_name = 'astro_ph_abstracts.json'
      output_file_path = os.path.join(output_data_folder, output_file_name)
      
      with open(output_file_path, 'w') as out:     
          json.dump(temp_dict, out)


if __name__ == '__main__':
    from datasets import load_dataset
    from huggingface_hub import login
    dataset = load_dataset("universeTBD/arxiv-astro-abstracts-all")
    create_data_json_lm_flow(dataset)