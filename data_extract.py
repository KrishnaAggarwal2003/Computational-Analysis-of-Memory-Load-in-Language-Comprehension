import os
import random
import pandas as pd
import conllu
from features_class import memory_load , fixed_effects
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import concurrent.futures

# Model and Filename dicts

languages = ['Catalan', 'Chinese', 'Croatian', 'Danish', 'Dutch', 'English', 'Finnish', 'French', 'German', 'Greek', 'Italian', 'Japanese', 'Korean', 'Lithuanian', 'Norwagein', 'Polish', 'Portuguese', 'Romanian', 'Russian', 'Slovenian', 'Spanish', 'Swedish', 'Ukranian']

models = ["ca_core_news_sm", "zh_core_web_sm", "hr_core_news_sm", "da_core_news_sm", "nl_core_news_sm", "en_core_web_sm", "fi_core_news_sm", "fr_core_news_sm", "de_core_news_sm", "el_core_news_sm", "it_core_news_sm", "ja_core_news_sm", "ko_core_news_sm", "lt_core_news_sm", "nb_core_news_sm", "pl_core_news_sm", "pt_core_news_sm", "ro_core_news_sm", "ru_core_news_sm", "sl_core_news_sm", "es_core_news_sm", "sv_core_news_sm", "uk_core_news_sm"]

model_dict = dict(zip(languages,models))

class DataExtract:
    def __init__(self,*languages, data_size=500):
        self.languages = languages
        self.data_size = data_size
        self.path_id = "C:\\Users\\DELL\\OneDrive\\Desktop\\Computation_linguistic"


    def file_extracts(self,language):
      folder_path = os.path.join(self.path_id, f"{language.capitalize()}_data")
      files = os.listdir(folder_path)
      return files
    
     
    def data_prep(self,language):
      files_list = self.file_extracts(language)
      sentences = []
      for filename in files_list:  
        if not filename:
          raise ValueError(f"No file found for language: {language}")

    # Validate file format
        if not filename.endswith('.conllu'):
          raise ValueError("Incorrect file format")
            
        file_path = os.path.join(self.path_id, f"{language.capitalize()}_data", filename)

        with open(file_path, 'r', encoding='utf-8') as f:
          data = f.read()

        parsed_data = conllu.parse(data)

        for item in parsed_data:
            sentences.append(item.metadata.get('text'))

      random.shuffle(sentences)
      return sentences
    
    def data_dictionary(self):
        sentence_dict = {language: self.data_prep(language)[:self.data_size] for language in self.languages}
        return sentence_dict
 


def process_language(language, sentence_dict):
    """
    This function processes a single language and returns its data dictionary.
    """
    print(f"Processing for language {language}")
    data = sentence_dict.get(language)  
    dl_length, inter_complexity, sentence_length, mem_load = [], [], [], []
    
    # Loop through each sentence and calculate the required features
    for i, my_string in enumerate(data):
        if i % 10 == 0:
            print(f'Processing sentence {i + 1}/{len(data)} for {language}')
        try:
            model = fixed_effects(sentence=my_string, language=language, **model_dict)
            memory_model = memory_load(sentence=my_string, language=language, **model_dict)
            dl_length.append(model.dep_length())
            inter_complexity.append(model.intervener_complexity())
            sentence_length.append(model.sentence_length())
            mem_load.append(memory_model.value())

            if i % 10 == 0:
                print(f'Finished sentence {i + 1} for {language}')
        except Exception as e:
            raise ValueError(f"Error processing sentence {i + 1} for {language}: {e}")
    
    # Create the nested dictionary for the current language
    language_index = languages.index(language)
    data_dict = {
        language_index: {
            'language': language,
            'sentences': data,
            'Dependency_length': dl_length,
            'Intervener_Complexity': inter_complexity,
            'Sentence_length': sentence_length,
            'Memory_load': mem_load
        }
    }
    return data_dict


# ThreadPoolExecutor version
def create_nested_dict_for_all_languages(sentence_dict):
    """
    This function processes all languages in parallel using multithreading.
    It returns a dictionary containing data for all languages.
    """
    print("Processing started for all languages...")

    # Use partial to pass sentence_dict as an additional argument to process_language
    process_language_with_dict = partial(process_language, sentence_dict=sentence_dict)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the process_language_with_dict function over all languages
        all_language_data = list(executor.map(process_language_with_dict, languages))
    
    # Combine all language data into a final nested dictionary
    final_data_dict = {}
    for data in all_language_data:
        final_data_dict.update(data)
     
    return final_data_dict


def preparing_dataframe(data_dict,csv_name):
    my_df_file = pd.DataFrame(data_dict).T
    expanded_df = my_df_file.explode([column for column in my_df_file.columns if column!='language']).reset_index(drop=True)
    final_df = expanded_df.sample(frac=1).reset_index(drop=True)
    return final_df.to_csv(csv_name)



if __name__ == "__main__":
  print("Starting data extraction protocol...")
  sentence_dict = DataExtract(*languages).data_dictionary()
  print("sentence_dict created\n Starting Threading for all languages...")
  final_data = create_nested_dict_for_all_languages(sentence_dict)
  print("Data processing complete for all languages.")
  preparing_dataframe(final_data, 'Project_data.csv')
  print("CSV file created\n End of data extraction protocol.")

