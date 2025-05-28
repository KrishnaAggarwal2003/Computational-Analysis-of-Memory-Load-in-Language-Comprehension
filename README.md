# Computational-Analysis-of-Memory-Load-in-Language-Comprehension

## Overview
This project investigates the **memory load** associated with processing sentences in multiple languages using dependency parsing and statistical modelling. It leverages Universal Dependencies treebanks, spaCy language models, and custom feature extraction to quantify syntactic complexity and memory demands across languages.

## Project Structure

- **data_extract.py**: Handles data extraction, feature computation, and CSV generation.
- **features_class.py**: Implements feature extraction classes for memory load and fixed effects.
- **main.ipynb**: Performs statistical analysis and visualization of the results.
- **file.ipynb**: Demonstrates dependency graph construction and adjacency matrix generation for a sample sentence.
- **Project_data.csv**: Output dataset with computed features for all sentences and languages.
- **deep-ud-2.8-data.tgz**: Source treebank archive.
- **[Language]_data/**: Folders containing `.conllu` files for each language.

## Data Processing Pipeline

1. **Extraction**:  
   - The code in script [`file.ipynb`](##) helps extract the `.conllu` files from the Source treebank archive
   - The script [`data_extract.py`](##) extracts sentences from `.conllu` files for each language.

3. **Feature Computation**:  
   For each sentence, the following features are computed using [`features_class.memory_load`](##) and [`features_class.fixed_effects`](##):
   - **Dependency Length** refers to the number of words between a head and its dependent in a sentence’s syntactic structure. It is used as a proxy for working memory load in language processing. The **Dependency Length Minimization (DLM)** hypothesis suggests that natural languages prefer shorter dependencies to reduce cognitive effort.

   - **Intervener Complexity** is a metric that counts the number of intervening heads between a dependent and its head in a dependency tree. It is proposed as a more precise approximation of memory load than dependency length, capturing structural rather than linear distance.

   - **Sentence Length**: Number of words/tokens.
   - **Memory Load**: Sum of feature interference and misbinding, reflecting syntactic working memory demands.

4. **Parallelization**:  
   Feature extraction is parallelized across languages for efficiency.

5. **Data Aggregation**:  
   All features are compiled into a nested dictionary and saved as **Project_data.csv**

## Statistical Analysis

Analysis is performed in [`main.ipynb`](##):

### Linear Mixed Model (LMM)

- **Model**:  
  The LMM predicts `Memory_load` using `Dependency_length`, `Intervener_Complexity`, and `Sentence_length` as fixed effects, with language as a random effect.
  ```python
  equation = "Memory_load ~ Dependency_length + Intervener_Complexity + Sentence_length"
  model = smf.mixedlm(equation, final_dataframe, groups=final_dataframe["language"])
  result = model.fit()
  print(result.summary())
  ```
- **Interpretation**:  
  The summary table provides coefficients, confidence intervals, and significance for each predictor, indicating their contribution to memory load.

  ![Image](https://github.com/user-attachments/assets/1f6d8e93-e7e1-4c6e-9ff9-e01223c4a236)

  




