# Computational-Analysis-of-Memory-Load-in-Language-Comprehension

## Overview
This project investigates the **memory load** associated with processing sentences in multiple languages using dependency parsing and statistical modelling. It leverages Universal Dependencies treebanks, spaCy language models, and custom feature extraction to quantify syntactic complexity and memory demands across languages.

## Project Structure

- **data_extract.py**: Handles data extraction, feature computation, and CSV generation.
- **features_class.py**: Implements feature extraction classes for memory load and fixed effects.
- **main.ipynb**: Performs statistical analysis and visualisation of the results.
- **file.ipynb**: Demonstrates dependency graph construction and adjacency matrix generation for a sample sentence.
- **Project_data.csv**: Output dataset with computed features for all sentences and languages.
- **deep-ud-2.8-data.tgz**: Source treebank archive.
- **[Language]_data/**: Folders containing `.conllu` files for each language.

## Install Dependencies

Before running the code, make sure to install the required Python packages. You can do this using pip:

```bash
pip install pandas matplotlib seaborn statsmodels scikit-learn spacy networkx janome konlpy
```

Run the following PowerShell command to download the appropriate spaCy models. The models are listed in the [`models.txt`](https://github.com/KrishnaAggarwal2003/Computational-Analysis-of-Memory-Load-in-Language-Comprehension/blob/main/models.txt)

```Powershell
Get-Content models.txt | ForEach-Object { python -m spacy download ($_.Trim()) }
```

**Note:**  
- For `konlpy` (Korean NLP), you may need Java installed and additional setup depending on your OS.
- For `janome` (Japanese), no extra setup is needed beyond pip.

## Data Processing Pipeline

1. **Extraction**:  
   - The code in script [`file.ipynb`](https://github.com/KrishnaAggarwal2003/Computational-Analysis-of-Memory-Load-in-Language-Comprehension/blob/main/file.ipynb) helps extract the `.conllu` files from the Source treebank archive
   - The script [`data_extract.py`](https://github.com/KrishnaAggarwal2003/Computational-Analysis-of-Memory-Load-in-Language-Comprehension/blob/main/data_extract.py) extracts sentences from `.conllu` files for each language.

3. **Feature Computation**:  
   For each sentence, the following features are computed using [`features_class.memory_load`](https://github.com/KrishnaAggarwal2003/Computational-Analysis-of-Memory-Load-in-Language-Comprehension/blob/main/features_class.py) and [`features_class.fixed_effects`](https://github.com/KrishnaAggarwal2003/Computational-Analysis-of-Memory-Load-in-Language-Comprehension/blob/main/features_class.py):
   - **Dependency Length** refers to the number of words between a head and its dependent in a sentence’s syntactic structure. It is used as a proxy for working memory load in language processing. The **Dependency Length Minimisation (DLM)** hypothesis suggests that natural languages prefer shorter dependencies to reduce cognitive effort.

   - **Intervener Complexity** is a metric that counts the number of intervening heads between a dependent and its head in a dependency tree. It is proposed as a more precise approximation of memory load than dependency length, capturing structural rather than linear distance.

   - **Sentence Length**: Number of words/tokens.
   - **Memory Load**: Sum of feature interference and misbinding, reflecting syntactic working memory demands.

4. **Parallelization**:  
   Feature extraction is parallelised across languages for efficiency.

5. **Data Aggregation**:  
   All features are compiled into a nested dictionary and saved as [`Project_data.csv`](https://github.com/KrishnaAggarwal2003/Computational-Analysis-of-Memory-Load-in-Language-Comprehension/blob/main/Project_data.csv).

## Statistical Analysis

Analysis is performed in [`main.ipynb`](https://github.com/KrishnaAggarwal2003/Computational-Analysis-of-Memory-Load-in-Language-Comprehension/blob/main/main.ipynb):

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
  The following result summary table provides coefficients, confidence intervals, and significance for each predictor, indicating their contribution to memory load.

  ![Image](https://github.com/user-attachments/assets/1f6d8e93-e7e1-4c6e-9ff9-e01223c4a236)


## Plots and Visualisations

1. ### Fixed Effects Coefficients Plot

   ![Image](https://github.com/user-attachments/assets/9de9e167-4eab-4f7b-a6c5-09d4de1e031a)

This plot displays the fixed effects coefficients of three predictors — **Dependency Length, Intervener Complexity**, and **Sentence Length** — on estimated **Memory Load**, with 95% confidence intervals.

- Sentence Length has the largest effect, suggesting it is a strong predictor of memory demand.

- Dependency Length and Intervener Complexity also contribute positively and significantly, indicating their roles in shaping memory load during sentence processing.

- The confidence intervals show all effects are statistically robust (non-zero).


2. ### Regression Plots

   ![Image](https://github.com/user-attachments/assets/b3bb0e7c-7ebc-4dc7-9f12-81bffc2bbd6d)

   ![Image](https://github.com/user-attachments/assets/e419417a-ec71-484c-8a87-7c2da4dd05a5)

   ![Image](https://github.com/user-attachments/assets/9f1ea498-9636-4d5a-a0b1-454835159310)


For each predictor, a regression plot shows its relationship with memory load.

3. ### Violin Plot

![Image](https://github.com/user-attachments/assets/7386a577-4d9c-4a31-a811-72401348d1d5)

This violin plot illustrates the distribution of estimated memory load across multiple languages. Each violin represents a language's density distribution, central tendency (white dot), and interquartile range (thick bar).

- While memory load varies within each language, the overall spread and central values differ across languages.

- English, Greek, and Portuguese languages show wider ranges and heavier tails, suggesting more syntactic variability.

- Japanese and Korean languages show more compact distributions, indicating lower and more consistent memory demands.

4. ### Actual vs Predicted Bar Plot

![image](https://github.com/user-attachments/assets/1b4bc5dd-dee3-4b99-b597-d6a47445063c)

 - Compares the mean actual and predicted memory load for each language.

**Model Metrics**
- **R² Score**: Measures how well the predicted values approximate the actual values. A score close to 1.0 (100%) indicates that the model captures most of the variance in the target variable, showing strong predictive alignment.
- **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)**: Quantify the average prediction error. MSE penalises larger errors more heavily due to squaring, while MAE gives a direct average of the absolute differences. Both are used to assess how closely the model's predictions match the true target values.
```
R² Score: 80.91%
Mean Squared Error: 8.0970
Mean Absolute Error: 2.0867

```

## Example: Dependency Graph Visualisation

[`file.ipynb`](c:/Users/DELL/OneDrive/Desktop/Completed_IITK_Project/file.ipynb) demonstrates how a sentence is parsed, visualised as a dependency graph, and converted to an adjacency matrix using NetworkX and spaCy.

![image](https://github.com/user-attachments/assets/ff55af4b-79f9-4602-826c-83854ad85cdf)


## Key findings / Conclusions
- **Dependency length, Intervener complexity**, and **Sentence length** all significantly contribute to memory load.

- There is substantial variation in memory load across languages, as visualised in the violin and bar plots.

- **Intervener Complexity** exerts a greater impact on **Memory load** than **Dependency Length**, suggesting that the structural complexity and type of intervening elements play a more critical role in cognitive processing during sentence comprehension. This indicates that memory load is not merely a function of linear distance (i.e., dependency length), but is more significantly influenced by interveners' syntactic and semantic properties between dependent elements.



  




