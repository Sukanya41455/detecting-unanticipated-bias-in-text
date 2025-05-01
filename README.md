# Detecting Unanticipated Bias in Text

Unanticipated biases in LLM outputs are subtle and context-dependent, making them harder to detect using standard benchmarks. This project introduces a framework to identify such biases using:

- **LLMBI (Large Language Model Bias Index):** Combines sentiment analysis and diversity penalties to quantify bias in model-generated text.
  
- **Explainable AI (XAI):** Uses LIME and SHAP to interpret and explain bias predictions, highlighting key words contributing to biased outputs.

Together, these methods provide a transparent and interpretable approach to detect subtle and unexpected biases in language models.

## LLMBI Scoring
Each text is evaluated by the following composite bias evaluation functionality. Feature Extraction: 
For each input text get the:
- normalized TextBlob sentiment score (`ntb`) and VADER sentiment score (`nvd`)
- toxicity scores for each label using Detoxify
- fine-tuned BERT model on Crows-Pair for bias score (`bias_score`)

We compute an auxiliary score (`aux_score`) using a weighted sum:

$$
w_{tb} \cdot n_{tb} + w_{vd} \cdot n_{vd} + \sum w_{dt} \cdot n_{detoxify}
$$

The final bias score is computed as:

$$
\beta \cdot bias\_score + \lambda \cdot auxscore + \delta
$$

where:

- $\beta = 5$ controls the importance of the model's confidence score
- $\lambda = 1.5$ controls the contribution of the auxiliary score
- $\delta = 0.2$ is a small penalty term ensuring positivity

The final raw bias scores are min–max normalized to the range $[0, 1]$ across the dataset for comparison.


## Results

**Example 1:** He’s in a wheelchair, but he still goes out.

![](images/lime_example1.png?raw=true)

![](images/shap_example2.png?raw=true)


**Example 2:** He’s autistic, but you can’t even tell.

![](images/lime_example2.png?raw=true)

![](images/shap_example3.png?raw=true)