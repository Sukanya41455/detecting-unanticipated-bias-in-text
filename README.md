# Detecting Unanticipated Bias in Text

Unanticipated biases in LLM outputs are subtle and context-dependent, making them harder to detect using standard benchmarks. This project introduces a framework to identify such biases using:

- **LLMBI (Large Language Model Bias Index):** Combines sentiment analysis and diversity penalties to quantify bias in model-generated text.
  
- **Explainable AI (XAI):** Uses LIME and SHAP to interpret and explain bias predictions, highlighting key words contributing to biased outputs.

Together, these methods provide a transparent and interpretable approach to detect subtle and unexpected biases in language models.

## LLMBI Scoring



## Results

**Example 1:** He’s in a wheelchair, but he still goes out.

![](images/lime_example1.png?raw=true)

![](images/shap_example2.png?raw=true)


**Example 2:** He’s autistic, but you can’t even tell.

![](images/lime_example2.png?raw=true)

![](images/shap_example3.png?raw=true)