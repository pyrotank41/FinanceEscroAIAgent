# Escrow 1024.17 Doc Chat Assistant

## Simplified chat assistant design diagram
![Design Diagram](media/design_diagram.png)


## How to interact with the Escrow 1024.17 Doc Chat Assistant

First install the dependencies
```shell
$ poetry install #first install the dependencies
$ poetry shell #activate the virtual environment
``` 

### Via Terminal

``` shell
$ python chat_assistant.py 
```
![running via terminal](media/terminal.png)

### Via Streamlit App
``` shell
$ streamlit run app.py 
```
![running via streamlit](media/streamlit.png)


## RAG evaluation data
Final rag eval results: [RAG evaluation results](rag_eval_results.csv)  
I ran a experiments on Direct RAG along with advanced retrieval methods like Sentence Window and Auto-Merging Retrival. Furthermore, experiments included prameter variance as well to find the best configuration for Retrueving Escrow 1024.17 documents. The evaluations was run on [these queries](application/valid_eval_queries.txt). The results are as follows:

![RAG Evaluation](media/rag_eval_table.png)

The best configuration (good balance of answer and context relevance, and groundedness) was found to be:
- Sentence retrieval window: 1 
- chunk size: 128
- effective retrieved context length (node): 384 characters

Notes: although not the cheapest configuration, it was the most effective in terms of Groundedness, answer relevance and context relevance.


## Prompt Template
The prompt template is created as follows (learnt from the [papers](#prompt-engineering-references) below):

\# **Role**  
(Role-play prompting is an effective stratigy where we assign a specific role to play during the interaction. This helps the model to "immerse" itself in the role and provide more accurate and relevant answers) [ref 1](#prompt-engineering-references)

\# **Task**  
(a direct description of what we want the model to do. One technique that works well is to use chain-of-thought prompting [ref 2](#prompt-engineering-references) to guide the model through the task)

\# **Specifics**
(provide most inportant notes regarding the task. Integrating Emotional Stimuli [ref 3](#prompt-engineering-references) has showin to increase response quality and accuracy)

\# **Context**   
(what is the environment in which the task is to be performed. Fairness-guided Few-shot Prompting [ref 4](#prompt-engineering-references) has shown that providing context helps the model to understand the task better)

\# **Examples**  
(giving a few q/a pairs of example questions and answers can help the model understand the task better. This is a good practice to follow. Rethinking the Role of Demonstrations [ref 6](#prompt-engineering-references) explains this in detail)

\# **Notes**  
(Additional and repeted notes that can help the model do the task better . lost in the middle paper [ref 7](#prompt-engineering-references) shows that the llms are good at remembering the start and the end of the context better that the middle. so it is important to repeat the task and the context in the notes section briefly. Though newer models are better at finding needle in a hay stack, it is still a good practice to follow)
### Prompt Engineering References

1. [Better Zero-Shot Reasoning with Role-Play Prompting](https://arxiv.org/abs/2308.07702)
2. [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
3. [Large Language Models Understand and Can be Enhanced by Emotional Stimuli](https://arxiv.org/abs/2307.11760)
4. [Fairness-guided Few-shot Prompting for Large Language Models](https://arxiv.org/abs/2303.13217)
5. [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)
6. [Rethinking the Role of Demonstrations: What Makes In-Context Learning Work?](https://arxiv.org/abs/2202.12837)
7. [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)

## Prompt Evaluation
We are using [promptfoo](https://www.promptfoo.dev) for evaluating our prompts. To run the evaluation, first install promptfoo

```shell
$ bun add promptfoo # or npm install -g promptfoo
```

Then form the eval command from the root folder of this repo as follows:
```shell
$ cd prompt_eval_cloud
$ promptfoo eval
```
make sure the GROQ_API_KEY and OPENAI_API_KEY is set in the .env file as this eval evaluated models from the OpenAI API (gpt-3.5-turbo) and GROQ API(mixtral-8b).  

***Note**: To save time and resources, the evaluation is not thurogh and only a few prompts are evaluated.*
![prompt foo evaluation](media/promptfoo_eval_terminal.png)

To get the detailed view of the evalutaiton, run the following command: 
```shell
$ promptfoo view -y
```
A new tab with the following view will open in your browser:
![prompt foo evaluation](media/promptfoo_dashboard.png)

## fine-tuneing
Refer the following [notebook](generate_dataset_finetune.ipynb) on how the dataset was generated and the model.

Refer the following [notebook](https://colab.research.google.com/drive/1Mf1qeQl8EXwbUQz8eE7dHG0xC1RLNYLi?usp=sharing) for see how the model was finetuned on the Escrow 1024.17 documents.  
(*note: this notebook is a colab motebook and it was easy to run the experiments on the google cloud with powerful gpus.*)

The finetuned gemma model is available at [huggingface](https://huggingface.co/pyrotank41/gemma-7b-it-escrow-merged-gguf/tree/main)

Download the model and place it in the `fine_tuned_model` folder in the repo and from the root of the repo run the following command to interact create ollama model.

```shell
$ ollama create escro_gemma -f ./ModelfileGemma
```

to interact with the model run the following command:
```shell
$ ollama run escro_gemma:latest 
```
