# Escrow 1024.17 Doc Chat Assistant

## Simplified chat assistant design diagram
![Design Diagram](media/design_diagram.png)


## How to interact with the Escrow 1024.17 Doc Chat Assistant

### Set the environment variables via .env file
Create a `.env` file in the root of the repo and set the following environment variables:

```shell
OPENAI_API_KEY=<your openai api key>
GROQ_API_KEY=<your groq api key> # required if you are running prompt evals, as the evals uses Mixtral-8b model from groq api to evaluate the prompts against openai's gpt-3.5-turbo model
```
the app will use the api keys to interact with the openai api and the groq api using this .env file, you do not need to export it manually.

### Run the chat assistant

Easiest way to interact with the chat assistant is via the strreamlit app. To run the app, without worrying about the dependencies, you can use the docker.  
To build and run our app in docker container, we use `docker compose`. 

From the root of the repo run the following command:

```shell
docker compose up --build
```

The comand above will bild the container and run the streamlit app. The app will be available at `http://localhost:8501` in your browser.

Please check the terminal for logs and errors if any. 


**NOTE**: It will take time to build the container as well as for the streamlit app to start. As the embedding model is large, it will take time to load the model, create index and start the app. (To save time and resources the vector database is not hosted on the cloud, it gets created on the fly when the app starts. This is not the best practice and should be avoided in production.)

#### You can run the app without Docker as well.
First install the dependencies
```shell
$ poetry install #first install the dependencies
$ poetry shell #activate the virtual environment
``` 

##### Chat Via Terminal

``` shell
$ python chat_assistant.py 
```
![running via terminal](media/terminal.png)

##### Chat via Streamlit App
``` shell
$ streamlit run app.py 
```
![running via streamlit](media/streamlit.png)

## Escrow 1024.17 data preparation
Please refer to the [escrow_data_retriever.ipynb notebook](escro_data_retriever.ipynb) for the data preparation steps. The notebook explains how the data was retrieved from the Escrow 1024.17 website and how the data was preprocessed to create the dataset for RAG indexingl.


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
$ ollama create escrow_gemma -f ./ModelfileGemma
```

to interact with the model run the following command:
```shell
$ ollama run escrow_gemma:latest 
```
 **NOTE**
 The model finetuning dataset consisted only the positive q/a pairs and no relevent context q/a, to get better performance we need to include negative q/a pairs as well along with some chat data. This will help the model to understand the context better and provide more accurate responses as intended for this application.

#### Fine-Tuned Model Evaluation
Please check out the [link](https://app.promptfoo.dev/eval/f:a0724f9a-6b9a-47c9-a988-42917d726ea9/) to see the evaluation of the fine-tuned model. The model was evaluated on these [test](prompt_eval_local/test.yaml) and compared to open source models: 'llama3-8b' and 'gemma-8b'. 

In the evaluation, the fine-tuned model is named 'escrow_gemma:latest'.

### Deploy fine-tuned model to AWS (for future referance)

```python
import json
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFaceModel, get_huggingface_llm_image_uri

try:
	role = sagemaker.get_execution_role()
except ValueError:
	iam = boto3.client('iam')
	role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

# Hub Model configuration. https://huggingface.co/models
hub = {
	'HF_MODEL_ID':'pyrotank41/gemma-7b-it-escrow-merged-gguf',
	'SM_NUM_GPUS': json.dumps(1)
}



# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
	image_uri=get_huggingface_llm_image_uri("huggingface",version="1.4.2"),
	env=hub,
	role=role, 
)

# deploy model to SageMaker Inference
predictor = huggingface_model.deploy(
	initial_instance_count=1,
	instance_type="ml.g5.2xlarge",
	container_startup_health_check_timeout=300,
  )
  
# send request
predictor.predict({
	"inputs": "What is the escrow 1024.17 document?",
})
```

