from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

from langchain.chains import LLMChain

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--language', default='python')
parser.add_argument('--task', default='return numbers from 1 to 10')
args = parser.parse_args()

code_prompt = PromptTemplate(
    template="Write a very short {language} funtion that will {task}",
    input_variables=['language', 'task']
)

llm = OpenAI()

code_chain = LLMChain(prompt=code_prompt, llm=llm)

result = code_chain({
    "language": args.language,
    "task": args.task
})

print(result['text'])
