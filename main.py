from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

from langchain.chains import LLMChain, SequentialChain

import argparse
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--language', default='python')
parser.add_argument('--task', default='return numbers from 1 to 10')
args = parser.parse_args()

code_prompt = PromptTemplate(
    input_variables=['language', 'task'],
    template="Write a very short {language} funtion that will {task}"
)

test_prompt = PromptTemplate(
    input_variables=['language', 'code'],
    template="Write a test for the following {language} code:\n{code}"
)

llm = OpenAI()

code_chain = LLMChain(prompt=code_prompt, llm=llm, output_key="code")
test_chain = LLMChain(prompt=test_prompt, llm=llm, output_key='test')

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=['language', 'task'],
    output_variables=['code', 'test']
)

result = chain({
    "language": args.language,
    "task": args.task
})

print('>>>>>>> GENERATED CODE:')
print(result['code'])

print('>>>>>>> GENERATED TEST:')
print(result['test'])
