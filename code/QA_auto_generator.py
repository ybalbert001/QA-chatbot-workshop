import os
import re
import argparse
import openai
import json
from tqdm import tqdm
from langchain.document_loaders import PyPDFLoader

# you need to install these packages: pypdf, tqdm, openai, langchain 

# Please excute => export OPENAI_API_KEY={key}
openai.api_key = os.getenv("OPENAI_API_KEY")


prompt_template = """
Here is one page of {product}'s manual document
```
{page}
```
Please automatically generate as many questions as possible based on this manual document, and follow these rules:
1. "{product}"" should be contained in every question
2. questions start with "Question:"
3. answers begin with "Answer:"
"""

def Generate_QA(prompt):
    messages = [{"role": "user", "content": f"{prompt}"}]
    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=messages,
      temperature=0,
      max_tokens=2048
    )
    
    content = response.choices[0]["message"]["content"]
    arr = content.split('Question:')[1:]
    qa_pair = [ p.split('Answer:') for p in arr ]
    return qa_pair


def Generate_QA_From_Pages(pages, product_name="Midea Dishwasher"):
    for page in tqdm(pages[:20]):
        prompt = prompt_template.format(product=product_name, page=page.page_content)
        qa_list = Generate_QA(prompt)
        for q,a in qa_list:
            ret = page.metadata
            ret["Q"] = q.strip()
            ret["A"] = a.strip()
            yield ret
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='./1-Manual.pdf', help='input file')
    parser.add_argument('--output_file', type=str, default='./FAQ.txt', help='output file')
    parser.add_argument('--product', type=str, default="Midea Dishwasher", help='specify the product name of pdf')
    args = parser.parse_args()
    pdf_path = args.input_file
    product_name = args.product
    qa_path = args.output_file
    
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    out_f = open(qa_path, 'w')
    
    with open(qa_path, 'w') as out_f:
        for result in Generate_QA_From_Pages(pages, product_name):
            out_f.write(json.dumps(result))
            out_f.write("\n")
            
    