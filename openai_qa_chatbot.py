# Import the required libraries
import os
import pandas as pd
import numpy as np
from docx import Document
from openai.embeddings_utils import distances_from_embeddings
import tiktoken
import openai

# Function to read text from a DOCX file and convert it into a DataFrame
def read_docx_to_dataframe(file_path):
    doc = Document(file_path)
    data = []

    # Extract text from each paragraph in the document
    for paragraph in doc.paragraphs:
        data.append(paragraph.text)

    # Create pandas DataFrame
    df = pd.DataFrame(data, columns=["Text"])

    return df

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens = max_tokens):

    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater 
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append(". ".join(chunk) + ".")
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of 
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks
    

# Main function to answer a question based on a DataFrame of texts
def answer_question(df, model="text-davinci-003", question="", max_len=1800, size="ada", debug=False, max_tokens=150, stop_sequence=None):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    # Create a context for the question
    context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )

    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")

    try:
        # Create completions using the question and context
        response = openai.Completion.create(
            prompt=f"Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

# Specify the path to your DOCX file
docx_file_path = "sample.docx"

# Read DOCX file and convert it to a DataFrame
df = read_docx_to_dataframe(docx_file_path)

# Remove rows with empty or None values
df = df.dropna()

# Filter out rows with empty text
df = df[df['Text'].map(lambda d: len(d)) > 0]

# Tokenize the text and save the number of tokens to a new column
tokenizer = tiktoken.get_encoding("cl100k_base")
df['n_tokens'] = df.Text.apply(lambda x: len(tokenizer.encode(x)))

# Visualize the distribution of the number of tokens per row using a histogram
df.n_tokens.hist()

# Define the maximum number of tokens allowed per text
max_tokens = 500

# Shorten long texts by splitting them into chunks with a maximum number of tokens
shortened = []
for row in df.iterrows():
    if row[1]['Text'] is None:
        continue

    if row[1]['n_tokens'] > max_tokens:
        shortened += split_into_many(row[1]['Text'], max_tokens)
    else:
        shortened.append(row[1]['Text'])

# Create a new DataFrame with the shortened texts and their token counts
df = pd.DataFrame(shortened, columns=['Text'])
df['n_tokens'] = df.Text.apply(lambda x: len(tokenizer.encode(x)))
df.n_tokens.hist()

# Set your OPENAI API key and organization
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")

# Fetch embeddings for the texts using the ada-002 model
df['embeddings'] = df.Text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])

# Answer a specific question based on the processed texts
answer = answer_question(df, question="What is core leadership skill?")
print(answer)
