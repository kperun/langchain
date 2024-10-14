import os
from dotenv import load_dotenv
from pinecone import Pinecone,ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever
import nltk
nltk.download('punkt_tab')

# get the api key
load_dotenv()
api_key = os.getenv('PINECONE_API_KEY')
os.environ["HF_TOKEN"] = os.getenv('HUGGING_FACE_API_KEY')

#create the index
print("Create pinecone api...")
index_name = "hybrid-search-langchain"
client = Pinecone(api_key=api_key)
if index_name not in client.list_indexes().names():
    print("Create index...")
    client.create_index(
        name=index_name,
        dimension=384, # has to be according to the embedding, here hugging face embedding is use
        # with a vector length of 384
        metric='dotproduct', # how is distance measured
        spec=ServerlessSpec(cloud='aws',region='us-east-1') # where is the underlying infrastructure placed
    )
else:
    print("Index already exists...")
print("Retrieve index...")
index = client.Index(name=index_name)


#get a hugging face embedding
print("Create embeddings...")
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
#all-MiniLM-L6-v2 is a sentence transformer lm

# create an encoder to get a query into sparse matrix
print("Create sparse encoder...")
#bm24_encoder = BM25Encoder().default()
#sentences = ["You may say I'm a dreamer, but I'm not the only one.",
#             "I hope someday you'll join us.",
#             "And the world will live as one."]

# make a fit, i.e., create a matrix which decode those sentences
#bm24_encoder.fit(sentences)
# you can store it in a file
#bm24_encoder.dump("assets/dreamer.json")
# load it again
bm24_encoder = BM25Encoder().load("assets/dreamer.json")
# create the retriever which uses the embeddings from hugging face to get the vector representation for the
# semantic search, the sparse encoder for the syntactic search and the index which combines both
print("Create retriever...")
retriever = PineconeHybridSearchRetriever(embeddings=embeddings,sparse_encoder=bm24_encoder,index=index)

# add the text inside the index, this means we are adding our documents in which we want to search
#print("Add sentences...")
#retriever.add_texts(sentences)

# invoke the retriever to retrieve information based on the content in the index, using both types of search
# we see that one document got the highest score (with the respective content), this we can now use for RAG
print(retriever.invoke('What do people say about me?'))

