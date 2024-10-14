from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv


def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i + 1}:\n\n" + d.page_content for i, d in enumerate(docs)]))


load_dotenv()
## Read in
print('Read pdf')
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("assets/hamlet.pdf"),
    PyPDFLoader("assets/romeo.pdf")
]

docs = []
for loader in loaders:
    single_doc = loader.load()
    docs.extend(single_doc)

chunk_size = 26
chunk_overlap = 4

print('Split...')
## Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)

splits = text_splitter.split_documents(docs)

## Create Embeddings
print('Embedd...')
embedding = OpenAIEmbeddings(api_key=os.getenv('OPENAI_API_KEY'))

persist_directory = 'assets/chroma/'

print('Create vector store...')
# Create the vector store
vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

## Search similar content from the documents based on query
print('Search...')
question = "what was FRANCISCO's part in act 1 scene 1"
docs_sim = vectordb.similarity_search(question, k=3)
docs_mmr = vectordb.max_marginal_relevance_search(question, k=3)
# filter={"source":"<path>"} can be used to look up in a specific doc

# Check the content of the first document
print('Found...')
print('SIM:')
print(docs_sim)
print('MMR:')
print(docs_mmr)

# Add metadata information about the pdfs, i.e., we can tell the llm how to interpret the metadata
metadata_field_info = [
    AttributeInfo(
        name="source",
        description="The 'source' play the chunk is from, should be one of 'assets/hamlet.pdf' or 'assets/romeo.pdf'",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="The 'page' from the play",
        type="integer",
    ),
]

# Define what those documents are in general about
document_content_description = "A play by Shakespear"
llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), temperature=0)
compressor = LLMChainExtractor.from_llm(llm)  # used for contextual compression
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever()
)

# In case no contextual compression is required, use
# retriever = SelfQueryRetriever.from_llm(
#     llm,
#     vectordb,
#     document_content_description,
#     metadata_field_info,
#     verbose=True
# )

print("Search relevant documents using metadata...")
docs = retriever.get_relevant_documents(question)

# Retrieve the concrete answer to the question without prompt template
print("Retrieve without prompt template...")
llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever
)

result = qa_chain({"query": question})
print(result["result"])

# Retrieve the concrete answer to the question with prompt template
print("Retrieve with prompt template...")
template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
result = qa_chain({"query": question})
# Check the result of the query
print(result["result"])
# Check the source document from where we
print(result["source_documents"][0])