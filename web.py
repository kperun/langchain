from langchain.document_loaders import WebBaseLoader

# Use a markdown file from github page
loader = WebBaseLoader("https://github.com/basecamp/handbook/blob/master/37signals-is-you.md")

docs = loader.load()
print(docs[0].page_content[:500])