from setuptools import setup

setup(
    name='langchain-example',
    version='1.0',
    description='A test langchain project. This project is based '
                'on https://medium.com/@onkarmishra/using-langchain-for-question-answering-on-own-data-3af0a82789ed.'
                'All credits to the original creator of the article.',

    author='kperun',
    packages=['langchain-example'],
    install_requires=['pypdf', 'langchain', 'langchain-community', 'yt-dlp', 'ffmpeg', 'langchain-openai', 'chromadb',
                      'python-dotenv','lark']
)
