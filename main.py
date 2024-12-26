from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import character
import re


embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5", model_kwargs={'device': "cpu"})


def load_pdf(filename):
    loader = PyPDFLoader(f"data/{filename}.pdf")
    text = loader.load()
    return text


def preprocess_text(text):
    text_lower = text.lower()
    text_no_punctuation = re.sub(r'[^\w\s$%.,\"\'!?()]', '', text_lower)
    text_normalized_tabs = re.sub(r'(\t)+', '', text_no_punctuation)
    return text_normalized_tabs


def chunky_text(documents):
    text_splitter = character.CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="\n")
    text_in_chunks = text_splitter.split_documents(documents)
    return text_in_chunks


def main():
    filename = input('Введите название файла (он должен лежать в папке data/, расширение должно быть .pdf): ')
    text = load_pdf(filename)

    for x in range(len(text)):
        text[x].page_content = preprocess_text(text[x].page_content)

    text_in_chunks = chunky_text(text)

    while True:
        query = input('Введите запрос: ')

        qdrant = Qdrant.from_documents(
            text_in_chunks,
            embeddings,
            location=":memory:",
            collection_name="msft_data",
            force_recreate=True
        )
        found_docs = qdrant.similarity_search_with_score(query, k=1)

        response = "\n\n".join(doc[0].page_content for doc in found_docs)

        print(response)

        new_query = input('Закончить (да/нет)? ')
        if new_query == 'да':
            break


if __name__ == "__main__":
    main()
