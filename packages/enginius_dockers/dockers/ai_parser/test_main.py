from enginius_dockers.dockers.ai_parser.main import summarize_documents

# test summarize_documents with doc_ids = [100]


def test_summarize_documents():
    doc_ids = [100]
    summarize_documents(doc_ids)


if __name__ == "__main__":
    test_summarize_documents()
