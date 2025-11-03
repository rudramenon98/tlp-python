# add the current directory to the path since enginius_dockers is not a package
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import predict_documents, summarize_documents

# test summarize_documents with doc_ids = [100]


def test_predict_documents():
    doc_id = 100
    document = predict_documents(
        doc_ids=[doc_id], pdf_type="pdf", repo_id=3, model=None
    )


def test_summarize_documents():
    doc_ids = [100]
    summarize_documents(doc_ids)


if __name__ == "__main__":
    test_predict_documents()
    # test_summarize_documents()
