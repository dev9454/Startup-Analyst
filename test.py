from tools.loaders import load_pdf_docai
docs = load_pdf_docai("data/Ctruth.pdf")
print(len(docs), "pages")
print(docs[0].page_content[:500])