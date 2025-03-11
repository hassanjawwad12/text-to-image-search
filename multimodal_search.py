import os 
from haystack import Document
from haystack import Pipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes.retriever.multimodal import MultiModalRetriever

class MultiModalSearch:
    # constructor function
    def __init__(self):
        # Initialize the document store and retriever
        self.document_store = InMemoryDocumentStore(embedding_dim=512) # dimension is based on the modal that u use
        doc_dir = "new_data"
        images = [
            Document(content= f"{doc_dir}/{filename}", content_type="image")
            for filename in os.listdir(f"./doc_dir")
        ]

        # write all the images in a document store
        self.document_store.write_documents(images)

        # 
        self.retriever_text_to_image = MultiModalRetriever(
            document_store=self.document_store,
            query_embedding_model="sentence-transformers/clip-ViT-B-32", 
            query_type="text",
            document_embedding_models={"image": "sentence-transformers/clip-ViT-B-32"},
        )
        
        # use the retreiver 
        self.document_store.update_embeddings(retriever=self.retriever_text_to_image)

        # create an empty pipeline
        self.pipeline = Pipeline()

        self.pipeline.add_node(component=self.retriever_text_to_image, name="retriever_text_to_image", inputs=["query"])

    # search function
    def search(self, query, top_k= 3):
        results= self.pipeline.run(query=query,params={"retriever_text_to_image": {"top_k": top_k}})
        return sorted(results["documents"], key=lambda d:d.score, reverse=True)
