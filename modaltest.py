import os 
import logging
from haystack import Document
from haystack import Pipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes.retriever.multimodal import MultiModalRetriever

class MultiModalSearch:
    # constructor function
    def __init__(self, model_name="sentence-transformers/clip-ViT-B-32", embedding_dim=512):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize the document store and retriever
        self.document_store = InMemoryDocumentStore(embedding_dim=embedding_dim)
        self.model_name = model_name
        
        # Load images from directory
        doc_dir = "new_data"
        self.logger.info(f"Loading images from {doc_dir}")
        
        try:
            # Fix the path issue in the original code
            images = [
                Document(
                    content=f"{doc_dir}/{filename}", 
                    content_type="image",
                    meta={"filename": filename}
                )
                for filename in os.listdir(doc_dir) if self._is_valid_image(filename)
            ]
            
            if not images:
                self.logger.warning(f"No images found in {doc_dir}")
            else:
                self.logger.info(f"Loaded {len(images)} images")
                
            # write all the images in a document store
            self.document_store.write_documents(images)
            
            # Initialize the retriever
            self.retriever_text_to_image = MultiModalRetriever(
                document_store=self.document_store,
                query_embedding_model=self.model_name, 
                query_type="text",
                document_embedding_models={"image": self.model_name},
            )
            
            # Generate embeddings for the images
            self.logger.info("Generating embeddings for images...")
            self.document_store.update_embeddings(retriever=self.retriever_text_to_image)
            
            # create pipeline
            self.pipeline = Pipeline()
            self.pipeline.add_node(component=self.retriever_text_to_image, name="retriever_text_to_image", inputs=["query"])
            
        except Exception as e:
            self.logger.error(f"Error initializing MultiModalSearch: {str(e)}")
            raise

    def _is_valid_image(self, filename):
        """Check if the file is a valid image based on extension"""
        valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        return any(filename.lower().endswith(ext) for ext in valid_extensions)

    # search function
    def search(self, query, top_k=3):
        """Search for images matching the text query"""
        self.logger.info(f"Searching for: '{query}' with top_k={top_k}")
        try:
            results = self.pipeline.run(
                query=query,
                params={"retriever_text_to_image": {"top_k": top_k}}
            )
            return sorted(results["documents"], key=lambda d: d.score, reverse=True)
        except Exception as e:
            self.logger.error(f"Error during search: {str(e)}")
            return []
    
    # Future implementation
    def search_by_image(self, image_path, top_k=3):
        """Search for similar images using an image query"""
        # This would require setting up an image-to-image retriever
        self.logger.info(f"Image search not yet implemented")
        return []