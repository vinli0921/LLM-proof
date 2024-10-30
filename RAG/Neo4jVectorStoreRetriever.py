import openai
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

class Neo4jVectorStoreRetriever(VectorStoreRetriever):
    def __init__(self, driver, top_k=5):
        self.driver = driver
        self.top_k = top_k

    def get_relevant_documents(self, query):
        # generate embedding for the query
        # need to change this
        response = openai.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        )
        query_embedding = response.data[0].embedding

        # Perform similarity search in Neo4j
        with self.driver.session() as session:
            results = session.read_transaction(
                self._similarity_search, query_embedding, self.top_k)
        
        # Convert results to LangChain Documents
        documents = []
        for record in results:
            content = record['content']
            metadata = {
                'id': record['id'],
                'similarity': record['similarity']
            }
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        
        return documents

    @staticmethod
    def _similarity_search(tx, query_embedding, k):
        result = tx.run("""
        MATCH (n:Node)
        WHERE n.embedding IS NOT NULL
        WITH n, gds.similarity.cosine(n.embedding, $query_embedding) AS similarity
        RETURN n.id AS id, n.content AS content, similarity
        ORDER BY similarity DESC
        LIMIT $k
        """, query_embedding=query_embedding, k=k)
        return result.data()