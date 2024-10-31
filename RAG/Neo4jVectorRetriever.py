import openai
import numpy as np
from typing import Any
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever

class Neo4jVectorRetriever(BaseRetriever):
    driver: Any
    top_k: int = 5

    def get_query_embedding(self, query):
        response = openai.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        )
        query_embedding = response.data[0].embedding
        return query_embedding

    def get_top_node(self, query_embedding):
        with self.driver.session() as session:
            results = session.read_transaction(
                self._similarity_search, query_embedding, top_k=1)
        if results:
            record = results[0]
            node = record['node']
            content = node.get('content', '')
            node_id = node.get('id')
            return node_id, content
        else:
            return None, None

    def get_neighbors(self, node_id, top_k):
        with self.driver.session() as session:
            results = session.read_transaction(
                self._get_neighbors_tx, node_id, top_k
            )
        return results

    def _get_relevant_documents(self, query):
        query_embedding = self.get_query_embedding(query)
        with self.driver.session() as session:
            results = session.read_transaction(
                self._similarity_search, query_embedding, self.top_k)
        documents = []
        for record in results:
            node = record['node']
            content = node.get('content', '')
            metadata = {
                'id': node.get('id'),
                'similarity': record.get('score')
            }
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        return documents

    @staticmethod
    def _similarity_search(tx, query_embedding, top_k=10):
        result = tx.run("""
        WITH $embedding AS query_embedding
        CALL db.index.vector.queryNodes(
            'nodeContent',
            $top_k,
            query_embedding
        ) YIELD node, score
        RETURN node, score
        ORDER BY score DESC
        """, embedding=query_embedding, top_k=top_k)
        
        nodes = [record.data() for record in result]
        return nodes

    @staticmethod
    def _get_neighbors_tx(tx, node_id, top_k):
        result = tx.run("""
        MATCH (n)-[]->(neighbor)
        WHERE n.id = $node_id
        RETURN neighbor.id AS id, neighbor.content AS content, neighbor.plotEmbedding AS embedding
        LIMIT $top_k
        """, node_id=node_id, top_k=top_k)
        neighbors = [record.data() for record in result]
        return neighbors
