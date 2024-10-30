import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
import pandas as pd
import json

load_dotenv()

# Neo4j config and constraints
neo4j_uri = os.environ.get("NEO4J_URI")
neo4j_user = os.environ.get("NEO4J_USERNAME")
neo4j_password = os.environ.get("NEO4J_PASSWORD")

gds = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

def create_constraints(session):
    """
    Creates a uniqueness constraint Cypher query on :Node(id).

    Args:
        session (Session): Neo4j Session object from the Neo4j driver.
    """   
     
    session.run("""
    CREATE CONSTRAINT node_id_unique IF NOT EXISTS
    FOR (n:Node)
    REQUIRE n.id IS UNIQUE
    """)

def load_nodes(session, filepath):
    """
    Loads nodes from a CSV file into the Neo4j database through batching.

    Args:
        session (Session): Neo4j session object from the Neo4j driver.
        filepath (str): The file path to the CSV file containing node data.
                        The CSV file should have columns: 'id', 'type', 'title', 'name', 'content'.
    """    
    
    df_nodes = pd.read_csv(filepath)
    # convert 'id' to integer if necessary
    df_nodes['id'] = df_nodes['id'].astype(int)

    # batch size
    batch_size = 1000
    total_rows = len(df_nodes)
    for i in range(0, total_rows, batch_size):
        batch = df_nodes.iloc[i:i+batch_size]
        nodes = batch.to_dict('records')
        session.execute_write(create_nodes_batch, nodes)
        print(f"Processed {min(i+batch_size, total_rows)} out of {total_rows} nodes")

def create_nodes_batch(tx, nodes):
    """
    Neo4j Cypher query to create nodes in the database.

    Args:
        tx (tx): The transaction object provided by the Neo4j driver.
        nodes (list): A list of node dictionaries to be created.
                      Each dictionary should have keys: 'id', 'type', 'title', 'name', 'content'.
    """    
    
    tx.run("""
    UNWIND $nodes AS node
    MERGE (n:Node {id: node.id})
    SET n.type = node.type,
        n.title = node.title,
        n.name = node.name,
        n.content = node.content
    """, nodes=nodes)

def load_relationships(session, filepath):
    """
    Loads relationships from a CSV file into the Neo4j database using batching.

    Args:
        session (Session): Neo4j Session object from the Neo4j driver.
        filepath (str): The file path to the CSV file containing relationships.
                        The CSV file should have columns: 'from_id', 'to_id', and 'type'.
    """    
    
    df_rels = pd.read_csv(filepath)
    # convert IDs to integers
    df_rels['from_id'] = df_rels['from_id'].astype(int)
    df_rels['to_id'] = df_rels['to_id'].astype(int)

    # batch size
    batch_size = 1000
    total_rows = len(df_rels)
    for i in range(0, total_rows, batch_size):
        batch = df_rels.iloc[i:i+batch_size]
        relationships = batch.to_dict('records')
        session.execute_write(create_relationships_batch, relationships)
        print(f"Processed {min(i+batch_size, total_rows)} out of {total_rows} relationships")

def create_relationships_batch(tx, relationships):
    """
    Executes a Cypher query to create relationships in the Neo4j database for a batch of data.

    Args:
        tx (tx): The transaction object provided by the Neo4j driver.
        relationships (list): A list of relationship dictionaries to be created.
                              Each dictionary should have keys: 'from_id', 'to_id', and 'type'.
    """    
    
    tx.run("""
    UNWIND $relationships AS rel
    MATCH (from:Node {id: rel.from_id})
    MATCH (to:Node {id: rel.to_id})
    MERGE (from)-[:LINK]->(to)
    """, relationships=relationships)
    
def append_embeddings(session, filepath):
    """
    Reads embeddings from a CSV file and updates the 'plotEmbedding' property of nodes in the database.

    Args:
        session (Session): Neo4j Session object from the Neo4j driver.
        filepath (str): The file path to the CSV file containing embeddings.
                        The CSV file should have columns: 'id', 'embedding'.
    """
    df_embeddings = pd.read_csv(filepath)
    df_embeddings['id'] = df_embeddings['id'].astype(int)

    # Convert 'embedding' from JSON string to list
    df_embeddings['embedding'] = df_embeddings['embedding'].apply(json.loads)

    # Batch size
    batch_size = 1000
    total_rows = len(df_embeddings)
    for i in range(0, total_rows, batch_size):
        batch = df_embeddings.iloc[i:i+batch_size]
        embeddings = batch.to_dict('records')
        session.execute_write(update_embeddings_batch, embeddings)
        print(f"Processed {min(i+batch_size, total_rows)} out of {total_rows} embeddings")
        
def update_embeddings_batch(tx, embeddings):
    """
    Updates the 'plotEmbedding' property of nodes in the database for a batch of embeddings.

    Args:
        tx (Transaction): The transaction object provided by the Neo4j driver.
        embeddings (list): A list of dictionaries with 'id' and 'embedding' keys.
    """
    tx.run("""
    UNWIND $embeddings AS embedding
    MATCH (n:Node {id: embedding.id})
    SET n.plotEmbedding = embedding.embedding
    """, embeddings=embeddings)

def main():
    with gds.session() as session:
        # create_constraints(session)
        # load_nodes(session, 'nodes.csv') 
        append_embeddings(session, 'embeddings.csv')
        # load_relationships(session, 'relationships.csv') 

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        gds.close()
