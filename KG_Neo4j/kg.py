import os
from dotenv import load_dotenv
import ssl
import certifi
from neo4j import GraphDatabase
import pandas as pd


load_dotenv()

# OpenAI API key

# Neo4j config and constraints
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_user = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

print(f"NEO4J_URI: {neo4j_uri}")
print(f"NEO4J_USERNAME: {neo4j_user}")

context = ssl.create_default_context(cafile=certifi.where())

gds = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

def create_constraints(session):
    # Create uniqueness constraint on :Node(id)
    session.run("""
    CREATE CONSTRAINT node_id_unique IF NOT EXISTS
    FOR (n:Node)
    REQUIRE n.id IS UNIQUE
    """)

# def load_nodes(session, filepath):
#     df_nodes = pd.read_csv(filepath)
#     # Convert 'id' to integer if necessary
#     df_nodes['id'] = df_nodes['id'].astype(int)

#     # Iterate over the DataFrame and create nodes
#     for index, row in df_nodes.iterrows():
#         session.run("""
#         MERGE (n:Node {id: $id})
#         SET n.type = $type,
#             n.title = $title,
#             n.name = $name,
#             n.content = $content
#         """, parameters={
#             'id': row['id'],
#             'type': row['type'],
#             'title': row['title'],
#             'name': row['name'],
#             'content': row['content']
#         })

# instead use batching
def load_nodes(session, filepath):
    df_nodes = pd.read_csv(filepath)
    df_nodes['id'] = df_nodes['id'].astype(int)

    # Batch size
    batch_size = 1000
    total_rows = len(df_nodes)
    for i in range(0, total_rows, batch_size):
        batch = df_nodes.iloc[i:i+batch_size]
        nodes = batch.to_dict('records')
        session.write_transaction(create_nodes_batch, nodes)

def create_nodes_batch(tx, nodes):
    for node in nodes:
        tx.run("""
        MERGE (n:Node {id: $id})
        SET n.type = $type,
            n.title = $title,
            n.name = $name,
            n.content = $content
        """, parameters=node)

def load_relationships(session, filepath):
    df_rels = pd.read_csv(filepath)
    # Convert IDs to integers
    df_rels['from_id'] = df_rels['from_id'].astype(int)
    df_rels['to_id'] = df_rels['to_id'].astype(int)

    # Iterate over the DataFrame and create relationships
    for index, row in df_rels.iterrows():
        session.run("""
        MATCH (from:Node {id: $from_id})
        MATCH (to:Node {id: $to_id})
        MERGE (from)-[r:LINK]->(to)
        """, parameters={
            'from_id': row['from_id'],
            'to_id': row['to_id']
        })

def main():
    with gds.session() as session:
        create_constraints(session)
        load_nodes(session, 'node.csv')  # Update the path if necessary
        load_relationships(session, 'relationships.csv')  # Update the path if necessary

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        gds.close()
