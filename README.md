# LLM-proof
Repo for Algoverse Research

# 1 Getting Started

# 1.1 Create a virtual environment 
- Create a virtual enviornment in the root directory of the repository.
- You may replace "myenv" with a name of your choice
```
python -m venv myenv
```

# 1.2 Activate the virtual environment

# On macOS and Linux
```
source myenv/bin/activate
```

# On Windows
```
myenv\Scripts\activate
```

# 1.3 Install Dependencies
To install all packages and code dependencies please run:
```
pip install -r requirements.txt
```

# 1.4 Set Enviornment Variables
1. In the root directory of the repository, make a new file named **.env**
2. Copy + Paste the content from the file named **.env.example** into the newly created **.env** file
3. Set the variables to the necessary values

# 2 Extracting Proofs to Your Own Neo4j AuraDB

# 2.1 Retrieving the XML Dump
1. Press this link: https://proofwiki.org/xmldump/latest.xml.gz which will download a zip file
2. Unzip the file to extract **latest.xml**
3. Move **latest.xml** to the root directory of the repository

# 2.2 Extracting to Nodes and Relationships from the XML file to CSV files
run the command 
```
python Graph_Creation/extractProofsXML.py latest.xml
```

# 2.3 Uploading CSV files to Neo4j AuraDB
run the file **neo4j_kg.py**
- this should successfully upload around 20k nodes and 80k relationships (give or take)

# 3 Using the Knowledge Graph to Perform Retrieval Augmented Generation
COMING SOON