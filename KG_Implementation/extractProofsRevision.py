import re
from lxml import etree
import mwparserfromhell
import pandas as pd
import os
import sys

# define the namespace mapping from xml file
NS = {'mw': 'http://www.mediawiki.org/xml/export-0.11/'}

def parse_xml(xml_file_path):
    """
    Parses the XML file and yields each <page> element.
    """
    print(f"Starting to parse XML: {xml_file_path}")
    try:
        # initialize iterparse with specified namespace and tag filtering
        # using iterparse to avoid loading the entire XML file into memory
        context = etree.iterparse(xml_file_path, events=('end',), tag='{http://www.mediawiki.org/xml/export-0.11/}page')
    except Exception as e:
        print(f"Error initializing iterparse: {e}")
        sys.exit(1)
    
    page_count = 0
    for event, elem in context:
        page_count += 1
        if page_count <= 5:
            # extract title for initial debugging
            title_elem = elem.find('mw:title', namespaces=NS)
            title = title_elem.text if title_elem is not None else 'No Title'
            print(f"Processing Page {page_count}: {title}")
        yield elem
        # clear element to free memory
        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]
    del context
    print(f"Total pages processed in parse_xml: {page_count}")

def is_relevant_page(title, categories, namespace):
    """
    Determines if a page is a Definition, Lemma, Theorem, or Abbreviation based on title patterns,
    categories, and namespace.
    """
    # focus on the Definition namespace (ns=102)
    if namespace != '102':
        return None

    # define patterns for titles
    title_patterns = {
        'def': re.compile(r'^(Definition|Def)\s*:', re.IGNORECASE),
        'lemma': re.compile(r'^(Lemma|Lem)\s*:', re.IGNORECASE),
        'theorem': re.compile(r'^(Theorem|Thm)\s*:', re.IGNORECASE),
        'abbreviation': re.compile(r'^(Abbreviation|Abbr)\s*:', re.IGNORECASE),
    }

    for key, pattern in title_patterns.items():
        if pattern.match(title):
            return key

    # ff title doesn't match, optionally check categories
    # define category patterns
    category_map = {
        'def': re.compile(r'\bDefinition\b', re.IGNORECASE),
        'lemma': re.compile(r'\bLemma\b', re.IGNORECASE),
        'theorem': re.compile(r'\bTheorem\b', re.IGNORECASE),
        'abbreviation': re.compile(r'\bAbbreviation\b', re.IGNORECASE),
    }

    for key, pattern in category_map.items():
        for cat in categories:
            if pattern.search(cat):
                return key

    return None

def extract_content(wikitext):
    """
    Extracts the content from the Wikitext, removing markup.
    """
    wikicode = mwparserfromhell.parse(wikitext)
    text = wikicode.strip_code()
    return text

def collect_nodes(xml_file_path):
    """
    Parses the XML and collects nodes of type Definition, Lemma, Theorem, Abbreviation.
    
    Returns a DataFrame with columns: id, type, title, name, content
    """
    data = []
    total_pages = 0
    relevant_pages = 0
    redirect_pages = 0

    for page in parse_xml(xml_file_path):
        total_pages += 1

        # extract title
        title_elem = page.find('mw:title', namespaces=NS)
        title = title_elem.text if title_elem is not None else ''

        # extract namespace
        ns_elem = page.find('mw:ns', namespaces=NS)
        namespace = ns_elem.text if ns_elem is not None else ''

        # extract categories
        categories = []
        for cat in page.findall('mw:category', namespaces=NS):
            if cat.text:
                categories.append(cat.text)

        # check if the page is a redirect
        redirect = page.find('mw:redirect', namespaces=NS)
        if redirect is not None:
            redirect_pages += 1
            # skip redirects
            continue

        # determine if the page is relevant
        node_type = is_relevant_page(title, categories, namespace)
        if node_type:
            relevant_pages += 1

            # extract text content
            text_elem = page.find('mw:revision/mw:text', namespaces=NS)
            if text_elem is not None and text_elem.text:
                wikitext = text_elem.text
                content = wikitext  # Store raw wikitext with markup

                # extract name from title
                name = title.split(':', 1)[1].strip() if ':' in title else title

                # extract node ID
                node_id_elem = page.find('mw:id', namespaces=NS)
                node_id = node_id_elem.text if node_id_elem is not None else None

                data.append({
                    'id': node_id,
                    'type': node_type,
                    'title': title,
                    'name': name,
                    'content': content  # Use raw wikitext
                })

                # debugging statement for relevant pages
                if relevant_pages <= 5:
                    print(f"Collected {node_type}: {name}")

        # print progress every 1000 pages for debugging
        if total_pages % 1000 == 0:
            print(f"Processed {total_pages} pages, found {relevant_pages} relevant pages so far, skipped {redirect_pages} redirects.")

    df = pd.DataFrame(data)
    print(f"Total pages processed: {total_pages}")
    print(f"Total relevant pages found: {relevant_pages}")
    print(f"Total redirect pages skipped: {redirect_pages}")
    return df


def extract_relationships_from_links(nodes_df):
    """
    Extracts relationships between nodes based on internal links in the content.
    
    Returns a DataFrame with columns: from_id, to_id, type
    """
    relationships = []
    
    # Build a mapping from both 'title' and 'name' to 'id' (case-insensitive)
    title_name_to_id = {}
    for idx, row in nodes_df.iterrows():
        if isinstance(row['title'], str):
            title_name_to_id[row['title'].lower()] = row['id']
        if isinstance(row['name'], str):
            title_name_to_id[row['name'].lower()] = row['id']
    
    for idx, row in nodes_df.iterrows():
        from_id = row['id']
        content = row['content']
        
        # Parse the content to get all wikilinks
        wikicode = mwparserfromhell.parse(content)
        links = wikicode.filter_wikilinks()
        
        for link in links:
            link_target = str(link.title).strip()
            # Remove any fragment (after '#') or pipe (after '|')
            link_target = re.split(r'[#|]', link_target)[0].strip()
            link_target_lower = link_target.lower()
            
            # Map the link target to node IDs
            to_id = title_name_to_id.get(link_target_lower)
            
            if to_id:
                relationships.append({
                    'from_id': from_id,
                    'to_id': to_id,
                    'type': 'LINK'  # Indicate that this is a link relationship
                })
    
    relationships_df = pd.DataFrame(relationships).drop_duplicates()
    print(f"Total relationships extracted: {len(relationships_df)}")
    return relationships_df


def save_to_csv(df, filename):
    """
    Saves the DataFrame to a CSV file.
    """
    if not df.empty:
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"Saved {len(df)} records to {filename}")
    else:
        print(f"No records to save for {filename}")

def main(xml_dump_path):
    """
    Main function to execute the extraction process.
    """
    if not os.path.isfile(xml_dump_path):
        print(f"Error: File '{xml_dump_path}' does not exist.")
        sys.exit(1)

    # Collect nodes
    nodes_df = collect_nodes(xml_dump_path)
    print(f"Number of nodes collected: {len(nodes_df)}")
    save_to_csv(nodes_df, 'nodes.csv')

    # Extract and save relationships
    if not nodes_df.empty:
        relationships_df = extract_relationships_from_links(nodes_df)
        save_to_csv(relationships_df, 'relationships.csv')
    else:
        print("No nodes to extract relationships from.")

if __name__ == "__main__":
    # ensure the script is called with the XML file path
    if len(sys.argv) < 2:
        print("Usage: python extract_knowledge_graph.py /path/to/wikiproof_dump.xml")
        sys.exit(1)

    xml_dump_path = sys.argv[1]
    main(xml_dump_path)
