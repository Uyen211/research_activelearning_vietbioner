import os
import json

def extract_entities_from_conll(file_path):
    """
    Extracts all entities of each type from a CoNLL-formatted file.
    Reconstructs entities by joining syllables/words with a space.
    Returns a dictionary mapping entity type to a set of lowercased entity strings.
    """
    entities_by_type = {}
    
    current_tokens = []
    current_type = None
    
    def save_entity():
        nonlocal current_tokens, current_type
        if current_tokens and current_type:
            entity_str = " ".join(current_tokens).lower().strip()
            # Normalize whitespace
            entity_str = " ".join(entity_str.split())
            # Replace underscores with spaces just in case the file has segmented words
            entity_str = entity_str.replace("_", " ")
            if current_type not in entities_by_type:
                entities_by_type[current_type] = set()
            entities_by_type[current_type].add(entity_str)
        current_tokens = []
        current_type = None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                save_entity()
                continue
            parts = line.split()
            if len(parts) < 2:
                save_entity()
                continue
            
            token = parts[0]
            tag = parts[1]
            
            if tag.startswith('B-'):
                save_entity()
                current_type = tag[2:]
                current_tokens.append(token)
            elif tag.startswith('I-'):
                etype = tag[2:]
                if current_type == etype:
                    current_tokens.append(token)
                else:
                    save_entity()
                    current_type = etype
                    current_tokens.append(token)
            else:
                save_entity()
                
        save_entity() # check for last one
        
    return entities_by_type

def main():
    workspace_dir = r"d:\Study\TLU\Nlp\active-learning-VietBioNER"
    dev_path = os.path.join(workspace_dir, "colab", "dataset", "vietbioner", "dev.txt")
    test_path = os.path.join(workspace_dir, "colab", "dataset", "vietbioner", "test.txt")
    
    print("Extracting entities from dev.txt...")
    dev_entities = extract_entities_from_conll(dev_path)
    print("Extracting entities from test.txt...")
    test_entities = extract_entities_from_conll(test_path)
    
    # Merge dev and test entities
    leakage_entities = {}
    for etype in set(list(dev_entities.keys()) + list(test_entities.keys())):
        dev_set = dev_entities.get(etype, set())
        test_set = test_entities.get(etype, set())
        leakage_entities[etype] = dev_set.union(test_set)
        print(f"Type: {etype} - Unique entities in Dev/Test: {len(leakage_entities[etype])}")
        
    # Map gazetteer files to entity types
    gazetteers = {
        "datetime.json": "DateTime",
        "location.json": "Location",
        "symptom_and_disease.json": "Symptom_and_Disease",
        "diagnostic_procedures_tb.json": "DiagnosticProcedure",
        "healthcare_organizations.json": "Organisation"
    }
    
    gaz_dir = os.path.join(workspace_dir, "colab", "dataset", "gazetteer")
    
    for filename, etype in gazetteers.items():
        filepath = os.path.join(gaz_dir, filename)
        if not os.path.exists(filepath):
            print(f"Skipping {filename} - file not found.")
            continue
            
        with open(filepath, 'r', encoding='utf-8') as f:
            terms = json.load(f)
            
        original_count = len(terms)
        leakage_set = leakage_entities.get(etype, set())
        
        # Filter terms
        filtered_terms = []
        removed_terms = []
        for term in terms:
            # Normalize term
            norm_term = " ".join(term.lower().split()).replace("_", " ")
            if norm_term in leakage_set:
                removed_terms.append(term)
            else:
                filtered_terms.append(term)
                
        print(f"Gazetteer: {filename} ({etype})")
        print(f"  Original: {original_count} terms")
        print(f"  Removed {len(removed_terms)} leakage terms")
        print(f"  Remaining: {len(filtered_terms)} terms")
        
        # Write back filtered gazetteer
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(filtered_terms, f, ensure_ascii=False, indent=2)
            
if __name__ == "__main__":
    main()
