def determine_label(entity_labels, plaintiff_string):
    for term in medical_terms:        
        if term in plaintiff_string.casefold():
            return { "primary": "MEDICAL", "secondary": term.upper() }

    for term in medical_embeddings_both:    
        if term in plaintiff_string.casefold():
            return { "primary": "MEDICAL_EMBED_BOTH", "secondary": term.upper() }
        
    for term in medical_embeddings_w2v:        
        if term in plaintiff_string.casefold():
            return { "primary": "MEDICAL_EMBED_W2V", "secondary": term.upper() }
        
    for term in medical_embeddings_glove:   
        if term in plaintiff_string.casefold():
            return { "primary": "MEDICAL_EMBED_GLOVE", "secondary": term.upper() }
        
    if len(entity_labels) > 0:
        return { "primary": max(set(entity_labels), key = entity_labels.count), "secondary": "N/A" }
        
    return { "primary": "NO_LABEL", "secondary": "N/A" }

named_entities = []
for index, row in p.iterrows():
    words = row['plaintiff']
    entity_labels = []
    if(pd.notna(words)): 
        word_token = nltk.word_tokenize(words)
        POS_tags = nltk.pos_tag(word_token)
        ne = nltk.ne_chunk(POS_tags)
        for node in ne:
            if isinstance(node, nltk.Tree):
                if(node.label() != "GPE"):
                    entity_labels.append(node.label())
                    
        lbl = determine_label(entity_labels, words)
        named_entities.append([lbl['primary'], lbl['secondary'], words])
        
ner_frame = pd.DataFrame(named_entities, columns =['Entity', 'Secondary Label', 'plaintiff'])
ner_frame.Entity.value_counts()