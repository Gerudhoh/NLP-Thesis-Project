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

labels = []
cases = []
for index, row in plaintiffs_clean.iterrows():
    words = row['Raw Data']
    misc = row['misc']
    if(pd.notna(words) and pd.notna(misc)):
        case_number = misc['Case Information']['Case Number']
        if not case_number in cases:
            entity_labels = []
            word_token = nltk.word_tokenize(words)
            POS_tags = nltk.pos_tag(word_token)
            ne = nltk.ne_chunk(POS_tags)
            for node in ne:
                if isinstance(node, nltk.Tree):
                    if(node.label() != "GPE"):
                        entity_labels.append(node.label())
                        
            lbl = determine_label(entity_labels, words)
            labels.append(lbl)

            # If it's a medical-related case, ensure we don't duplicate it
            if lbl["secondary"] != "N/A":
                cases.append(case_number)
        
        # Test we aren't skipping any cases. If we are, we'll have to add the case number to plaintiffs_clean
        elif not pd.notna(misc):
            print("jinkies")

named_entities = [label["primary"] for label in labels]
secondary_labels = [label["secondary"] for label in labels]          
plaintiffs_clean.insert(0, "Entity", named_entities, True)
plaintiffs_clean.insert(1, "Secondary Label", secondary_labels, True)
plaintiffs_clean