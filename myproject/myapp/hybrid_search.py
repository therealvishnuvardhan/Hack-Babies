from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))
REMOVED_STOPWORDS = ['of']
for word in REMOVED_STOPWORDS:
    STOPWORDS.discard(word)

RESTRICTED_WORDS = ['say','says','about','means','mean','a', "an", "the", "on", "in", "at", "by", "to", "for", "with", "about", 
    "into", "onto", "upon", "from", "over", "under", "through", "between", 
    "and", "or", "but", "so", "yet", "nor", "either", "neither", "both", 
    "it", "its", "they", "them", "he", "she", "him", "her", "this", "that", 
    "these", "those", "who", "whom", "whose", "we", "us", "you", "your", 
    "i", "me", "my", "is", "are", "was", "were", "will", "would", "should", 
    "could", "might", "must", "can", "do", "did", "does", "be", "been", "being", 
    "have", "had", "has", "here", "there", "where", "when", "how", "why", "also", 
    "then", "now", "states", "describe", "explain", "define", 
    "what", "does", "any", "please", "tell", "give", "list", "related to", 
    "about", "according to", "what does", "what is", "can you", "tell me", 
    "give me", "list all", '?','??','???']
ABBREVIATIONS = {
    "mva": "motor vehicles act",
    "nia": "negotiable instruments act",
    "ida": "indian divorce act",
    "iea": "indian evidence act",
}



# Initialize the sentence transformer model for semantic search
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load your constitutional names from the JSON file

with open('final_database_v1.json', 'r') as file:
    names =  json.load(file)




def preprocess_query(query):
    
    # Ensure query is a string
    if isinstance(query, list):
        query = " ".join(query)  # Combine list into a single string
    elif not isinstance(query, str):
        raise ValueError(f"Invalid query type: {type(query)}. Expected a string or list.")

    query = query.lower()

    # Expand abbreviations
    for abbr, full_form in ABBREVIATIONS.items():
        query = re.sub(rf'\b{abbr}\b', full_form, query, flags=re.IGNORECASE)

    words = query.split()

    # Filter out stopwords
    filtered_words = [word for word in words if (word not in STOPWORDS) and (word not in RESTRICTED_WORDS)]

    filtered_query = " ".join(filtered_words) # Combine words

    # Match specific patterns
    article_match = re.search(r'\barticle (\d+)\b', filtered_query, re.IGNORECASE)
    mva_section_match = re.search(r'\bmotor vehicles act\s+section (\d+)\b', filtered_query, re.IGNORECASE)
    nia_section_match = re.search(r'\bnegotiable instruments act\s+section (\d+)\b', filtered_query, re.IGNORECASE)
    ida_section_match = re.search(r'\bindian divorce act\s+section (\d+)\b', filtered_query, re.IGNORECASE)
    iea_section_match = re.search(r'\bindian evidence act\s+section (\d+)\b', filtered_query, re.IGNORECASE)

    # Return normalized format based on matches
    if article_match:
        return f"Article {article_match[1]}"
    elif mva_section_match:
        return f"Motor Vehicles Act Section {mva_section_match[1]}"
    elif nia_section_match:
        return f"Negotiable Instruments Act Section {nia_section_match[1]}"
    elif ida_section_match:
        return f"Indian Divorce Act Section {ida_section_match[1]}"
    elif iea_section_match:
        return f"Indian Evidence Act Section {iea_section_match[1]}"

    return filtered_query




def keyword_search(query, names):
    keyword_results = []
    # Extract specific article number if present
    match = re.search(r'\barticle (\d+)\b', query, re.IGNORECASE)
    article_number = match[1] if match else None

    for item in names:
        keyword_score = 0

        # Check for exact article match
        if article_number and f"Article {article_number}" == item["name"]:
            keyword_score += 5  # Assign higher score for exact matches

        # Check for regex match in name, title, and description
        if query.lower() in item['name'].lower():
            keyword_score += 1
        if query.lower() in item['title'].lower():
            keyword_score += 1
        if query.lower() in item['description'].lower():
            keyword_score += 1

        if keyword_score > 0:  # Only consider matches with a score
            keyword_results.append({
                "name": item["name"], 
                "title": item["title"],
                "description": item["description"], 
                "info": item["info"],
                "score": keyword_score
            })

    # Sort results by relevance (highest score first)
    keyword_results.sort(key=lambda x: x["score"], reverse=True)
    return keyword_results



# Create list of names for keyword indexing (BM25)
index_data = [{"name": name["name"], "title": name["title"], "description": name["description"]} for name in names]

# Generate sentence embeddings for the semantic index
name_texts = [name["description"] for name in names]
name_embeddings = semantic_model.encode(name_texts, convert_to_tensor=True)



def hybrid_search(query, weight_keyword, weight_semantic):
    # Preprocess query to normalize and extract specific article number
    processed_query = preprocess_query(query)
    
    match = re.search(r'\barticle (\d+)\b', processed_query, re.IGNORECASE)
    article_number = match[1] if match else None

    # Perform keyword-based search
    keyword_results = keyword_search(processed_query, names)
    # Perform semantic search
    query_embedding = semantic_model.encode([processed_query], convert_to_tensor=True)
    similarities = cosine_similarity(query_embedding, name_embeddings)
    semantic_results_indices = similarities.argsort()[0][-10:][::-1]  # Top 10 results

    results = {
        name["name"]: {
            "name": name["name"],
            "title": name["title"],
            "description": name["description"],
            "info": name["info"],
            "score": float(name["score"])
            * weight_keyword,  # Adjust score by weight
        }
        for name in keyword_results
    }
    # Process semantic-based results
    for idx in semantic_results_indices:
        name = names[idx]
        semantic_score = similarities[0][idx] * weight_semantic  # Adjust score by weight

        # If the query contains an article number, filter semantic results
        if article_number and f"Article {article_number}" != name["name"]:
            continue  # Skip non-matching articles

        # Add or merge results
        if name["name"] not in results:
            results[name["name"]] = {
                "name": name["name"],
                "title": name["title"],
                "description": name["description"],
                "info": name["info"],
                "score": semantic_score
            }
        else:
            # Merge scores
            existing_score = results[name["name"]]["score"]
            results[name["name"]]["score"] = max(existing_score, semantic_score)

    return sorted(results.values(), key=lambda x: x['score'], reverse=True)

