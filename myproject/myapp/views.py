from .hybrid_search import hybrid_search
from django.shortcuts import render
from nltk.corpus import words

VALID_WORDS = set(words.words())

def is_gibberish(query):
    words_in_query = query.split()
    valid_words_count = sum(1 for word in words_in_query if word.lower() in VALID_WORDS)

    # If fewer than 20% of words are valid, consider it gibberish
    if valid_words_count / len(words_in_query) < 0.2:
        return True
    return False




def index(request):
    return render(request, 'index.html')


def search(request):
    
    query = request.POST['searched']
    if is_gibberish(query): query = None
    
    search_results = [] 
    
    
    def dynamic_weighting(query):
    # Simple rule: if the query is short and specific, prioritize keyword search
        query_length = len(query.split())
        if query_length > 5:
            return 0.3, 0.7
        elif query_length <= 3:
            return 0.7, 0.3
        else:
            return 0.5, 0.5
        
        
    if query:
        weight_keyword, weight_semantic = dynamic_weighting(query)
        search_results = hybrid_search(query,weight_keyword,weight_semantic)
        
    return render(request, 'search.html', {'query': query, 'results': search_results})


# # Load JSON data (from a file or database)
# def load_json_data():
#     with open('final_database_v1.json', 'r') as file:
#         return json.load(file)

# # Load pre-trained model for semantic similarity
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Define weights for keyword and semantic search
# # KEYWORD_WEIGHT = 0.5  # Weight for exact keyword match
# # SEMANTIC_WEIGHT = 0.5  # Weight for semantic similarity

# # Search function
# def hybrid_search(request):
#     query = request.POST.get('searched', '')
#     json_data = load_json_data()
    
#     def dynamic_weighting(query):
#     # Simple rule: if the query is short and specific, prioritize keyword search
#         query_length = len(query.split())
#         if query_length > 5:
#             keyword_weight = 0.3
#             semantic_weight = 0.7
#         elif query_length <= 3:
#             keyword_weight = 0.7
#             semantic_weight = 0.3
#         else:
#             keyword_weight = 0.5
#             semantic_weight = 0.5
        
#         # Normalize weights to ensure they sum to 1
#         total_weight = keyword_weight + semantic_weight
#         keyword_weight /= total_weight
#         semantic_weight /= total_weight
        
#         return keyword_weight, semantic_weight

#     if query:
#         KEYWORD_WEIGHT, SEMANTIC_WEIGHT = dynamic_weighting(query)

#         # **1. Keyword Search**: Basic matching
#         keyword_results = []
#         for item in json_data:
#             # Check for keyword match
#             keyword_score = 0
#             if query.lower() in item['name'].lower():
#                 keyword_score += 1
#             if query.lower() in item['title'].lower():
#                 keyword_score += 1
#             if query.lower() in item['description'].lower():
#                 keyword_score += 1

#             # Add keyword result if there's a match
#             if keyword_score > 0:
#                 keyword_results.append((item, keyword_score))

#         semantic_results = []
#         if not keyword_results:
#             # Ensure valid data
#             filtered_data = [item for item in json_data if 'description' in item]
#             descriptions = [item['description'] for item in filtered_data]
            
#             if not descriptions:
#                 return JsonResponse({'results': []})  # No descriptions to search

#             # Generate embeddings
#             query_embedding = model.encode(query, convert_to_tensor=True)
#             description_embeddings = model.encode(descriptions, convert_to_tensor=True)

#             # Calculate similarity scores
#             scores = util.pytorch_cos_sim(query_embedding, description_embeddings)[0]

#             # Append results with range check
#             for i, score in enumerate(scores):
#                 if i < len(filtered_data):  # Ensure index is valid
#                     semantic_results.append((filtered_data[i], score.item()))
#                 else:
#                     print(f"Index {i} out of range for filtered_data.")

        
        
#         # **Combine Keyword & Semantic Results with Weighting**:
#         all_results = []

#         # Add weighted keyword results
#         for result, keyword_score in keyword_results:
#             total_score = (keyword_score * KEYWORD_WEIGHT)
#             all_results.append((result, total_score))

#         # Add weighted semantic results
#         for result, semantic_score in semantic_results:
#             total_score = (semantic_score * SEMANTIC_WEIGHT)
#             all_results.append((result, total_score))

#         # Sort results by total score (highest first)
#         sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)

#         # Return sorted results in the JSON response or template
#         top_results = [result[0] for result in sorted_results]  # Get the top result items
    
#     return render(request, 'search.html',{'query': query,'results': top_results})






# def cosine_similarity(embedding1, embedding2):
#     return 1 - cosine(embedding1, embedding2)


# def search(request):
#     from .models import Article
#     query = request.POST['searched']
#     articles = []

#     if query:
#         # Keyword-based search
#         keyword_results = Article.objects.filter(
#             Q(title__icontains=query) | Q(description__icontains=query)
#         )

#         # Semantic search
#         query_embedding = get_article_embedding(query)
#         semantic_results = []
#         for article in Article.objects.all():
#             similarity = cosine_similarity(query_embedding, article.embedding)
#             semantic_results.append((article, similarity))

#         # Sort semantic results by similarity
#         semantic_results.sort(key=lambda x: x[1], reverse=True)
        
#         # Combine and remove duplicates (if any)
#         articles = list(set(keyword_results) | set([x[0] for x in semantic_results]))
        
#     else:
#         articles = Article.objects.all()

#     return render(request, 'search_results.html', {'articles': articles})






# def search(request):
#     # Initialize the BM25 index with txtai
#     keyword_index = Embeddings()

#     # Initialize the sentence transformer model for semantic search
#     semantic_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

#     # Load your constitutional articles from the JSON file
#     with open('constitution_of_india.json', 'r') as file:
#         articles = json.load(file)

#     # Create list of articles for keyword indexing (BM25)
#     index_data = [{"article": article["article"], "description": article["description"]} for article in articles]  
#     # Index the articles for keyword search
#     keyword_index.index(index_data)

#     # Generate sentence embeddings for the semantic index
#     article_texts = [article["description"] for article in articles]
#     article_embeddings = semantic_model.encode(article_texts, convert_to_tensor=True)

#     # Now you can use keyword_index and article_embeddings for hybrid search functionality  
#     # Add your search logic here (e.g., handle user query)
#     query = request.POST.get("query")
#     # Keyword search example (using BM25)
#     keyword_results = keyword_index.search(query, 5)  # Adjust the number of results
    
#     # Semantic search example (using cosine similarity with sentence embeddings)
#     query_embedding = semantic_model.encode([query], convert_to_tensor=True)
#     semantic_results = cosine_similarity(query_embedding, article_embeddings)
    
#     # You can combine the results from both searches to perform hybrid search here

#     # For example, if you return the results:
#     return render(request, 'search.html',{
#         "keyword_results": keyword_results,
#         "semantic_results": semantic_results.tolist(),  # Convert tensor to list for JSON serialization
#     })






# Load the model and articles
# model = SentenceTransformer("nli-mpnet-base-v2")
# with open('constitution_of_india.json', 'r') as f:
#     articles = json.load(f)
# article_embeddings = [model.encode(article['description']) for article in articles]

# def search(request):
#     query = request.POST['searched']
#     if query:
#         query_embedding = model.encode(query)
#         similarities = cosine_similarity([query_embedding], article_embeddings)
#         top_indices = similarities[0].argsort()[-10:][::-1]
#         results = [{"article":articles[i]["article"],"title": articles[i]["title"], "description": articles[i]["description"]} for i in top_indices]
#     else:
#         results = []

#     return render(request, 'search.html', {'query': query, 'results': results})

# Create your views here.



# def search(request):
#     laws=None
#     if request.method == "POST":
#         raw_searched = request.POST['searched']
#         searched_values = raw_searched.split()
#         for searched in searched_values:
#             laws = Laws.objects.filter(desc__contains=searched) | Laws.objects.filter(title__contains=searched) | Laws.objects.filter(law_name__contains=searched)
#         return render(request, 'search.html', {'laws':laws,'raw_searched': raw_searched})
#     else:
#         return render(request, 'search.html')


