import os
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import openai
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Initialize the LLM (GPT model via OpenAI API)
openai.api_key = os.getenv('OPENAI_API_KEY')  # Ensure the API key is set as an environment variable

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
embedding_dimension = 384  # Dimensionality of embeddings for 'all-MiniLM-L6-v2'
index = faiss.IndexFlatL2(embedding_dimension)

# Metadata storage
metadata = []  # List to store metadata (e.g., URLs, section titles)


# Function to scrape content from a website
def scrape_website(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract all paragraphs
        paragraphs = soup.find_all('p')
        content = [para.get_text(strip=True) for para in paragraphs if para.get_text(strip=True)]
        return content
    except Exception as e:
        print(f"Error scraping website: {e}")
        return []


# Function to chunk content into smaller parts
def chunk_content(content, chunk_size=100):
    chunks = []
    for text in content:
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
    return chunks


# Function to embed chunks using SentenceTransformer
def embed_chunks(chunks):
    embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
    return embeddings


# Function to add embeddings and metadata to the FAISS index
def add_to_index(embeddings, url, chunks):
    global metadata
    embeddings = embeddings.astype('float32')
    index.add(embeddings)

    # Store metadata (e.g., source URL and chunk text)
    metadata.extend([{"url": url, "text": chunk} for chunk in chunks])


# Function to query the FAISS index
def query_index(query, top_k=3):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True).astype('float32')
    distances, indices = index.search(query_embedding, top_k)
    return indices[0], distances[0]


# Function to generate a response using OpenAI GPT
def generate_response(query, relevant_chunks):
    context = "\n".join(relevant_chunks)
    prompt = (
        f"Answer the following query based on the given context:\n\n"
        f"Context:\n{context}\n\n"
        f"Query: {query}\n\n"
        f"Response:"
    )
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # Choose the GPT engine
            prompt=prompt,
            max_tokens=200,
            temperature=0.7
        )
        return response['choices'][0]['text'].strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I couldn't generate a response."


# Main RAG Pipeline Function
def rag_pipeline(urls, query):
    all_chunks = []

    # Step 1: Scrape and Process Website Content
    for url in urls:
        content = scrape_website(url)
        if content:
            chunks = chunk_content(content)
            all_chunks.extend(chunks)

            # Step 2: Embed and Add to FAISS Index
            embeddings = embed_chunks(chunks)
            add_to_index(embeddings, url, chunks)

    # Step 3: Query Handling
    indices, distances = query_index(query)
    relevant_chunks = [metadata[i]["text"] for i in indices] if indices.size > 0 else []

    # Step 4: Response Generation
    response = generate_response(query, relevant_chunks)

    return response


# Example Usage
if __name__ == "__main__":
    urls = ["https://in.bookmyshow.com/explore/home/hyderabad"]  # Replace with your target URLs
    user_query = "What is the main topic of the article?"

    response = rag_pipeline(urls, user_query)
    print(f"{Fore.GREEN}Response: {Style.BRIGHT}{response}")  # Print the response in bright green
