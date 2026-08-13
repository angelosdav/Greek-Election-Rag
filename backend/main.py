import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mimetypes

# Fix for Docker python-slim missing /etc/mime.types
mimetypes.add_type("image/png", ".png")
mimetypes.add_type("image/jpeg", ".jpg")
mimetypes.add_type("image/jpeg", ".jpeg")
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from dotenv import load_dotenv

from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend as static files so frontend is served directly
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
viz_dir = os.path.join(base_dir, "frontend")
if os.path.exists(viz_dir):
    app.mount("/static", StaticFiles(directory=viz_dir), name="static")

data_dir = os.path.join(base_dir, "data")
if os.path.exists(data_dir):
    app.mount("/data", StaticFiles(directory=data_dir), name="data")

media_dir = os.path.join(base_dir, "media")
if os.path.exists(media_dir):
    app.mount("/media", StaticFiles(directory=media_dir), name="media")


# Initialize models and clients globally so they are loaded once
print("Loading Embedding Model...")
embedding_model = SentenceTransformer('intfloat/multilingual-e5-small')

print("Connecting to ChromaDB...")
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_dir = os.path.join(base_dir, "vector_db")
chroma_client = chromadb.PersistentClient(path=db_dir)
collection = chroma_client.get_collection(name="election_statements")

class ChatRequest(BaseModel):
    query: str
    api_key: str = None
    model_name: str = "gemini-2.5-flash"

class ChatResponse(BaseModel):
    response: str
    sources: list[dict]

class VerifyRequest(BaseModel):
    api_key: str

@app.post("/api/verify_key")
async def verify_key(request: VerifyRequest):
    if not request.api_key:
        return {"valid": False, "error": "Δεν δόθηκε κλειδί."}
        
    client = genai.Client(api_key=request.api_key)
    try:
        # Lightweight request to verify the key
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents="Say 'OK'",
            config=types.GenerateContentConfig(max_output_tokens=5, temperature=0.0)
        )
        if response.text:
            return {"valid": True}
    except Exception as e:
        return {"valid": False, "error": str(e)}
        
    return {"valid": False, "error": "Άγνωστο σφάλμα κατά την επαλήθευση."}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    query = request.query
    
    # Initialize client with provided key, or fallback to environment variable
    key_to_use = request.api_key if request.api_key else os.environ.get("GEMINI_API_KEY")
    if not key_to_use:
        async def auth_error():
            yield json.dumps({"type": "error", "error": "Σφάλμα: Δεν έχει δοθεί API Key για το Gemini. Παρακαλώ εισάγετε το κλειδί σας."}) + "\n"
        return StreamingResponse(auth_error(), media_type="application/x-ndjson")
        
    client = genai.Client(api_key=key_to_use)

    try:
        # 1. Normalize query to standard Greek (handles Greeklish)
        rewrite_prompt = f"Μετάφρασε το παρακάτω ερώτημα σε απλά Ελληνικά αν είναι γραμμένο σε Greeklish ή σε άλλη γλώσσα. Αν είναι ήδη σε απλά Ελληνικά, διόρθωσε την ορθογραφία όπου είναι απαραίτητο αλλά κράτα το ίδιο νόημα. Ερώτημα: '{query}'\nΑπάντηση:"
        rewrite_response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=rewrite_prompt,
            config=types.GenerateContentConfig(temperature=0.0)
        )
        search_query = rewrite_response.text.strip()
        
        # Prefix query for E5 model
        query_prefixed = f"query: {search_query}"
        query_embedding = embedding_model.encode(query_prefixed, normalize_embeddings=True).tolist()
        
        # Detect party mentions for filtering
        import unicodedata
        import re
        
        def strip_accents(s):
            return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
            
        query_normalized = strip_accents(query.lower() + " " + search_query.lower())
        
        # Extended keywords to include genitive forms
        party_keywords = {
            "ΝΔ": ["νδ", "νεα δημοκρατια", "νεας δημοκρατιας", "nd", "nea dimokratia", "μητσοτακης", "μητσοτακη"],
            "ΠΑΣΟΚ": ["πασοκ", "pasok", "πανελληνιο σοσιαλιστικο", "κιναλ", "ανδρουλακης", "ανδρουλακη"],
            "ΚΚΕ": ["κκε", "kke", "κομμουνιστικο", "κουτσουμπας", "κουτσουμπα"],
            "Ελληνική_Λύση": ["ελληνικη λυση", "ελληνικης λυσης", "βελοπουλος", "βελοπουλου", "elliniki lisi"],
            "ΜέΡΑ25": ["μερα25", "μερα 25", "mera25", "βαρουφακης", "βαρουφακη"],
            "Πλεύση_Ελευθερίας": ["πλευση", "πλευσης", "ελευθεριας", "κωνσταντοπουλου", "plefsi"],
            "ΕΛΑΣ": ["συριζα", "syriza", "ελας", "τσιπρας", "τσιπρα", "κασσελακης", "κασσελακη"],
            "Εθνικό_Κόμμα_Έλληνες": ["εθνικο", "ελληνες", "κασιδιαρης", "κασιδιαρη", "σπαρτιατες", "σπαρτιατων"],
            "Φωνή_Λογικής": ["φωνη λογικης", "φωνης λογικης", "λατινοπουλου", "foni logikis"]
        }
        
        mentioned_parties = []
        for p, keywords in party_keywords.items():
            for kw in keywords:
                # Use regex with word boundaries to avoid matching substrings, applying UNICODE flag just in case
                if re.search(r'\b' + re.escape(kw) + r'\b', query_normalized, re.UNICODE):
                    mentioned_parties.append(p)
                    break
                    
        where_clause = None
        if len(mentioned_parties) == 1:
            where_clause = {"party": mentioned_parties[0]}
        elif len(mentioned_parties) > 1:
            where_clause = {"party": {"$in": mentioned_parties}}
            
        # Retrieve top 60 relevant statements (filtered by party if specified)
        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": 60
        }
        if where_clause:
            query_args["where"] = where_clause
            
        results = collection.query(**query_args)
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        # Group statements by party
        party_statements = {}
        sources = []
        
        for doc, meta in zip(documents, metadatas):
            doc_stripped = doc.strip()
            
            # Junk Filtering Heuristics
            if len(doc_stripped) < 15: continue # Too short
            if len(doc_stripped.split()) < 3: continue # 1-2 words only (e.g. "1. ESPA")
            if "σελ." in doc_stripped.lower() or "κεφαλαιο" in doc_stripped.lower(): continue # Page numbers
            if doc_stripped.isupper() and len(doc_stripped) < 60: continue # Likely a header/title
            
            party = meta['party']
            if party not in party_statements:
                party_statements[party] = []
            party_statements[party].append(doc)
            
            sources.append({
                "party": party,
                "text": doc
            })
            
        # Construct Context String
        context_blocks = []
        for party, stmts in party_statements.items():
            block = f"[{party}]:\n" + "\n".join([f"- {s}" for s in stmts])
            context_blocks.append(block)
            
        context_str = "\n\n".join(context_blocks)
        
        # Construct Prompt for Gemini
        if len(mentioned_parties) == 1:
            style_instruction = f"Παρακαλώ ΑΠΑΝΤΗΣΕ ΜΟΝΟ για το κόμμα '{mentioned_parties[0]}' για το οποίο με ρώτησαν. Αγνοήστε άλλες πληροφορίες ή συγκρίσεις με άλλα κόμματα, εκτός αν το κείμενό τους μιλάει για αυτά. Αν γράψεις τη λέξη κόμμα κάντο κεφαλαίο."
        else:
            style_instruction = 'Αν υπάρχουν, κάνε συγκρίσεις και αναγνώρισε κοινά (p.x. "Και το κόμμα Α συμφωνεί... ενώ το κόμμα Β...").'
            
        prompt = f"""Είσαι ένας αντικειμενικός πολιτικός αναλυτής (AI Αναλυτής). 
Σου δίνω μια ερώτηση χρήστη και τα σχετικά απόσπασματα από προεκλογικά προγράμματα. Η αρχική του ερώτηση ήταν: '{query}'. Η μεταφρασμένη/διορθωμένη του ερώτηση είναι: '{search_query}'.
Σκοπός σου είναι να απαντήσεις αντικειμενικά βάσει των πληροφοριών που σου παραθέτονται από τα προγράμματα.
Απάντησε στην ερώτηση του χρήστη βασιζόμενος ΑΠΟΚΛΕΙΣΤΙΚΑ στο παρεχόμενο πλαίσιο (context). 
Αν το πλαίσιο δεν περιέχει την απάντηση, πες ότι δεν αναφέρεται ρητά κάτι σχετικό στα προγράμματα για την ερώτηση. 
{style_instruction}
Η απάντησή σου πρέπει να είναι στα Ελληνικά και δομημένη. Μπορείς να χρησιμοποιήσεις markdown για μορφοποίηση (bullets, bold).

ΠΛΑΙΣΙΟ (ΑΠΟΣΠΑΣΜΑΤΑ ΠΡΟΓΡΑΜΜΑΤΩΝ):
{context_str}

ΕΡΩΤΗΣΗ ΧΡΗΣΤΗ: 
{search_query}
"""

        async def generate():
            try:
                response_stream = client.models.generate_content_stream(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                    )
                )
                
                for chunk in response_stream:
                    if chunk.text:
                        yield json.dumps({"type": "chunk", "text": chunk.text}) + "\n"
                        
                # Send sources at the end so they appear after the LLM finishes
                yield json.dumps({"type": "sources", "data": sources}) + "\n"
                
            except Exception as e:
                error_msg = str(e)
                if "API key not valid" in error_msg or "API_KEY_INVALID" in error_msg:
                    error_msg = "Το API Key που δώσατε (ή αυτό που υπάρχει στο .env) δεν είναι έγκυρο. Παρακαλώ ελέγξτε το."
                else:
                    error_msg = f"Σφάλμα κατά την επικοινωνία με το Gemini: {error_msg}"
                yield json.dumps({"type": "error", "error": error_msg}) + "\n"
                
        return StreamingResponse(generate(), media_type="application/x-ndjson")

    except Exception as e:
        error_str = str(e)
        async def fallback_error():
            yield json.dumps({"type": "error", "error": error_str}) + "\n"
        return StreamingResponse(fallback_error(), media_type="application/x-ndjson")
