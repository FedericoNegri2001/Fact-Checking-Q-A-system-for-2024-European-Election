import streamlit as st
from openai import OpenAI
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# file CSS 
def add_custom_css(file_name: str):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Configurazione del client MongoDB
mongo_uri = "insert your mongodb uri"
client = MongoClient(mongo_uri)
db = client['db_europee']
collection = db['articles']
circoscrizioni_collection = db['circoscrizioni']

# Configurazione del modello di embedding
embedding_model = SentenceTransformer("thenlper/gte-large")

# Funzione per ottenere gli embedding
def get_embedding(text: str) -> list:
    if not text.strip():
        st.error("Testo vuoto fornito per l'embedding.")
        return []
    embedding = embedding_model.encode(text)
    return embedding.tolist()

def vector_search(user_query, collection):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
    user_query (str): The user's query string.
    collection (MongoCollection): The MongoDB collection to search.

    Returns:
    list: A list of unique matching documents with the highest vectorSearchScore.
    """

    # Generate embedding for the user query
    query_embedding = get_embedding(user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    # Define the vector search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "europee_indice",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 150,  # Number of candidate matches to consider
                "limit": 50,  # Retrieve more than 4 to ensure uniqueness
            }
        },
        {
            "$project": {
                "_id": 0,  
                "title": 1,  
                "text": 1,  
                "publish_date": 1,  
                "author": 1,  
                "score": {"$meta": "vectorSearchScore"},  # Include the search score
            }
        },
        {
            "$sort": {"score": -1}  # Sort by vectorSearchScore descending
        }
    ]

    # Execute the search
    results = list(collection.aggregate(pipeline))

    # Ensure the results are unique by title
    unique_results = []
    seen_titles = set()

    for result in results:
        if result['title'] not in seen_titles:
            unique_results.append(result)
            seen_titles.add(result['title'])
        
        if len(unique_results) == 3:
            break

    return unique_results


# Funzione per ottenere i risultati della ricerca
def get_search_result(query, collection):
    results = vector_search(query, collection)
    if not results:
        return "Nessun risultato trovato."
    search_result = ""
    for result in results:
        search_result += f"Titolo: {result.get('title', 'N/A')}, Testo: {result.get('text', 'N/A')}, Publicato il {result.get('publish_date', 'N/A')}, Autore: {result.get('author', 'N/A')} \n"
    return search_result, results

# Funzione per la ricerca vettoriale nel database
def vector_search2(user_query, collection):
    query_embedding = get_embedding(user_query)
    if not query_embedding:
        return "Invalid query or embedding generation failed."
    # Define the vector search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "europee_indice",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 150,  # Number of candidate matches to consider
                "limit": 50,  # Retrieve more than 4 to ensure uniqueness
            }
        },
        {
            "$project": {
                "_id": 0,  
                "title": 1,  
                "text": 1,  
                "publish_date": 1,  
                "author": 1,  
                "score": {"$meta": "vectorSearchScore"},  # Include the search score
            }
        },
        {
            "$sort": {"score": -1}  # Sort by vectorSearchScore descending
        }
    ]
    results = list(collection.aggregate(pipeline))
    unique_results = []
    seen_titles = set()
    for result in results:
        if result['text'] not in seen_titles:
            unique_results.append(result)
            seen_titles.add(result['text'])
        if len(unique_results) == 20:
            break
    return unique_results

# Funzione per ottenere i risultati della ricerca
def get_search_result2(query, collection):
    results = vector_search2(query, collection)
    if not results:
        return "Nessun risultato trovato."
    search_result = ""
    for result in results:
        search_result += f"Titolo: {result.get('title', 'N/A')}, Testo: {result.get('text', 'N/A')}, Publicato il {result.get('publish_date', 'N/A')}, Autore: {result.get('author', 'N/A')} \n"
    return search_result, results


#Page config
st.set_page_config(
    page_title='Elezioni Europee 2024',
    page_icon='🇪🇺'
    )
add_custom_css("style.css")


#title
st.title("Benvenuto Nella Dashboard delle Elezioni Europee 2024 ")

#Aggiunta video
video_path = "Elezioni .mp4"

# Visualizza il video
video_file = open(video_path, "rb")
video_bytes = video_file.read()
st.video(video_bytes)

st.write(" ")
#DOMANDE A LLM
st.markdown("### Inserisci la tua domanda al nostro assistente politico")
user_query = st.text_input("")
if user_query:
  # Ottenere informazioni dalle fonti
  source_information,results = get_search_result(user_query, collection)
  
  
  # Combinare la query e le informazioni di provenienza
  combined_information = f"Query: {user_query}\nPer rispondere considera i seguenti risultati:\n{source_information}"

  # Chiamata al modello di linguaggio
  client = OpenAI(api_key='insert your openai api')

  completion = client.chat.completions.create(
      model="gpt-4o",
      messages=[
          {"role": "system", "content": "Sei un assistente politico che dovrà rispondere alle domande sulle elezioni europee 2024. Rispondi in modo chiaro e conciso, utilizzando paragrafi per separare le diverse idee o punti."},
          {"role": "user", "content": combined_information}
      ]
  )

  # Visualizzare la risposta dell'LLM
  response_message = completion.choices[0].message.content
  st.markdown("### Risposta alla tua domanda:")
  st.write(response_message)

  st.markdown("### Articoli di riferimento:")
  i=1
  for result in results:
        st.markdown("#### Riferimento "+str(i))
        st.write(result.get('title', 'N/A'))
        st.write(result.get('text', 'N/A'))
        st.write(result.get('publish_date', 'N/A'))
        st.write(result.get('author', 'N/A'))
        i=i+1


partiti = [ "FRATELLI D'ITALIA", 'LEGA SALVINI PREMIER', "STATI UNITI D'EUROPA", 'AZIONE - SIAMO EUROPEI','PARTITO DEMOCRATICO', 'MOVIMENTO 5 STELLE', 'ALLEANZA VERDI E SINISTRA', 'FORZA ITALIA - NOI MODERATI - PPE', "PACE TERRA DIGNITA'", 'ALTERNATIVA POPOLARE', "LIBERTA'", 'RASSEMBLEMENT VALDOTAIN', 'SUDTIROLER VOLKSPARTEI (SVP)'
]

#1 sezione Sidebar
#st.sidebar.title("Ricerca Programmi e Candidati per Partito (Powered by GPT-4o)")
st.sidebar.markdown('<div class="sidebar-title">Ricerca Programmi e Candidati per Partito (Powered by GPT-4o)</div>', unsafe_allow_html=True)
# Sezione per il programma politico
with st.sidebar.expander("Ricerca Programma Partito"):
    # Selezione del partito
    partito_scelto = st.selectbox("Seleziona il Partito", partiti)

    # Bottone per visualizzare il programma politico
    visualizza_programma = st.button("Visualizza Programma")

if visualizza_programma:
    if not partito_scelto:
        st.error("Seleziona un partito per visualizzare il programma politico.")
    else:
        #query per il programma politico del partito selezionato
        user_query = f"parlami del programma politico alle europee del 2024 del {partito_scelto}"

        # Ottenere informazioni dalle fonti
        source_information, results = get_search_result(user_query, collection)
        combined_information = f"Query: {user_query}\nPer rispondere considera i seguenti risultati:\n{source_information}"

        # Chiamata al modello di linguaggio
        client = OpenAI(api_key='insert your openai api')

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Sei un assistente politico che dovrà rispondere alle domande sulle elezioni europee 2024. Rispondi in modo chiaro e conciso, utilizzando paragrafi per separare le diverse idee o punti."},
                {"role": "user", "content": combined_information}
            ]
        )

        #risposta dell'LLM fuori dalla sidebar
        response_message = completion.choices[0].message.content
        st.markdown("### Presentazione del Partito alle Europee")
        st.image("foto/" + partito_scelto + ".png")
        st.write(response_message)


# Sezione per la selezione del candidato
with st.sidebar.expander("Ricerca Informazioni Candidato"):
    if partito_scelto:
        query_info = {"PARTITO": partito_scelto}
        candidati_circoscrizione = circoscrizioni_collection.find(query_info)
        candidati_set = set()  # Utilizziamo un set per rimuovere i duplicati
        for candidato in candidati_circoscrizione:
            candidati_set.add(f"{candidato['NOME']} {candidato['COGNOME']}")
        candidati = list(candidati_set)  # Convertiamo il set in lista per poterlo usare in selectbox

        candidato_scelto = st.selectbox("Seleziona il Candidato", candidati)

    visualizza_candidato = st.button("Visualizza Informazioni Candidato")


if visualizza_candidato:
    if not candidato_scelto:
        st.error("Seleziona un candidato per visualizzare le informazioni.")
    else:
        #query per il candidato selezionato
        user_query = f"parlami della carriera politica di {candidato_scelto}"

        # Ottenere informazioni dalle fonti
        source_information, results = get_search_result(user_query, collection)
        combined_information = f"Query: {user_query}\nPer rispondere considera i seguenti risultati:\n{source_information}"

        # Chiamata al modello di linguaggio
        client = OpenAI(api_key='insert your openai api')

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Sei un assistente politico che dovrà rispondere alle domande sulle elezioni europee 2024. Rispondi in modo chiaro e conciso, utilizzando paragrafi per separare le diverse idee o punti."},
                {"role": "user", "content": combined_information}
            ]
        )

        response_message = completion.choices[0].message.content

        # Visualizzare la risposta dell'LLM fuori dalla sidebar
        st.markdown("### Carriera Politica del Candidato")
        st.write(response_message)



#2 sezione Sidebar
#st.sidebar.title("Analytics su Candidati, Circoscrizioni e Partiti")
st.sidebar.markdown('<div class="sidebar-title">Analytics su Candidati, Circoscrizioni e Partiti</div>', unsafe_allow_html=True)
elenco_circoscrizioni = [
    " ITALIA NORD-OCCIDENTALE",
    " ITALIA NORD-ORIENTALE",
    " ITALIA CENTRALE",
    " ITALIA MERIDIONALE",
    " ITALIA INSULARE"
]

# Sezione Analytics
with st.sidebar.expander("Ricerca Candidati per Circoscrizione e Partito"):
    circoscrizione_scelta = st.selectbox("Seleziona la Circoscrizione", ["Tutte"] + elenco_circoscrizioni)
    partito_scelto2 = st.selectbox("Seleziona il Partito", ["Tutti"] + partiti)
    Visualizza_Politici_Candidati=st.button("Visualizza Politici Candidati")
# Bottone per visualizzare i politici candidati
if Visualizza_Politici_Candidati:
    if not circoscrizione_scelta:
        st.error("Seleziona una circoscrizione per visualizzare i politici candidati.")
    else:
        #query per i politici candidati nella circoscrizione selezionata
        if circoscrizione_scelta != "Tutte" and partito_scelto2 != "Tutti":
            query_politici = {"CIRCOSCRIZIONE": circoscrizione_scelta, "PARTITO": partito_scelto2}
        elif circoscrizione_scelta != "Tutte":
            query_politici = {"CIRCOSCRIZIONE": circoscrizione_scelta}
        elif partito_scelto2 != "Tutti":
            query_politici = {"PARTITO": partito_scelto2}
        else:
            query_politici = {}

        candidati_circoscrizione = circoscrizioni_collection.find(query_politici)
        elenco_politici_candidati = [
            {
                "Nome": candidato['NOME'],
                "Cognome": candidato['COGNOME'],
                "Partito": candidato['PARTITO'],
                "Circoscrizione": candidato['CIRCOSCRIZIONE']
            }
            for candidato in candidati_circoscrizione
        ]
        df_politici = pd.DataFrame(elenco_politici_candidati)

        # Visualizzazione della tabella
        st.markdown("### Elenco dei Politici Candidati")
        with st.expander("Mostra Elenco dei Politici Candidati"):
            styled_df_politici = df_politici.style.apply(lambda x: ['color: black'] * len(df_politici.columns), axis=1)
            st.write(styled_df_politici)

       

        # Creazione dei grafici a torta
        if circoscrizione_scelta == "Tutte" and partito_scelto2 == "Tutti":
            # Numero di candidati per partito
            partiti_count = df_politici['Partito'].value_counts()
            fig1 = go.Figure(data=[go.Pie(labels=partiti_count.index, values=partiti_count, hole=0.3)])
            fig1.update_layout(title="Numero di candidati per partito")
            st.plotly_chart(fig1, use_container_width=True)

            # Numero di candidati per circoscrizione
            circoscrizioni_count = df_politici['Circoscrizione'].value_counts()
            fig2 = go.Figure(data=[go.Pie(labels=circoscrizioni_count.index, values=circoscrizioni_count, hole=0.3)])
            fig2.update_layout(title="Numero di candidati per circoscrizione")
            st.plotly_chart(fig2, use_container_width=True)
        # Creazione dei grafici a torta aggiuntivi in base alla selezione
        elif partito_scelto2 != "Tutti":
            if circoscrizione_scelta=="Tutte":
                # Numero di candidati per partito e circoscrizione
                candidati_partito = df_politici[df_politici['Partito'] == partito_scelto2]
                circoscrizioni_count = candidati_partito['Circoscrizione'].value_counts()

                fig = go.Figure(data=[go.Pie(labels=circoscrizioni_count.index, values=circoscrizioni_count, hole=0.3)])
                fig.update_layout(title=f"Numero di candidati per circoscrizione del partito {partito_scelto2}")
                st.plotly_chart(fig, use_container_width=True)
        elif circoscrizione_scelta != "Tutte":
             if partito_scelto2=="Tutti":
                # Numero di candidati per circoscrizione e partito
                candidati_circoscrizione = df_politici[df_politici['Circoscrizione'] == circoscrizione_scelta]
                partiti_count = candidati_circoscrizione['Partito'].value_counts()

                fig = go.Figure(data=[go.Pie(labels=partiti_count.index, values=partiti_count, hole=0.3)])
                fig.update_layout(title=f"Numero di candidati per partito della circoscrizione {circoscrizione_scelta}")
                st.plotly_chart(fig, use_container_width=True)


# Nuovo expander per la ricerca per fascia d'età
with st.sidebar.expander("Ricerca Candidati per Fascia d'Età"):
    fascia_eta = st.radio(
        "Seleziona la Fascia d'Età",
        options=["20-30", "30-50", "50-70", "70+"]
    )
    visualizza_eta = st.button("Visualizza Candidati per Fascia d'Età")

if visualizza_eta:
    if not fascia_eta:
        st.error("Seleziona una fascia d'età per visualizzare i candidati.")
    else:
        # Definire i limiti di età in base alla selezione
        if fascia_eta == "20-30":
            eta_min, eta_max = 20, 30
        elif fascia_eta == "30-50":
            eta_min, eta_max = 30, 50
        elif fascia_eta == "50-70":
            eta_min, eta_max = 50, 70
        else:
            eta_min, eta_max = 70, 150  

        #query per i candidati nella fascia d'età selezionata
        query_eta = {"ETA": {"$gte": eta_min, "$lt": eta_max}}
        candidati_eta = circoscrizioni_collection.find(query_eta)
        elenco_candidati_eta = [
            {
                "Nome": candidato['NOME'],
                "Cognome": candidato['COGNOME'],
                "Partito": candidato['PARTITO'],
                "Età": candidato['ETA']
            }
            for candidato in candidati_eta
        ]
        df_eta = pd.DataFrame(elenco_candidati_eta)
        df_eta = df_eta.drop_duplicates(subset=["Nome", "Cognome", "Partito", "Età"])
        # Visualizzazione della tabella
        st.markdown("### Candidati per Fascia d'Età")
        with st.expander("Mostra Elenco dei Politici "):
            styled_df_eta = df_eta.style.apply(lambda x: ['color: black'] * len(df_eta.columns), axis=1)
            st.write(styled_df_eta)
            #st.table(df_eta)

          # Creazione dell'istogramma
        if not df_eta.empty:
            fig = px.histogram(df_eta, x='Partito', title='Numero di Candidati per Partito nella Fascia d\'Età Selezionata', color_discrete_sequence=['#FFD700'] )
            fig.update_layout(
                xaxis_title='Partito',
                yaxis_title='Numero di Candidati',
                bargap=0.2
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Nessun candidato trovato nella fascia d'età selezionata.")
            

#3 sezione Sidebar
#FAQ
st.sidebar.markdown('<div class="sidebar-title">FAQ</div>', unsafe_allow_html=True)

# Nuovo expander per la selezione dei topic
with st.sidebar.expander("Topic più discussi delle Elezioni"):
    # Lista dei topic
    elenco_topic = [
        "esercito comune Europeo",
        "guerra in Ucraina",
        "politica migratoria",
        "riconoscimento Palestina",
        "sanzioni contro la Russia",
        "cambiamento climatico",
        "transizione energetica"
    ]

    # Selezione del topic
    topic_scelto = st.selectbox("Seleziona un Topic", elenco_topic)

    # Bottone per visualizzare le informazioni sul topic selezionato
    visualizza_topic = st.button("Visualizza Informazioni sul Topic")


# Logica per il topic selezionato
if visualizza_topic:
    if not topic_scelto:
        st.error("Seleziona un topic per visualizzare le informazioni.")
    else:
        #query per il topic selezionato
        user_query = f"parlami delle posizioni di tutti i partiti italiani candidati alle europee 2024 riguardo:  '{topic_scelto}' "

        # Ottenere informazioni dalle fonti
        source_information, results = get_search_result(user_query, collection)
        combined_information = f"Query: {user_query}\nPer rispondere considera i seguenti risultati:\n{source_information}"

        # Chiamata al modello di linguaggio
        client = OpenAI(api_key='insert your openai api')

        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Sei un assistente politico che dovrà rispondere alle domande sulle elezioni europee 2024. Rispondi in modo chiaro e conciso, utilizzando paragrafi per separare le diverse idee o punti."},
                {"role": "user", "content": combined_information}
            ]
        )

        response_message = completion.choices[0].message.content

        # Visualizzazione della risposta
        st.markdown(f"### Informazioni sul Topic '{topic_scelto}'")
        st.write(response_message)

        st.markdown("### Articoli di riferimento:")
        i = 1
        for result in results:
            st.markdown(f"#### Riferimento {i}")
            st.write(result.get('title', 'N/A'))
            st.write(result.get('text', 'N/A'))
            st.write(result.get('publish_date', 'N/A'))
            st.write(result.get('author', 'N/A'))
            i += 1


#QUIZ
st.markdown("### TROVA IL PARTITO PIU' AFFINE A TE")
with st.expander("Fai il quiz per scoprire i partiti più affini"):
    
    questions = [
        "Favorevole ad un esercito comune europeo?",
        "Favorevole al Green Deal?",
        "Favorevole al Nucleare?",
        "Favorevole allo stop delle auto a combustione dal 2035?",
        "Favorevole a introdurre il diritto all'aborto nei diritti fondamentali UE?",
        "Favorevole al salario minimo?",
        "Favorevole a dare armi all'Ucraina?"
    ]

    responses = {}
    for question in questions:
        responses[question] = st.radio(question, ("Sì", "No"))

    if st.button("Scopri i partiti più affini"):
        user_preferences = ""
        for question, answer in responses.items():
            if answer == "Sì":
                modified_query = question.rstrip('?')
            else:
                modified_query = "Sfavorevole " + question.split("Favorevole ")[1]


            user_preferences += f"{modified_query}. "

        # Ottenere informazioni dalle fonti per tutte le risposte combinate
        source_information, results = get_search_result2(user_preferences, collection)
        combined_information = f"Preferenze utente: {user_preferences}\nPer rispondere considera i seguenti risultati:\n{source_information}"


        # Chiamata al modello di linguaggio
        client = OpenAI(api_key='insert your openai api')

        completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Sei un assistente politico che dovrà rispondere alle domande sulle elezioni europee 2024. Rispondi in modo chiaro e conciso, confrontando le risposte dell'utente con le posizioni di tutti i partiti italiani in modo da inferire il/i partito/i più afffine/i all'utente."},
            {"role": "user", "content": combined_information}
            ]
            )

        response_message = completion.choices[0].message.content

        st.markdown("### Risultati Dettagliati del Quiz:")
        st.write(response_message)



