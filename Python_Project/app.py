import streamlit as st
from dotenv import load_dotenv
import os
import time
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript
from urllib.parse import urlparse, parse_qs
from langdetect import detect
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_classic.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain_classic.chains.combine_documents.map_reduce import MapReduceDocumentsChain


# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

translation_prompt = """You are an expert language translator. Please translate the following text into English:
"""

def extract_full_transcript(youtube_video_url):
    start_time = time.time()
    try:
        parsed_url = urlparse(youtube_video_url)
        query_params = parse_qs(parsed_url.query)
        video_id = query_params.get("v", [None])[0] # Video ID
        if not video_id:
            return "Invalid YouTube URL."
        ytt_api = YouTubeTranscriptApi()  # IMP
        transcript_list = ytt_api.list(video_id)
        transcript = None
        try:

            transcript = transcript_list.find_manually_created_transcript(
                [t.language_code for t in transcript_list]
            )
        except Exception:
            try:
                # Fallback to generated
                transcript = transcript_list.find_generated_transcript(
                    [t.language_code for t in transcript_list]
                )
            except Exception:
                return "No transcript found in any language."
        fetched = transcript.fetch()  # returns a FetchedTranscript object
        
        # Turn it into the old list-of-dicts format, then join text
        raw = fetched.to_raw_data()
        transcript_text = " ".join([i.get("text", "") for i in raw]).strip()

        print(f"Time taken for extract_full_transcript: {time.time() - start_time:.2f} seconds")
        return transcript_text if transcript_text else "Transcript fetch returned empty text."

    except (TranscriptsDisabled, NoTranscriptFound, CouldNotRetrieveTranscript):
        return "Could not retrieve transcript. Try a different video."
    except Exception as e:
        return f"Error: {str(e)}"


def chunk_text(text, chunk_size=3000):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

def translate_full_transcript(text):
    start_time = time.time()
    model = ChatOpenAI(model="gpt-3.5-turbo")
    chunks = chunk_text(text, 3000)
    translated_chunks = [model.invoke([HumanMessage(content=translation_prompt + chunk)]).content for chunk in chunks]
    print(f"Time taken for translate_full_transcript: {time.time() - start_time:.2f} seconds")
    return " ".join(translated_chunks)

def get_vector_store(text):
    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    text_chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in text_chunks]
    db = FAISS.from_documents(documents, OpenAIEmbeddings())
    db.save_local("yt_faiss_index")
    print(f"Time taken for get_vector_store: {time.time() - start_time:.2f} seconds")

def get_conversational_chain():
    start_time = time.time()
    prompt_template = """
    Answer the question as detailed as possible from the provided context, including chat history. 
    If the answer is not in the provided context, say "answer is not available in the context."
    And also if user asks about the chat history respond accordingly

    Chat History:
    {chat_history}
    
    Context:
    {context}
    
    Question: {question}
    Answer:
    """
    model = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history", "context", "question"])
    print(f"Time taken for get_conversational_chain: {time.time() - start_time:.2f} seconds")
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def get_conversational_2nd():
    start_time = time.time()
    query_analysis_prompt = PromptTemplate(
        template="""
        Classify the following question into one of the categories: 'short', 'detailed', or 'long_summary'.
        
        Question: {user_question}
        Category:
        """,
        input_variables=["user_question"]
    )
    model = ChatOpenAI(model="gpt-3.5-turbo")
    chain = LLMChain(llm=model, prompt=query_analysis_prompt)
    print(f"Time taken for get_conversational_2nd: {time.time() - start_time:.2f} seconds")
    return chain

def summarize_text(text):
    """
    This function takes a long text (e.g., YouTube transcript) as input and returns a summarized response.
    """
    
    # Step 1: Split the input text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Each chunk has around 1000 characters
        chunk_overlap=100  # Overlapping to maintain context
    )
    documents = text_splitter.create_documents([text])
    
    # Step 2: Initialize OpenAI LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo") # Use "gpt-3.5-turbo" for a cheaper option

    # Step 3: Define prompts for summarization
    map_prompt = PromptTemplate.from_template(
    "Summarize the following transcript chunk in 6-10 bullet points, capturing key facts and examples:\n\n{context}"
     )

    reduce_prompt = PromptTemplate.from_template(
    "Combine the chunk summaries into a single structured summary with these sections:\n"
    "1) Overview\n2) Key Concepts\n3) Step-by-step Process (if any)\n4) Examples\n5) Conclusion\n\n"
    "{context}"
    )
    # Step 4: Create chains
    map_chain = LLMChain(llm=llm, prompt=map_prompt)
    reduce_llm_chain = LLMChain(llm=llm, prompt=reduce_prompt)
    
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_llm_chain,
        document_variable_name="context"
    )
    
    reduce_documents_chain = ReduceDocumentsChain(
        combine_documents_chain=combine_documents_chain
    )

    # Step 5: Define the full MapReduceDocumentsChain
    chain = MapReduceDocumentsChain(
        llm_chain=map_chain,
        reduce_documents_chain=reduce_documents_chain
    )
    # Step 6: Run the chain and return the summary
    summary = chain.run(documents)
    return summary

def user_input(after_translation):
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    user_question = st.chat_input("Ask a question from the YouTube video")
    if user_question:
        new_db = FAISS.load_local("yt_faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        total_chunks = new_db.index.ntotal  # ✅ Correct way to get total indexed vectors
        print("Total chunks in the vector store: ", total_chunks)
        chain1 = get_conversational_2nd()  # Get the chain to classify question type
        response1 = chain1.run(user_question)
        print(response1,"outside of the if statement")
        chat_history = "\n".join(st.session_state["chat_history"][-10:])
        docs=[]
        if "long_summary" in response1.lower(): 
            print("it is in long summary if sattement")
            response = {}
            if after_translation and isinstance(after_translation, str) and after_translation.strip():
                response['output_text']= summarize_text(after_translation)
            else:
                response['output_text']= " In long summary statement"
                st.error(response['output_text'])
        else: 
            print("it is  in else statement")
            docs = new_db.similarity_search(user_question,k=4) 
            chain = get_conversational_chain()
            response = chain({"chat_history": chat_history, "input_documents": docs, "question": user_question}, return_only_outputs=True)

        if not docs:  # If no documents are retrieved
            docs = ["Sorry, I couldn't find relevant information in the database."]
        # Store chat history properly as strings, not tuples
        st.session_state["chat_history"].append(f"User: {user_question}")
        st.session_state["chat_history"].append(f"AI: {response['output_text']}")
        
    # Display chat history as bubbles
    for message in st.session_state["chat_history"][-10:]:
        role, text = message.split(": ", 1)
        with st.chat_message("user" if role == "User" else "assistant"):
            st.write(text)

def main():
    st.set_page_config("Chat with YouTube Videos")
    st.header("Chat with YouTube Video using OpenAI 💁")

    if "after_translation" not in st.session_state:
        st.session_state.after_translation = None
    if "current_video" not in st.session_state:
        st.session_state.current_video = None

    with st.sidebar:
        st.title("Menu:")
        youtube_url = st.text_input("Provide the Link")
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if st.session_state.current_video != youtube_url:
                    st.session_state.current_video = youtube_url
                    st.session_state.chat_history = []  # Reset chat history for new video
                transcript = extract_full_transcript(youtube_url)
                if transcript and isinstance(transcript, str) and transcript.strip():
                    detected_lang = detect(transcript)
                    st.session_state.after_translation = (
                        transcript if detected_lang == "en" else translate_full_transcript(transcript)
                    )
                    get_vector_store(st.session_state.after_translation)
                    st.success("Processing Complete")
                else:
                    st.error("Failed to retrieve or process transcript.")
                    st.session_state.after_translation = None

    
    user_input(st.session_state.after_translation)

if __name__ == "__main__":
    main()