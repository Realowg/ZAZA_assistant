import hydralit_components as hc
import streamlit as st
# import streamlit_analytics
from streamlit_modal import Modal

import re

from dotenv import load_dotenv
from PyPDF2 import PdfReader
import streamlit as st
from werkzeug.utils import secure_filename
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import json
from dotenv import load_dotenv

import streamlit.components.v1 as html
# from main_test import *
from htmlTemplates import css, bot_template , user_template

from db_functions import *
from prompts import CHAT_TEMPLATE, INITIAL_TEMPLATE
from prompts import CORRECTION_CONTEXT, COMPLETION_CONTEXT, OPTIMIZATION_CONTEXT, GENERAL_ASSISTANT_CONTEXT, \
    GENERATION_CONTEXT, COMMENTING_CONTEXT, EXPLANATION_CONTEXT, LEETCODE_CONTEXT, SHORTENING_CONTEXT

from utils.components import footer_style, footer
try:
    from streamlit import rerun as rerun
except ImportError:
    # conditional import for streamlit version <1.27
    from streamlit import experimental_rerun as rerun


import os
os.environ["OPENAI_API_KEY"]=st.secrets["openai_key"]

os.environ["TOGETHER_API_KEY"]=st.secrets["togetherai_key"]

from langchain_together import Together




def disclaimer():
    modal2 = Modal(key="ZAZA Key", title="Disclaimers - Welcome to ZAZA AI Assistant", padding=5, max_width=900)

    if 'popup_closed' not in st.session_state:
        st.session_state.popup_closed = False

    if not st.session_state.popup_closed:
        with modal2.container():
            st.markdown('')
            st.markdown(
                'Welcome to ZAZA, your coding assistant chatbot! Before we proceed, please note the following:'
            )
            st.markdown(
                'ZAZA is designed to assist you with coding-related queries and tasks. While we strive to provide accurate and helpful information, we cannot guarantee the accuracy, completeness, or reliability of the responses provided by the chatbot.'
            )
            st.markdown(
                'Please be aware that ZAZA may provide general guidance and suggestions, but it should not be considered a substitute for professional advice. Always verify critical information and consult qualified professionals for specific coding or programming concerns.'
            )
            st.markdown(
                'ZAZA may also utilize third-party APIs and resources to enhance its functionality. Please review the terms of service and data usage policies of these third-party services.'
            )

            value = st.checkbox(
                "By checking this box, you acknowledge and agree to the terms and conditions outlined above."
            )
            if value:
                if st.button('Sumbmit'):
                    st.session_state.popup_closed = True

def approve_password(password):
    if len(password) >= 8 and re.search(r"(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])(?=.*[_@$#!?&*%])", password):
        return True
    return False
    

def approve_email(email):
    email_regex = '^[a-zA-Z0-9]+[\._]?[a-zA-Z0-9]+[@]\w+[.]\w{2,3}$'
    if re.search(email_regex, email):
        return True
    else:
        return False

def set_page_config():
    st.set_page_config(
        page_title='ZAZA',
        page_icon=r"img\ZAZA.png",
        # initial_sidebar_state="expanded"
    )
def user_authentication_tab():
    modal = Modal(key="ZAZA Key", title="Welcome to ZAZA AI Assistant", padding=15, max_width=800)
    if 'popup_closed' not in st.session_state:
        st.session_state.popup_closed = False
    
    if not st.session_state.popup_closed:
        with modal.container():
            with st.expander("User Authentication", expanded=True):
                login_tab, create_account_tab = st.tabs(["Login", "Create Account"])
            # if "user_authenticated" not in st.session_state:
                st.session_state['user_authenticated'] = False
                with login_tab:
                    email = st.text_input("Email:") 
                    password = st.text_input("Password:", type='password')
                    def click_button():
                        st.session_state.popup_closed = True

                    if st.button("Login",on_click=click_button):
                        if authenticate_user(email=email,password=password):
                            st.session_state.popup_closed = True
                            st.session_state.user_authenticated = True
                            # disclaimer()

                        else:
                            st.caption('Incorrect Username or Password.')




                with create_account_tab:
                    new_email = st.text_input("New Email:")
                    new_password = st.text_input("New Password:", type='password')
                    confirm_password = st.text_input("Confirm Password:", type='password')
                    if st.button("Create Account"):
                        if not approve_email(new_email):
                            st.caption("Invalid Email")
                            return
                        if not approve_password(new_password):
                            st.caption("Invalid Password")
                            return
                        if new_password != confirm_password:
                            st.caption("Passwords do not match")
                            return
                        add_user_to_db(email=new_email, password=new_password)
                        st.caption(f"{new_email} Successfully Added")

# ******************************************************************************************************************************************************************************************************************************************************************************************************
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


user_authentication_tab()
set_page_config()
CodeLlama = Together(
    model="codellama/CodeLlama-70b-Python-hf",

)
# ******************************************************************************************************************************************************************************************************************************************************************************************************


# if 'lottie' not in st.session_state:
#     st.session_state.lottie = False

# if not st.session_state.lottie:
#     lottfinder = load_lottiefile(r"C:\Users\Midhun\upwork\Task-1\Eve-Coding-Assistant\AppV7\TFinder\TFinder\.streamlit\Animation - 1715971093216.json")
#     st.lottie(lottfinder, speed=1.3, loop=False)

#     time.sleep(10)
#     st.session_state.lottie = True
#     rerun()

max_width_str = f"max-width: {75}%;"

st.markdown(f"""
        <style>
        .appview-container .main .block-container{{{max_width_str}}}
        </style>
        """,
            unsafe_allow_html=True,
            )

st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    
                }
        </style>
        """, unsafe_allow_html=True)

# Footer

st.markdown(footer_style, unsafe_allow_html=True)

# NavBar
# ******************************************************************************************************************************************************************************************************************************************************************************************************
def init_ses_states():
    default_values = {
        'chat_history': [],
        'initial_input': "",
        'initial_context': "",
        'scenario_context': "",
        'thread_id': "",
        'docs_processed': False,
        'docs_chain': None,
        'user_authenticated': False,
        'uploaded_docs': None,
        'current_user_id': None
    }
    for key, value in default_values.items():
        st.session_state.setdefault(key, value)
# Get Current Users User ID


def page_title_header():
    # top_image = Image.open('trippyPattern.png')
    # st.image(top_image)
    st.title("Zaza Coding Assistant")
    st.caption("Powered by OpenAI, LangChain, Streamlit")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# def create_retrieval_chain(vectorstore):
#     if model =='CodeLlama':
#         llm=CodeLlama
#     else:
#         llm = ChatOpenAI(temperature=temperature, model_name=model)
#     # llm = ChatOpenAI(temperature=temperature, model_name=model)

#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain


def create_llm_chain(prompt_template):
    memory = ConversationBufferMemory(input_key="input", memory_key="chat_history", )
    if model =='CodeLlama':
        llm=CodeLlama
    else:
        llm = ChatOpenAI(temperature=temperature, model_name=model)
    return LLMChain(llm=llm, prompt=prompt_template, memory=memory,output_key="result")

def settings():
    global language, scenario, temperature, model, scenario_context, libraries, pdf_docs, uploaded_docs
    languages = sorted(['Python', 'GoLang','TypeScript', 'JavaScript', 'Java', 'C', 'C++', 'C#', 'R', 'SQL'])
    python_libs = sorted(['SQLite','PyGame','Seaborn',"Pandas",'Numpy','Scipy','Scikit-Learn','PyTorch','TensorFlow','Streamlit','Flask','FastAPI'])
    scenarios = ['General Assistant', 'Code Correction', 'Code Completion', 'Code Commenting', 'Code Optimization', 
                 'Code Shortener','Code Generation', 'Code Explanation', 'LeetCode Solver']
    scenario_context_map = {
        "Code Correction": CORRECTION_CONTEXT,
        "Code Completion": COMPLETION_CONTEXT,
        "Code Optimization": OPTIMIZATION_CONTEXT,
        "General Assistant": GENERAL_ASSISTANT_CONTEXT,
        "Code Generation": GENERATION_CONTEXT,
        "Code Commenting": COMMENTING_CONTEXT,
        "Code Explanation": EXPLANATION_CONTEXT,
        "LeetCode Solver": LEETCODE_CONTEXT,
        "Code Shortener": SHORTENING_CONTEXT,
    }

    with st.sidebar:
    #     # user_authentication_tab()
        if st.session_state.user_authenticated:
            with st.expander(label="Settings",expanded=True):
                coding_settings_tab, chatbot_settings_tab = st.tabs(['Coding','ChatBot'])
                with coding_settings_tab:
                    language = st.selectbox(label="Language", options=languages)
                    if language == "Python":
                        libraries = st.multiselect(label="Libraries",options=python_libs)
                        if not libraries:
                            libraries = ""
                    else:
                        libraries=""
                    scenario = st.selectbox(label="Scenario", options=scenarios, index=0)
                    scenario_context = scenario_context_map.get(scenario, "")
            
                with chatbot_settings_tab:
                    model = st.selectbox("Language Model", options=['CodeLlama','gpt-3.5-turbo','gpt-4-0613','gpt-4'])
                    temperature = st.slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.5)
            
            # with st.expander("Previous Chats", expanded=True):
            #     selected_thread_id = st.selectbox(label="Previous Thread IDs", options=get_unique_thread_ids(), index=0)
            #     if st.button("Render Chat"):
            #         st.session_state.thread_id = selected_thread_id
            #         st.session_state.chat_history = get_all_thread_messages(selected_thread_id)
            #         st.experimental_rerun()


def handle_initial_submit():
    settings()
    global code_input, code_context
    initial_template = PromptTemplate(
        input_variables=['input','language','scenario','scenario_context','code_context','libraries'],
        template= INITIAL_TEMPLATE
    )
    # if pdf_docs and st.session_state.docs_processed:
    if st.session_state.docs_processed:

        initial_llm_chain =st.session_state.docs_chain
    else:
        initial_llm_chain = create_llm_chain(initial_template)
    code_input = st.text_area(label=f"User Code", height=200)
    code_context = st.text_area(label="Additional Context", height=60)
    if (st.button(f'Submit Initial Input') and (code_input or code_context)):
        with st.spinner('Generating Response...'):
            if st.session_state.docs_processed:
                llm_input = initial_template.format(input=code_input,
                                                code_context=code_context, 
                                                language=language, 
                                                scenario=scenario,
                                                scenario_context=scenario_context,
                                                libraries=libraries)
                initial_response = initial_llm_chain({'question':llm_input})['answer']
            else: 
                initial_response = initial_llm_chain.run(input=code_input,
                                                        code_context=code_context, 
                                                        language=language, 
                                                        scenario=scenario,
                                                        scenario_context=scenario_context,
                                                        libraries=libraries)
        st.session_state.update({
            'initial_input': code_input,
            'initial_context': code_context,
            'chat_history': [f"{code_input}", f"{initial_response}"],
        })
        st.session_state.thread_id = (code_context+code_input)[:40]
        write_to_messages_db(thread_id=st.session_state.thread_id, 
                             role='USER',
                             message=f"USER: CODE CONTEXT:{code_context} CODE INPUT: {code_input}"
                            )
        write_to_messages_db(thread_id=st.session_state.thread_id, 
                             role='AI',
                             message=f"AI: {initial_response}"
                            )


def handle_user_message():
    chat_template = PromptTemplate(
        input_variables=['input','language','scenario','scenario_context','chat_history','libraries','code_input','code_context','most_recent_ai_message'],
        template=CHAT_TEMPLATE
    )
    if st.session_state.docs_processed:
        chat_llm_chain = st.session_state.docs_chain
    else:
        chat_llm_chain = create_llm_chain(prompt_template=chat_template)
    if st.session_state.chat_history:
        user_message = st.text_area("Further Questions for Coding AI?", key="user_input", height=60)
        if st.button("Submit Message") and user_message:
            with st.spinner('Generating Response...'):
                most_recent_ai_message = st.session_state.chat_history[-1]
                if st.session_state.docs_processed:
                    chat_input = chat_template.format(input=user_message, 
                                                    language=language, 
                                                    scenario=scenario,
                                                    scenario_context=scenario_context,
                                                    libraries=libraries,
                                                    code_input=st.session_state.initial_input,
                                                    code_context=st.session_state.initial_context,
                                                    most_recent_ai_message=most_recent_ai_message)
                    chat_response = chat_llm_chain({'question':chat_input})['answer']
                else:
                    chat_response = chat_llm_chain.run(input=user_message,
                                                        language=language, 
                                                        scenario=scenario,
                                                        scenario_context=scenario_context,
                                                        chat_history=st.session_state.chat_history,
                                                        libraries=libraries,
                                                        code_input=st.session_state.initial_input,
                                                        code_context=st.session_state.initial_context,
                                                        most_recent_ai_message=most_recent_ai_message)
                st.session_state['chat_history'].append(f"USER: {user_message}")
                st.session_state['chat_history'].append(f"AI: {chat_response}")
                write_to_messages_db(thread_id=st.session_state.thread_id, 
                             role='USER',
                             message=f"USER: {user_message}"
                            )
                write_to_messages_db(thread_id=st.session_state.thread_id, 
                             role='AI',
                             message=f"AI: {chat_response}"
                            )
# def display_convo():
#     with st.container():
#         # for message in st.session_state.chat_history:
#         #     with st.chat_message(message[USER]):
#         #         st.markdown(message["AI"])
#         # for i, message in enumerate(st.session_state.chat_history):
#         #     print("********************************************************")
#         #     print(message)    
#         #     if i % 2 == 0:
#         #         st.write(user_template.replace(
#         #             "{{MSG}}", message), unsafe_allow_html=True)
#         #     else:
#         #         st.write(bot_template.replace(
#         #             "{{MSG}}", message), unsafe_allow_html=True)

#         for i, message in enumerate((st.session_state.chat_history)):
#             if i % 2 == 0:
#                 st.markdown(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)
#             else:
#                 st.markdown(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)
user_template = """
<div style="background-color: #DCF8C6; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
    <strong>User:</strong> {{MSG}}
</div>
"""

bot_template = """
<div style="background-color: #E1F3FB; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
    <strong>Bot:</strong> {{MSG}}
</div>
"""

def display_convo():
    with st.container():
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.markdown(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)
            else:
                st.markdown(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)



def home_page():
    st.write("This is the Home page")
    # settings()
    # main()
    create_users_db()
    create_messages_db()
    create_log_table()
    init_ses_states()
    page_title_header()


def allapp_page():
    st.write("This is the All Application page")
    with st.expander("Previous Chats", expanded=True):
        selected_thread_id = st.selectbox(label="Previous Thread IDs", options=get_unique_thread_ids(), index=0)
        if st.button("Render Chat"):
            st.session_state.thread_id = selected_thread_id
            st.session_state.chat_history = get_all_thread_messages(selected_thread_id)
            st.experimental_rerun()


def resource_page():
    # st.sidebar.set_visible(False)
    st.write("This is the Resource page")
    handle_initial_submit()
    handle_user_message()
    display_convo()

HOME = 'Home'
APPLICATION = 'AI Assistant'
RESOURCE = 'Session History'
CONTACT = 'Settings'

tabs = [
    HOME,
    APPLICATION,
    RESOURCE,
    CONTACT
]

option_data = [
    {'icon': "🏠", 'label': HOME},
    {'icon': "🖥️", 'label': APPLICATION},
    {'icon': "📑", 'label': RESOURCE},
    {'icon': "📑", 'label': CONTACT},
]

over_theme = {'txc_inactive': '#6c6c6f',  # Matches the very dark grey text color
              'menu_background': '#072f3e',  # Matches the very dark primary color
              'txc_active': '#f7f7f8',  # Matches the very dark grey text color
              'option_active': '#0f5f77'}  # Matches the original bright cyan color

font_fmt = {'font-class': 'h3', 'font-size': '50%', 'color': '#f7f7f8'}  # Matches the very dark grey text color



chosen_tab = hc.option_bar(
    option_definition=option_data,
    title='',
    key='PrimaryOptionx',
    override_theme=over_theme,
    horizontal_orientation=True)



# streamlit_analytics.start_tracking()

# Credit
st.sidebar.image(r"img/Logo_4.png",width=150,use_column_width=True)


if __name__ == '__main__':
    load_dotenv()
    # settings()
    # home_page()
    if chosen_tab == HOME:

        home_page()
        

    elif chosen_tab == APPLICATION:
        resource_page()   

    elif chosen_tab == RESOURCE:
        allapp_page()

    elif chosen_tab == CONTACT:
        # settings()
        st.write("coming soon")

    for i in range(4):
        st.markdown('#')
    st.markdown(footer, unsafe_allow_html=True)
