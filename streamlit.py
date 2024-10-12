import ollama
import streamlit as st
import torch
import pandas as pd

st.title("Musicify Chatbot")

# initialize history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# init models
if "model" not in st.session_state:
    st.session_state["model"] = ""

models = [model["name"] for model in ollama.list()["models"]]
st.session_state["model"] = st.selectbox("Choose your model", models)

"TABLET CIPROFLOXACIN (500 mg)
1 TABLET IN MORNING AND 1 TABLET IN EVENING FOR 5 DAYS
TABLET PARACETAMOL (500 mg)
TAKE 1 TABLET WHENEVER FEVER IS MORE THAN 100 F TO
THE MAXIMUM OF 4 TABLETS IN A DAY"

def model_res_generator():
    if torch.cuda.is_available():
        # Set the global PyTorch device to GPU
        device = torch.device("cuda")
        # torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        # Use CPU if no GPU available
        device = torch.device("cpu")

    stream = ollama.chat(
        model=st.session_state["model"],
        messages=st.session_state["messages"],
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]


st.write(
    pd.DataFrame({"first column": [1, 2, 3, 4], "second column": [10, 20, 30, 40]})
)
# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?", ("Email", "Home phone", "Mobile phone")
)

# Add a slider to the sidebar:
add_slider = st.sidebar.slider("Select a range of values", 0.0, 100.0, (25.0, 75.0))

# Display chat messages from history on app rerun
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter prompt here.."):
    # add latest message to history in format {role, content}
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message = st.write_stream(model_res_generator())
        st.session_state["messages"].append({"role": "assistant", "content": message})
