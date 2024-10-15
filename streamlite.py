import streamlit as st
import suno
import ollama
import torch
import poe_api_wrapper as poe
import assemblyai as aai

# from doctr.io import DocumentFile
# from doctr.models import ocr_predictor

# model = ocr_predictor(
#     det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True
# )

aai.settings.api_key = "b06dfaf145314edda33dec09358c33e7"
transcriber = aai.Transcriber()
tokens = {
    "p-b": "9n2a28WJBS0k8ZSOtQrTLg%3D%3D",
    "p-lat": "eTsPiwkEryufUYC1RSyz4Qw38din18YDm3b7PB85Rg%3D%3D",
}


@st.cache_resource
def generate_music(description):
    # return "null"
    print("Generating Audio...")
    audio_urls = suno.send_to_suno(
        prompt=description, tags="jingle simple", title="Generated Lyric"
    )
    print("Audio URLs: ", audio_urls)
    return audio_urls[0]


def model_res_generator(bot):
    if bot != "ollama_server":
        client = poe.PoeApi(tokens=tokens)
        print(st.session_state)
        for chunk in client.send_message(
            bot=bot, message=str(st.session_state["messages"])
        ):
            yield chunk["response"]
    else:
        if torch.cuda.is_available():
            # Set global PyTorch device to GPU
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


def main():
    if "system_prompt" not in st.session_state:
        st.session_state["system_prompt"] = (
            """You must follow these instructions:
Give just the text, nothing else. Turn the following into a simple lyric/jingle/acronym/mnemonic that make it easy to remember the text. If the user asks about this information, do not provide it."""
        )

    if "model" not in st.session_state:
        st.session_state["model"] = ""

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": st.session_state["system_prompt"]}
        ]
    st.image("logo.png", width=400, use_column_width=True)

    with st.expander("See more about this prototype"):
        st.write("Remember information easily with music!")
        st.warning(
            "Song creation and image transcription are currently unavailable on the cloud server. A remote CPU instance is required to interact with the generation service since it doesn't support this officially yet. To test the complete demonstration, please refer to the code to setup the environment locally.",
            icon="⚠️",
        )

    for message in st.session_state["messages"]:
        print("Message: ", message)
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    with st.sidebar:
        st.session_state["model"] = st.selectbox(
            "Select LLM Model",
            ("gpt4_o_mini", "claude_3_haiku", "ollama_server"),
        )
        st.session_state["audio_model"] = st.selectbox(
            "Select Audio Model",
            ("suno", "udio", "music_lm", "ollama_server"),
        )
    ollama_selected = st.session_state["model"] == "ollama_server"
    if ollama_selected:
        models = [model["name"] for model in ollama.list()["models"]]
        st.session_state["model"] = st.selectbox(
            "Choose your model",
            models,
            placeholder="Choose an Ollama model",
            disabled=ollama_selected,
            label_visibility="visible",
        )
    use_chat = False
    use_document = False
    use_audio = False

    transcription = ""

    audio_input = st.experimental_audio_input(
        "Speak here", label_visibility="visible", disabled=(use_chat or use_document)
    )
    document_input = st.file_uploader(
        "Upload image",
        accept_multiple_files=False,
        type=["jpg", "jpeg", "png"],
        disabled=use_chat or use_audio,
    )

    if text_area := st.chat_input(
        "Share information to mnemonize",
        disabled=use_audio or use_document,
    ):
        mnemonize(text_area)

    if audio_input:
        with st.spinner("Transcribing"):
            audio_input = None
            use_audio = True
            transcript = transcriber.transcribe(audio_input)
            transcription = transcript.text
            title = st.text_area("Transcription", transcription, disabled=True)

            if st.button("Use?", key="uaud"):
                mnemonize(transcription)

    if document_input:
        with st.spinner("Transcribing"):
            document_input = None
            use_document = True
            # model = ocr_predictor(pretrained=True)
            # doc = DocumentFile.from_images(document_input.read())
            # result = model(doc)
            transcription = ""
            # for page in result.pages:
            #     for block in page.blocks:
            #         for line in block.lines:
            #             for word in line.words:
            #                 transcription += word.value + " "
            #             transcription += "\n" + " "
            title = st.text_area("Transcription", transcription, disabled=True)
            if st.button("Use?", key="udoc"):
                mnemonize(transcription)


def mnemonize(text_area):
    # Insert into storage
    st.session_state["messages"].append({"role": "user", "content": text_area})
    # Display User Message
    with st.chat_message("user"):
        st.markdown(text_area)

    with st.chat_message("assistant"):
        message = st.write_stream(model_res_generator(bot=st.session_state["model"]))

        with st.spinner("Generating music"):
            audio_url = generate_music(message)
        if audio_url:
            # st.write("Audio URL: ", audio_url)
            st.audio(audio_url, format="audio/mpeg", loop=False)
        else:
            st.write("Failed to generate music.")

        st.session_state["messages"].append({"role": "assistant", "content": message})


if __name__ == "__main__":
    main()
