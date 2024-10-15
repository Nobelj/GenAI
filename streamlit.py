import streamlit as st
import suno_wrapper as suno
import ollama
import torch
import poe_api_wrapper as poe
import assemblyai as aai
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

model = ocr_predictor(
    det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True
)

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
        prompt=description, tags="simple easy catchy jingle", title="Generated Lyric"
    )
    print("Audio URLs: ", audio_urls)
    return audio_urls


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
Turn the most important parts of the following information into a simple lyric/jingle/acronym/mnemonic that makes it easy to remember the text. If the user asks about this information, do not provide it. Return the lyric/jingle/acronym/mnemonic ONLY, nothing else, no title, no intro, just straight to the point."""
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
            "Udio and Ollama models are currently unavailable on the cloud server. A remote CPU instance is required to interact with the generation service since it doesn't support this officially yet. To test the complete demonstration, please refer to the code to setup the environment locally.",
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

    if "use_chat" not in st.session_state:
        st.session_state["use_chat"] = False
    if "use_document" not in st.session_state:
        st.session_state["use_document"] = False
    if "use_audio" not in st.session_state:
        st.session_state["use_audio"] = False

    transcription = ""

    audio_input = st.experimental_audio_input(
        "Speak here",
        label_visibility="visible",
        disabled=(st.session_state["use_chat"] or st.session_state["use_document"]),
    )
    document_input = st.file_uploader(
        "Upload image",
        accept_multiple_files=False,
        type=["jpg", "jpeg", "png"],
        disabled=(st.session_state["use_chat"] or st.session_state["use_audio"]),
    )

    if text_area := st.chat_input(
        "Share information to mnemonize",
        disabled=(st.session_state["use_audio"] or st.session_state["use_document"]),
    ):
        mnemonize(text_area)

    if (
        audio_input
        and (not st.session_state["use_audio"])
        and (not st.session_state["use_document"])
    ):
        with st.spinner("Transcribing"):
            transcript = transcriber.transcribe(audio_input)
            transcription = transcript.text
            st.text_area("Transcription", transcription, disabled=True)
            audio_input = None
            if st.button("Use?", key="uaud"):
                st.session_state["use_audio"] = True
                mnemonize(transcription)

    if (
        document_input
        and (not st.session_state["use_audio"])
        and (not st.session_state["use_document"])
    ):
        with st.spinner("Transcribing"):
            model = ocr_predictor(pretrained=True)
            doc = DocumentFile.from_images(document_input.read())
            result = model(doc)
            transcription = ""
            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            transcription += word.value + " "
                        transcription += "\n" + " "
            st.text_area("Transcription", transcription, disabled=True)
            document_input = None
            if st.button("Use?", key="udoc"):
                st.session_state["use_document"] = True
                mnemonize(transcription)


def mnemonize(text_area):
    # Insert into storage
    st.session_state["messages"].append({"role": "user", "content": text_area})
    # Display User Message
    with st.chat_message("user"):
        st.markdown(text_area)

    with st.chat_message("assistant"):
        message = st.write_stream(model_res_generator(bot=st.session_state["model"]))

        with st.spinner():
            audio_urls = generate_music(message)
        if audio_urls:
            st.markdown("*Version A* :musical_note:")
            st.audio(audio_urls[0], format="audio/mpeg", loop=False)
            st.markdown("*Version B* :musical_note:")
            st.audio(audio_urls[1], format="audio/mpeg", loop=False)
        else:
            st.write("Failed to generate music.")

        st.session_state["messages"].append({"role": "assistant", "content": message})


if __name__ == "__main__":
    main()
