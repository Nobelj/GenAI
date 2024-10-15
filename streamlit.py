import requests
import streamlit as st
import soundfile as sf
import suno
import ollama
import torch
import pandas as pd
import poe_api_wrapper as poe
import asyncio

print("Loaded")
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


# def prompt_formatter(prompt):
#     prompt_format = """Turn the following into a simple lyric/jingle/acronym/mnemonic that make it easy to remember the text:

# %s

# Give just the text, nothing else""" % (
#         prompt
#     )
#     return prompt_format


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

    st.title("🎵Everybaba MEMnic🎵")

    with st.expander("See explanation"):
        st.write("Remember information easily with music!")

    for message in st.session_state["messages"]:
        print("Message: ", message)
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    st.session_state["model"] = st.selectbox(
        "Select Model",
        ("gpt4_o_mini", "claude_3_haiku", "ollama_server"),
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

    smth = st.experimental_audio_input(
        "Speak here to transcribe", label_visibility="visible"
    )

    if text_area := st.chat_input("Enter your description......."):
        st.session_state["messages"].append({"role": "user", "content": text_area})

        with st.chat_message("user"):
            st.markdown(text_area)

        # retry = True
        # while retry:
        #     retry = True
        #     with st.chat_message("assistant"):
        #         message = st.write_stream(model_res_generator())
        #         if st.button("Continue?"):
        #             retry = False
        #         while retry:
        #             pass

        with st.chat_message("assistant"):
            message = st.write_stream(
                model_res_generator(bot=st.session_state["model"])
            )

            with st.spinner("Generating music"):
                audio_url = generate_music(message)
            if audio_url:
                st.write("Audio URL: ", audio_url)
                st.audio(audio_url, format="audio/mpeg", loop=False)
            else:
                st.write("Failed to generate music.")

            st.session_state["messages"].append(
                {"role": "assistant", "content": message}
            )


# time_slider = st.slider("Select time duration (In Seconds)", 0, 20, 10)

# if text_area and time_slider:

#     if st.button("Generate Music"):
#         st.write("Generating music...")
#         audio_file_path = generate_music(text_area, time_slider)

#         if audio_file_path:
#             st.write("Saving generated music...")
#             st.audio(audio_file_path, format="audio/wav")
#         else:
#             st.write("Failed to generate music.")


if __name__ == "__main__":
    main()