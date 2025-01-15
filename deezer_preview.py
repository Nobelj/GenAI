import streamlit as st
import requests


# Function to search Deezer API
def search_deezer(query):
    url = f"https://api.deezer.com/search?q={query}"
    response = requests.get(url)
    return response.json()

# Search bar
query = st.text_input("Search for a track or artist:")

if query:
    results = search_deezer(query)
    tracks = results.get("data", [])

    if tracks:
        track_titles = [
            track["title"] + " - " + track["artist"]["name"] for track in tracks
        ]
        selected_track = st.selectbox("Select a track:", track_titles)

        if selected_track:
            track_info = next(
                track
                for track in tracks
                if track["title"] + " - " + track["artist"]["name"] == selected_track
            )
            st.audio(track_info["preview"], format="audio/mp3")
    else:
        st.write("No results found.")
