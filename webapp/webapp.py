from flask import (
    Flask,
    render_template,
    jsonify,
    request,
    Response,
    send_file,
    stream_with_context,
)
from urllib.parse import unquote, quote
from flask_cors import CORS
import logging

import pandas as pd
import numpy as np
import os
import pickle

from options import SONGS_DATA_DF_PATH, USER_DATA_PATH
from search import search, USE_SAMPLE_DATA


app = Flask(__name__)
CORS(app)
# MUSIC_BASE_DIR = "/d16Tb/soundcloud_music/files/"
DATA_DIR = "/home/keld/projects/music_gen/sc_scrape/data/"
DATA_WEBAPP_DIR = "/n4Ta/sc_webapp_data/"


# archive_tracker_path = "/d16Tb/soundcloud_music/archive_tracker"
# out_path = "/d16Tb/soundcloud_music/files/"
data_obj_path = f"{DATA_DIR}sc_data_v2.pckl"

data_all = pickle.load(open(data_obj_path, 'rb'))

logger = logging.getLogger(__name__)
app.logger.setLevel(logging.INFO)

app.logger.info(f"Loading music df...")
all_music_info_df = pd.read_parquet(SONGS_DATA_DF_PATH)
app.logger.info(f"Done")

user_data = pickle.load(open(USER_DATA_PATH, "rb"))


def save_user_data(user_data):
    pickle.dump(user_data, open(USER_DATA_PATH, "wb"))


def stream_audio(filepath):
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(32096)  # You can adjust the chunk size to your preference
            if not chunk:
                break
            yield chunk


@app.route("/stream/<path:music_path>", methods=["GET"])
def stream(music_path):
    app.logger.info(f"call stream")
    # MUSIC_BASE_DIR +
    # print(music_path)
    decoded_filepath = unquote("/" + music_path)
    if 0:
        return Response(stream_audio(decoded_filepath), content_type="audio/mpeg")
    else:
        return send_file(
            decoded_filepath, as_attachment=True, download_name=decoded_filepath
        )

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search_req():
    data = request.get_json()
    query = data.get("query", "").strip() if data else ""
    idx_top_k = []
    if query:
        if query == "*random*":
            idx_top_k = np.random.randint(len(all_music_info_df), size=20)
        else:
            idx_top_k = search(query, all_music_info_df)

    idx_top_k = [int(i) for i in idx_top_k]

    return jsonify(idx_top_k)

@app.route("/get_song_info", methods=["POST"])
def get_song_info():
    data = request.get_json()
    song_ids = data.get("song_ids", [])
    df = all_music_info_df.iloc[song_ids]
    df["song_id"] = song_ids
    songs = df.fillna("").to_dict(orient="records")
    return jsonify(songs)

@app.route("/last-modified")
def last_modified():
    timestamp = str(
        max(
            os.path.getmtime(root + "/" + file)
            for root, _, files in os.walk(".")
            for file in files
        )
    )
    return jsonify({"last_modified": timestamp})


@app.route("/get_fans_also_like", methods=["POST"])
def get_fans_also_like():
    data = request.get_json()
    return jsonify(data_all["fans_also_like"].get(data["sc_user"], []))

@app.route("/get_pinned_songs", methods=["POST"])
def get_pinned_songs():
    return jsonify({"pinnedSongs": user_data.get("pinned_song_ids", [])})

@app.route("/pin_song", methods=["POST"])
def pin_song():
    data = request.get_json()
    pinned_songs = user_data.get("pinned_song_ids", [])
    pinned_songs.append(int(data["song_id"]))
    user_data["pinned_song_ids"] = pinned_songs
    save_user_data(user_data)
    return jsonify({"pinnedSongs": user_data.get("pinned_song_ids", [])})

@app.route("/unpin_song", methods=["POST"])
def unpin_song():
    data = request.get_json()
    unping_song_id = int(data["song_id"])

    pinned_songs = user_data.get("pinned_song_ids", [])
    user_data["pinned_song_ids"] = [
        song_id for song_id in pinned_songs if song_id != unping_song_id
    ]
    save_user_data(user_data)
    return jsonify({"pinnedSongs": user_data.get("pinned_song_ids", [])})


if __name__ == "__main__":
    app.run(debug=True, port=5900)
