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
from search import search

import pandas as pd
import os


app = Flask(__name__)
CORS(app)
# MUSIC_BASE_DIR = "/d16Tb/soundcloud_music/files/"
DATA_DIR = "/home/keld/projects/music_gen/sc_scrape/data/"
DATA_WEBAPP_DIR = "/n4Ta/sc_webapp_data/"

# archive_tracker_path = "/d16Tb/soundcloud_music/archive_tracker"
# out_path = "/d16Tb/soundcloud_music/files/"
# data_obj_path = f"{DATA_DIR}sc_data_v2.pckl"
logger = logging.getLogger(__name__)
app.logger.setLevel(logging.INFO)

info_df_path = DATA_WEBAPP_DIR + "df_with_mp3_info.tsv.gz"

app.logger.info(f"Loading music df...")
all_music_info_df = pd.read_csv(
    info_df_path,
    compression="gzip",
    sep="\t",
    low_memory=False,
)
app.logger.info(f"Done")


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
    print(music_path)
    decoded_filepath = unquote("/" + music_path)
    if 0:
        return Response(stream_audio(decoded_filepath), content_type="audio/mpeg")
    else:
        return send_file(
            decoded_filepath, as_attachment=True, download_name=decoded_filepath
        )

@app.route("/", methods=["GET", "POST"])
def index():
    songs = []
    query = None
    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query:
            idx_top_k = search(query)
            top_df = all_music_info_df.loc[idx_top_k]
            top_df["songPath"] = top_df["paths"].apply(lambda x: quote(x))
            songs = top_df.to_dict(orient="records")
    # Serve the HTML from the templates directory
    return render_template("index.html", songs=songs, query=query)

@app.route("/search", methods=["POST"])
def search_req():
    query = request.form.get("query", "").strip()
    if query:
        idx_top_k = search(query)
        top_df = all_music_info_df.loc[idx_top_k]
        top_df["songPath"] = top_df["paths"].apply(lambda x: quote(x))
        songs = top_df.to_dict(orient="records")
    else:
        songs = dict()
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


if __name__ == "__main__":
    app.run(debug=True, port=5900)
