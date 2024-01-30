let playlist = [];
let currentSongIndex = 0;

function addSong(artist, title, album, trackNumber, filepath) {
    playlist.push({ artist, title, album, trackNumber, filepath });
    displayPlaylist();
}

function displayPlaylist() {
    const playlistElement = document.getElementById('playlist');
    playlistElement.innerHTML = ''; // Clear existing content

    playlist.forEach((song, index) => {
        playlistElement.innerHTML += `
            <div id="song_${index}">
                <strong>${index + 1}. ${song.title}</strong>
                <p>Artist: ${song.artist}</p>
                <p>Album: ${song.album} (Track ${song.trackNumber})</p>
                <audio controls ${index === currentSongIndex ? 'autoplay' : ''}>
                    <source src="${song.filepath}" type="audio/mpeg">
                    Your browser does not support the audio element.
                </audio>
            </div>
        `;
    });

    // Add event listeners for each song
    playlist.forEach((song, index) => {
        const audioElement = document.querySelector(`#song_${index} audio`);
        audioElement.onended = () => playNextSong(index);
    });
}

function playNextSong(index) {
    if (index + 1 < playlist.length) {
        currentSongIndex = index + 1;
        displayPlaylist();
    } else {
        // Optional: Loop to start
        currentSongIndex = 0;
        displayPlaylist();
    }
}

// Example Usage
// addSong('Artist Name', 'Song Title', 'Album Title', 1, 'path/to/song.mp3');
