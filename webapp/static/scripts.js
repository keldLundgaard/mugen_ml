// Search 
function SearchRequest(){
    var formBody = JSON.stringify({
        query: document.getElementById("SearchBar").value,
    });
    console.log(formBody);
    fetch('/search', {
        method: 'POST',
        body: formBody,
    })
        .then(response => response.json())
        .then(data => console.log(data))
        .catch(error => console.error('Error:', error));
};

// function checkForUpdates() {
//     fetch('/last-modified')
//         .then(response => response.json())
//         .then(data => {
//             if (lastTimestamp && lastTimestamp !== data.last_modified) {
//                 // Reload the page if a change is detected
//                 location.reload(true);
//             }
//             lastTimestamp = data.last_modified;
//         })
//         .catch(error => console.error("Error checking for updates:", error));
// }
// // Poll every 5 seconds
// setInterval(checkForUpdates, 5000);

let playlist = [];
let currentSongIndex = 0;

function clearPlaylist(){
    playlist = [];
    displayPlaylist();
};

// let lastTimestamp = null;
function playSong(artist, title, album, trackNumber, path) {
        // Assuming 'path' is a direct link to the audio file.
        // Update the 'src' attribute of the audio element to play the new song
        var audioPlayer = document.getElementById('audioPlayer');
        audioPlayer.src = "stream"+path;
        audioPlayer.type = "audio/mpeg"
        audioPlayer.load(); // Reloads the audio element to apply the new source
        audioPlayer.play(); // Play the new song
        audioPlayer.onended = onSongFinish;
    }

function onSongFinish(){
    clickPlaylist((currentSongIndex + 1) % playlist.length);
};

function addSong(artist, title, album, trackNumber, filepath) {
    let new_playlist = playlist.length === 0
    playlist.push({ artist, title, album, trackNumber, filepath });
    displayPlaylist();
    if (new_playlist) { 
        clickPlaylist(0) } 
        else {
            document.getElementById("playlist_" + currentSongIndex).classList.add('playing');
        }
}

function clickPlaylist(index){
    document.getElementById("playlist_" + currentSongIndex).classList.remove('playing');
    let song = playlist[index];
    currentSongIndex = index
    playSong(song.artist, song.title, song.album, song.trackNumber, song.filepath);
    document.getElementById("playlist_"+ currentSongIndex).classList.add('playing');
}

function displayPlaylist() {
    const playlistElement = document.getElementById('playlist');
    playlistElement.innerHTML = ''; // Clear existing content

    playlist.forEach((song, index) => {
        playlistElement.innerHTML += `
        <div id="playlist_${index}"  onclick="clickPlaylist(${index})">
            <p>${song.artist} - ${song.title} (${song.album} - ${song.trackNumber}) </p>
        </div>
    `;
    });
}