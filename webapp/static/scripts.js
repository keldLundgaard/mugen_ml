function SearchRequest(){
    var formBody = JSON.stringify({
        query: document.getElementById("SearchBar").value,
    });
    // console.log("Search");
    // console.log(formBody);
    // console.log("done")
    fetch('/search', {
        method: 'POST',
        headers: {'Content-Type': 'application/json',},
        body: formBody,
    })
        .then(response => response.json())
        .then(data => displaySearchResults(data))
        .catch(error => console.error('Error:', error));
};
let lastTimestamp = null;
function checkForUpdates() {
    fetch('/last-modified')
        .then(response => response.json())
        .then(data => {
            if (lastTimestamp && lastTimestamp !== data.last_modified) {
                // Reload the page if a change is detected
                location.reload(true);
            }
            lastTimestamp = data.last_modified;
        })
        .catch(error => console.error("Error checking for updates:", error));
}
setInterval(checkForUpdates, 2000); // How often to pull in ms 

let playlist = [];
let currentSongIndex = 0;

function clearPlaylist(){
    playlist = [];
    displayPlaylist();
};

function displaySearchResults(songs) {
  const searchResultsUl = document.getElementById('searchResults');
  searchResultsUl.innerHTML = '';

  songs.forEach(song => {
    const li = document.createElement('li');
    const div = document.createElement('div');
    div.className = 'song-link';
    div.setAttribute('onclick', `addSong('${song.artist}', '${song.title}', '${song.album}', '${song["track number"]}', '${song.songPath}')`);
    div.textContent = `${song.artist} - ${song.title}`;
    li.appendChild(div);
    searchResultsUl.appendChild(li);
  });
}


function playSong(artist, title, album, trackNumber, path) {
        var audioPlayer = document.getElementById('audioPlayer');
        audioPlayer.src = "stream"+path;
        audioPlayer.type = "audio/mpeg"
        audioPlayer.load(); // Reloads the audio element to apply the new source
        audioPlayer.play(); // Play the new song
        audioPlayer.onended = onSongFinish;
    }

function onSongFinish(){nextSong();};
function prevSong(){clickPlaylist((playlist.length + currentSongIndex - 1) % playlist.length);};
function nextSong(){clickPlaylist((currentSongIndex + 1) % playlist.length);};

if ('mediaSession' in navigator) {
  navigator.mediaSession.setActionHandler('nexttrack', function() {nextSong();});
  navigator.mediaSession.setActionHandler('previoustrack', function() {prevSong();});
}
document.addEventListener('keydown', function(event) {if (event.key === "ArrowLeft") {prevSong();}});
document.addEventListener('keydown', function(event) {if (event.key === "ArrowRight") {nextSong();}});

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