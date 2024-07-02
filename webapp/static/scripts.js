let searchTimeout;
document.getElementById("SearchBar").addEventListener("input", () => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(SearchRequest, 300);
});

function SearchRequest() {
    const query = document.getElementById("SearchBar").value;

    if (!query) {
        console.error('Error: Search query is empty.');
        return;
    }

    const formBody = JSON.stringify({ query });

    fetch('/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: formBody,
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok ' + response.statusText);
            }
            return response.json();
        })
        .then(data => displaySearchResults(data))
        .catch(error => {
          console.error('Error:', error);
          alert('An error occurred while searching. Please try again.');
        });
}

function SearchRandom() {
    SearchQuery('*random*')
}

function SearchQuery(query) {
    var searchBar = document.getElementById("SearchBar");
    searchBar.value = query; // Directly set the value
    
    // Manually dispatch an event if your application relies on it
    var event = new Event('input', { bubbles: true, cancelable: true });
    searchBar.dispatchEvent(event);
    SearchRequest();
}

function SearchFansAlsoLike(sc_user){
    var formBody = JSON.stringify({
        sc_user: sc_user,
    });
  fetch('/get_fans_also_like', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: formBody,
    })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok ' + response.statusText);
            }
            return response.json();
        })
        .then(data => displayFansAlsoLike(data))
        .catch(error => console.error);
} 

function displayFansAlsoLike(user_list){
  console.log(user_list)
  const fansAlsoLikeDiv = document.getElementById('fansAlsoLike');
  fansAlsoLikeDiv.innerHTML = '<h4>Fans also like <span onclick="removeFansAlsoLike()">❌</span></h4><ul>';
  user_list.forEach(sc_user => {
        fansAlsoLikeDiv.innerHTML += `
        <li onclick="userSearch('${sc_user}')">${sc_user}</li>`;
    });
  fansAlsoLikeDiv.innerHTML += '</ul>';
}

function removeFansAlsoLike() {
  const fansAlsoLikeDiv = document.getElementById('fansAlsoLike');
  fansAlsoLikeDiv.innerHTML =''
}


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
let currentSongIndex = null;

function userSearch(sc_user){
  SearchQuery("u: '"+ sc_user+"'");
  SearchFansAlsoLike(sc_user);
}

async function displaySearchResults(song_ids) {
  const searchResultsDiv = document.getElementById('searchResults');
  const albumTrackToggle = document.getElementById('albumTrackToggle');
  searchResultsDiv.innerHTML = '';

  try {
    const songs = await get_song_ids_info(song_ids);

    if (songs && songs.length > 0) {
      const table = document.createElement('table');
      table.className = 'song-table';

      const thead = document.createElement('thead');
      const headerRow = document.createElement('tr');
      const headers = [
        { text: 'User', key: 'user', class: 'search_results_user' },
        { text: 'Artist', key: 'artist', class: 'search_results_artist' },
        { text: 'Title', key: 'title', class: 'search_results_title' },
        { text: 'Genre', key: 'genre', class: 'search_results_genre' },
        { text: '', key: 'action', class: 'results_action' }
      ];

      if (albumTrackToggle.checked) {
        headers.splice(3, 0, 
          { text: 'Album', key: 'album', class: 'search_results_album' },
          { text: 'Track', key: 'track number', class: 'search_results_track' }
        );
      }

      headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header.text;
        th.dataset.key = header.key;
        th.className = header.class;
        th.style.cursor = 'pointer';
        th.addEventListener('click', () => sortTableByColumn(table, header.key));
        headerRow.appendChild(th);
      });

      thead.appendChild(headerRow);
      table.appendChild(thead);

      const tbody = document.createElement('tbody');

      songs.forEach(song => {
        const row = document.createElement('tr');

        headers.forEach(header => {
          const cell = document.createElement('td');
          cell.className = header.class;

          switch (header.key) {
            case 'user':
              cell.textContent = song.sc_user;
              cell.addEventListener('click', () => userSearch(song.sc_user));
              break;
            case 'artist':
              cell.textContent = song.artist;
              cell.addEventListener('click', () => SearchQuery(`a: '${song.artist}'`));
              break;
            case 'title':
              cell.textContent = song.title;
              break;
            case 'album':
              cell.textContent = song.album;
              cell.addEventListener('click', () => SearchQuery(`album: '${song.album}'`));
              break;
            case 'track number':
              cell.textContent = song["track number"];
              break;
            case 'genre':
              cell.textContent = song.genre;
              break;
            case 'action':
              const addIcon = document.createElement('span');
              addIcon.textContent = '▶️';
              addIcon.onclick = () => addSong(song.song_id);
              cell.appendChild(addIcon);
              break;
          }

          row.appendChild(cell);
        });

        tbody.appendChild(row);
      });

      table.appendChild(tbody);

      albumTrackToggle.addEventListener('change', () => {
        displaySearchResults(song_ids);
      });

      searchResultsDiv.appendChild(table);
    } else {
      searchResultsDiv.textContent = 'No results found.';
    }
  } catch (error) {
    console.error('Error:', error);
    searchResultsDiv.textContent = 'An error occurred while fetching results.';
  }
}

function sortTableByColumn(table, columnKey) {
  const tbody = table.querySelector('tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  
  const sortedRows = rows.sort((a, b) => {
    const aText = a.querySelector(`td:nth-child(${getColumnIndex(columnKey)})`).textContent.trim();
    const bText = b.querySelector(`td:nth-child(${getColumnIndex(columnKey)})`).textContent.trim();
    
    if (columnKey === 'track number') {
      return parseInt(aText) - parseInt(bText);
    } else {
      return aText.localeCompare(bText);
    }
  });

  // Remove all rows and append sorted rows
  tbody.innerHTML = '';
  sortedRows.forEach(row => tbody.appendChild(row));
}

function getColumnIndex(columnKey) {
  switch (columnKey) {
    case 'artist': return 1;
    case 'title': return 2;
    case 'album': return 3;
    case 'track number': return 4;
    case 'genre': return 5;
    default: return 1;
  }
}


if ('mediaSession' in navigator) {
  navigator.mediaSession.setActionHandler('nexttrack', function() {nextSong();});
  navigator.mediaSession.setActionHandler('previoustrack', function() {prevSong();});
}
// document.addEventListener('keydown', function(event) {if (event.key === "ArrowLeft") {prevSong();}});
// document.addEventListener('keydown', function(event) {if (event.key === "ArrowRight") {nextSong();}});


async function clickPlaylist(index){
  if (currentSongIndex !== null){
    document.getElementById("playlist_" + currentSongIndex).classList.remove('playing');
  }
  let song_id = playlist[index];
  currentSongIndex = index
  try {
          const songs = await get_song_ids_info([song_id]);
          if (songs.length > 0) {
              playSong(songs[0]);
          }
  } catch (error) {
      console.error('Error:', error);
  }
  document.getElementById("playlist_"+ currentSongIndex).classList.add('playing');
}

function onSongFinish(){nextSong();};
function prevSong(){clickPlaylist((playlist.length + currentSongIndex - 1) % playlist.length);};
function nextSong(){clickPlaylist((currentSongIndex + 1) % playlist.length);};

function playSong(song_obj) {
  var audioPlayer = document.getElementById('audioPlayer');
  audioPlayer.src = "stream" + song_obj.paths;
  audioPlayer.type = "audio/mpeg";
  audioPlayer.load();
  audioPlayer.play();
  audioPlayer.onended = onSongFinish;

  // Update song details
  const songDetailsDiv = document.getElementById('songDetails');
  songDetailsDiv.innerHTML = `
    <div class="song-title">${song_obj.title}</div>
    <div class="song-artist" onclick="SearchQuery('a: ${song_obj.artist}')">${song_obj.artist}</div>
    ${song_obj.album ? `<div class="song-album" onclick="SearchQuery('album: ${song_obj.album}')">Album: ${song_obj.album}${song_obj['track number'] ? ` (Track ${song_obj['track number']})` : ''}</div>` : ''}
    <div class="song-info">
      <span>Bitrate: ${song_obj['bitrate (kbps)']} kbps</span>
      <span>Genre: ${song_obj.genre}</span>
      <span>Year: ${song_obj['release year']}</span>
    </div>
    <div class="song-user" onclick="userSearch('${song_obj.sc_user}')">User: ${song_obj.sc_user}</div>
  `;
}

function clearPlaylist(){
    playlist = [];
    currentSongIndex=null;
    const playlistElement = document.getElementById('playlist');
    playlistElement.innerHTML = ""
};

async function PlaylistAddSong(song_id) {
  const playlistElement = document.getElementById('playlist');
  const index = playlist.length;
  try {
    const songs_to_add = await get_song_ids_info([song_id]);
    songs_to_add.forEach(song => {
        playlistElement.innerHTML += `
        <div class="playlist-song" id="playlist_${index-1}">
            <p onclick="clickPlaylist(${index-1})">${song.artist} - ${song.title}</p>
            <div class="song-actions">
                <span class="remove-icon" onclick="playlistRemoveSong(${index-1})">❌</span>
                <span class="pin-icon" onclick="pinSong('${song.song_id}')">📌</span>
            </div>
        </div>`
    });

    if (index === 1) {
      clickPlaylist(0);
    }
  
  } catch (error) {
      console.error('Error:', error);
  }
}

function addSong(song_id) {
    let new_playlist = playlist.length === 0
    playlist.push(song_id);
    PlaylistAddSong(song_id);
}

function playlistRemoveSong(index) {
  if (index === currentSongIndex) {
    currentSongIndex=null
  }
  playlist.splice(index, 1);
  
  const playlistElement = document.getElementById(`playlist_${index}`);
  if (playlistElement) {
    playlistElement.parentNode.removeChild(playlistElement);
  }

}

function get_song_ids_info(song_ids) {
    var formBody = JSON.stringify({
        song_ids: song_ids,
    });

    return fetch('/get_song_info', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: formBody,
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok ' + response.statusText);
        }
        return response.json();
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

// Pinned songs...

let pinnedSongs = [];

function pinSong(song_id) {
    if (!pinnedSongs.includes(song_id)) {
        fetch('/pin_song', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ song_id: song_id })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to pin song');
            }
            return response.json();
        })
        .then(data => {
            pinnedSongs = data.pinnedSongs;
            displayPinnedSongs();
        })
        .catch(error => console.error('Error pinning song:', error));
    }
}

        // <div class="playlist-song" id="playlist_${index-1}">
        //     <p onclick="clickPlaylist(${index-1})">${song.artist} - ${song.title}</p>
        //     <span class="remove-icon" onclick="playlistRemoveSong(${index-1})">❌</span>
        //     <span class="pin-icon" onclick="pinSong('${song.song_id}')">📌</span>
        // </div>`;

function unpinSong(song_id) {
    fetch('/unpin_song', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ song_id: song_id })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Failed to unpin song');
        }
        return response.json();
    })
    .then(data => {
        pinnedSongs = data.pinnedSongs;
        displayPinnedSongs();
    })
    .catch(error => console.error('Error unpinning song:', error));
}

async function displayPinnedSongs() {
    const pinnedSongsContainer = document.getElementById('pinnedSongs');
    pinnedSongsContainer.innerHTML = ''; // Clear existing content

    try {
        const songs = await get_song_ids_info(pinnedSongs); // Assuming song_id_obj contains song_id
        songs.forEach((song, index) => {
            pinnedSongsContainer.innerHTML += `
            <div class="pinned-song">
                <p onclick="addSong('${pinnedSongs[index]}')">${song.artist} - ${song.title}</p>
                <span onclick="unpinSong('${pinnedSongs[index]}')">❌</span>
            </div>`;
        });

    } catch (error) {
        console.error('Error:', error);
    }
}

async function fetchPinnedSongs() {
    fetch('/get_pinned_songs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    })
    .then(response => response.json())
    .then(data => {
        pinnedSongs = data.pinnedSongs;
        displayPinnedSongs();
    })
    .catch(error => console.error('Error fetching pinned songs:', error));
}

document.addEventListener('DOMContentLoaded', fetchPinnedSongs);
