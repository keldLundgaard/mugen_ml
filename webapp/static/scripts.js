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
        .catch(error => console.error('Error:', error));
}

function SearchRandom() {
    var searchBar = document.getElementById("SearchBar");
    searchBar.value = '*random*'; // Directly set the value
    
    // Manually dispatch an event if your application relies on it
    var event = new Event('input', { bubbles: true, cancelable: true });
    searchBar.dispatchEvent(event);
    SearchRequest();
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
let currentSongIndex = 0;

function clearPlaylist(){
    playlist = [];
    displayPlaylist();
};



// async function displaySearchResults(song_ids) {
//   const searchResultsUl = document.getElementById('searchResults');
//   searchResultsUl.innerHTML = '';

//    // Create table
//     try {
//         const songs = await get_song_ids_info(song_ids);
//         songs.forEach(song => {
//             const li = document.createElement('li');
//             const div = document.createElement('div');
//             div.className = 'song-link';
//             div.setAttribute(
//                 'onclick', 
//                 `addSong(
//                     '${song.artist}', 
//                     '${song.title}', 
//                     '${song.album}', 
//                     '${song["track number"]}', 
//                     '${song.genre}', 
//                 '${song.songPath}')`);
//             div.textContent = `${song.artist} - ${song.title}`;
//             li.appendChild(div);
//             searchResultsUl.appendChild(li);
//         });
//     } catch (error) {
//     console.error('Error:', error);
//   }
// }

async function displaySearchResults(song_ids) {
  const searchResultsDiv = document.getElementById('searchResults');
  searchResultsDiv.innerHTML = '';

  try {
    const songs = await get_song_ids_info(song_ids);

    if (songs) {
      // Create table
      const table = document.createElement('table');
      table.className = 'song-table';

      // Create header row
      const thead = document.createElement('thead');
      const headerRow = document.createElement('tr');
      const headers = [
        { text: 'Artist', key: 'artist' },
        { text: 'Title', key: 'title' },
        { text: 'Album', key: 'album' },
        { text: 'Track Number', key: 'track number' },
        { text: 'Genre', key: 'genre' }
      ];

      headers.forEach(header => {
        const th = document.createElement('th');
        th.textContent = header.text;
        th.dataset.key = header.key;
        th.style.cursor = 'pointer';
        th.addEventListener('click', () => sortTableByColumn(table, header.key));
        headerRow.appendChild(th);
      });

      thead.appendChild(headerRow);
      table.appendChild(thead);

      // Create body rows
      const tbody = document.createElement('tbody');

      songs.forEach(song => {
        const row = document.createElement('tr');

        const artistCell = document.createElement('td');
        artistCell.textContent = song.artist;
        row.appendChild(artistCell);

        const titleCell = document.createElement('td');
        titleCell.textContent = song.title;
        row.appendChild(titleCell);

        const albumCell = document.createElement('td');
        albumCell.textContent = song.album;
        row.appendChild(albumCell);

        const trackNumberCell = document.createElement('td');
        trackNumberCell.textContent = song["track number"];
        row.appendChild(trackNumberCell);

        const genreCell = document.createElement('td');
        genreCell.textContent = song.genre;
        row.appendChild(genreCell);

        const actionCell = document.createElement('td');
        const addButton = document.createElement('button');
        addButton.textContent = 'Add';
        addButton.setAttribute(
          'onclick',
          `addSong('${song.song_id}')`
        );
        actionCell.appendChild(addButton);
        row.appendChild(actionCell);

        tbody.appendChild(row);
      });

      table.appendChild(tbody);
      searchResultsDiv.appendChild(table);
    }
  } catch (error) {
    console.error('Error:', error);
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


function onSongFinish(){nextSong();};
function prevSong(){clickPlaylist((playlist.length + currentSongIndex - 1) % playlist.length);};
function nextSong(){clickPlaylist((currentSongIndex + 1) % playlist.length);};

if ('mediaSession' in navigator) {
  navigator.mediaSession.setActionHandler('nexttrack', function() {nextSong();});
  navigator.mediaSession.setActionHandler('previoustrack', function() {prevSong();});
}
// document.addEventListener('keydown', function(event) {if (event.key === "ArrowLeft") {prevSong();}});
// document.addEventListener('keydown', function(event) {if (event.key === "ArrowRight") {nextSong();}});

function addSong(song_id) {
    let new_playlist = playlist.length === 0
    playlist.push({song_id});
    displayPlaylist();
    if (new_playlist) { 
        clickPlaylist(0) } 
        else {
            document.getElementById("playlist_" + currentSongIndex).classList.add('playing');
        }
}

async function clickPlaylist(index){
    document.getElementById("playlist_" + currentSongIndex).classList.remove('playing');
    console.log(playlist);
    let song_id_obj = playlist[index];
    currentSongIndex = index
    try {
            const songs = await get_song_ids_info([song_id_obj.song_id]);
            if (songs.length > 0) {
                playSong(songs[0].paths);
            }
    } catch (error) {
        console.error('Error:', error);
    }
    document.getElementById("playlist_"+ currentSongIndex).classList.add('playing');
}

function playSong(path) {
        var audioPlayer = document.getElementById('audioPlayer');
        audioPlayer.src = "stream"+path;
        audioPlayer.type = "audio/mpeg"
        audioPlayer.load(); // Reloads the audio element to apply the new source
        audioPlayer.play(); // Play the new song
        audioPlayer.onended = onSongFinish;
    }


async function displayPlaylist() {
    const playlistElement = document.getElementById('playlist');
    playlistElement.innerHTML = ''; // Clear existing content

    try {
        for (let index = 0; index < playlist.length; index++) {
            const song_id_obj = playlist[index];
            const songs = await get_song_ids_info([song_id_obj.song_id]); // Assuming song_id_obj contains song_id

            songs.forEach(song => {
                playlistElement.innerHTML += `
                <div id="playlist_${index}" onclick="clickPlaylist(${index})">
                    <p>${song.artist} - ${song.title} </p>
                </div>`;
            });
        }
    } catch (error) {
        console.error('Error:', error);
    }
}