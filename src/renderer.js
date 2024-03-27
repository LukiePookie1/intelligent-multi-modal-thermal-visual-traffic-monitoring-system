// renderer.js
const { ipcRenderer } = require('electron');

const videoUpload = document.getElementById('videoUpload');
const videoPlayer = document.getElementById('videoPlayer');
const processBtn = document.getElementById('processBtn');
const outputContainer = document.getElementById('outputContainer');

videoUpload.addEventListener('change', function(event) {
  const file = event.target.files[0];
  const url = URL.createObjectURL(file);
  videoPlayer.src = url;
});

processBtn.addEventListener('click', () => {
  const inputVideoPath = videoUpload.files[0].path;
  ipcRenderer.send('process-video', inputVideoPath);
});

ipcRenderer.on('processing-update', (event, message) => {
  outputContainer.innerHTML += `<p>${message}</p>`;
});

ipcRenderer.on('processing-complete', () => {
  outputContainer.innerHTML += '<p>Video processing completed.</p>';
});