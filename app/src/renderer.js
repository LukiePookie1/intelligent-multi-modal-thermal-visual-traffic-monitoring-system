// renderer.js
const videoUpload = document.getElementById('videoUpload');
const inputVideo = document.getElementById('inputVideo');
const processBtn = document.getElementById('processBtn');
const outputVideo = document.getElementById('outputVideo');
const outputContainer = document.getElementById('outputContainer');

videoUpload.addEventListener('change', function (event) {
  const file = event.target.files[0];
  const url = URL.createObjectURL(file);
  inputVideo.src = url;
});

processBtn.addEventListener('click', () => {
  const inputVideoPath = videoUpload.files[0].path;
  const outputVideoPath = window.electronAPI.join('output_video.mp4');
  console.log('Sending process-video event with paths:', inputVideoPath, outputVideoPath);
  window.electronAPI.send('process-video', inputVideoPath, outputVideoPath);
});

window.electronAPI.receive('processing-update', (message) => {
  outputContainer.innerHTML += `<p>${message}</p>`;
});

window.electronAPI.receive('processing-complete', (outputVideoPath) => {
  outputContainer.innerHTML += '<p>Video processing completed.</p>';
  outputVideo.src = outputVideoPath;
  outputVideo.style.display = 'block';
});