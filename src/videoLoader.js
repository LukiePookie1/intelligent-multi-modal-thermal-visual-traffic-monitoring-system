document.getElementById('videoUpload').addEventListener('change', function(event) {
    const file = event.target.files[0];
    const url = URL.createObjectURL(file);
    document.getElementById('videoPlayer').src = url;
  });
  