const recordAudio = () =>
  new Promise(async resolve => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    let audioChunks = [];

    mediaRecorder.addEventListener('dataavailable', event => {
      audioChunks.push(event.data);
    });

    const start = () => {
      audioChunks = [];
      mediaRecorder.start();
    };

    const pause = () => {
      mediaRecorder.pause();
    }

    const resume = () => {
    mediaRecorder.resume();
    }

    const stop = () =>
      new Promise(resolve => {
        mediaRecorder.addEventListener('stop', () => {
          const audioBlob = new Blob(audioChunks, { type: 'audio/mpeg' });
          const audioUrl = URL.createObjectURL(audioBlob);
          const audio = new Audio(audioUrl);
          const play = () => audio.play();http://0.0.0.0/
          resolve({ audioChunks, audioBlob, audioUrl, play });
        });

        mediaRecorder.stop();
      });

    resolve({ start, stop, pause, resume, mediaRecorder});
  });

const sleep = time => new Promise(resolve => setTimeout(resolve, time));

const recordButton = document.querySelector('#record');
const pauseButton = document.querySelector('#pause')
const stopButton = document.querySelector('#stop');
const playButton = document.querySelector('#play');
const saveButton = document.querySelector('#save');
const savedAudioMessagesContainer = document.querySelector('#saved-audio-messages');
const downloadMe = document.querySelector('#downloadMe');

let recorder;
let audio;

recordButton.addEventListener('click', async () => {
  recordButton.setAttribute('disabled', true);
  pauseButton.removeAttribute('disabled');
  stopButton.removeAttribute('disabled');
  playButton.setAttribute('disabled', true);
  saveButton.setAttribute('disabled', true);
  downloadMe.style.display = 'None';
  if (!recorder) {
    recorder = await recordAudio();
  }
  recorder.start();
});

pauseButton.addEventListener('click', async () => {
  console.log(recorder.mediaRecorder.state);
  stopButton.removeAttribute('disabled');
  if(recorder.mediaRecorder.state === "recording") {
        recorder.pause();
        pauseButton.textContent = "Resume";
        // recording paused
    } else if(recorder.mediaRecorder.state === "paused") {
      recorder.resume();
      pauseButton.textContent = "Pause";
      // resume recording
    }
});

stopButton.addEventListener('click', async () => {
  recordButton.removeAttribute('disabled');
  pauseButton.setAttribute('disabled', true);
  stopButton.setAttribute('disabled', true);
  playButton.removeAttribute('disabled');
  saveButton.removeAttribute('disabled');
  pauseButton.textContent = "Pause";
  audio = await recorder.stop();
});

playButton.addEventListener('click', () => {
  console.log(audio.duration);
  audio.play();
});

saveButton.addEventListener('click', () => {
  downloadMe.href = audio.audioUrl;
  downloadMe.download = 'v_test.wav';
  downloadMe.style.display="block";
});

//saveButton.addEventListener('click', () => {
//  const reader = new FileReader();
//  reader.readAsDataURL(audio.audioBlob);
//  reader.onload = () => {
//    const base64AudioMessage = reader.result.split(',')[1];
//
//    fetch('/messages', {
//      method: 'POST',
//      headers: { 'Content-Type': 'application/json' },
//      body: JSON.stringify({ message: base64AudioMessage })
//    }).then(res => {
//      if (res.status === 201) {
//        return populateAudioMessages();
//      }
//      console.log('Invalid status saving audio message: ' + res.status);
//    });
//  };
//});

const populateAudioMessages = () => {
  return fetch('/messages').then(res => {
    if (res.status === 200) {
      return res.json().then(json => {
        json.messageFilenames.forEach(filename => {
          let audioElement = document.querySelector(`[data-audio-filename="${filename}"]`);
          if (!audioElement) {
            audioElement = document.createElement('audio');
            audioElement.src = `/messages/${filename}`;
            audioElement.setAttribute('data-audio-filename', filename);
            audioElement.setAttribute('controls', true);
            savedAudioMessagesContainer.appendChild(audioElement);
          }
        });
      });
    }
    console.log('Invalid status getting messages: ' + res.status);
  });
};

<!--      populateAudioMessages();-->