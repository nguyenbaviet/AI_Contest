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
const s2tResult = document.querySelector('#s2tResult');
const waitingS2T = document.querySelector('#waitingS2T');

let recorder;
let audio;

recordButton.addEventListener('click', async () => {
  recordButton.setAttribute('disabled', true);
  pauseButton.removeAttribute('disabled');
  stopButton.removeAttribute('disabled');
  playButton.setAttribute('disabled', true);
  waitingS2T.style.display = 'None';
  if (!recorder) {
    recorder = await recordAudio();
  }
  s2tResult.innerHTML = "";
  recorder.start();
});

pauseButton.addEventListener('click', async () => {
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
  pauseButton.textContent = "Pause";
  audio = await recorder.stop();

  waitingS2T.style.display='block';
  // display speech to text
  const reader = new FileReader();
   reader.readAsDataURL(audio.audioBlob);
   reader.onload = () => {
     const base64AudioMessage = reader.result.split(',')[1];
     const myHeaders = new Headers();
      myHeaders.append("Content-Type", "application/json");

      const raw = JSON.stringify({
        "base_64_voice": base64AudioMessage
      });

      const requestOptions = {
        method: 'POST',
        headers: myHeaders,
        body: raw,
        redirect: 'follow'
      };
     fetch("https://tbot1.anhph.com/s2t", requestOptions)
    .then(response => response.text())
    .then(result => {
        waitingS2T.style.display="None";
        s2tResult.innerHTML = JSON.parse(JSON.parse(result).result).transcript;
    })
    .catch(error => console.log('error', error));
    };
});

playButton.addEventListener('click', () => {
  audio.play();
});