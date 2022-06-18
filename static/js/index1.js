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

function convertVoiceToB64(){
  return new Promise((resolve, reject) => {
    let reader = new FileReader();
    let b64;
    reader.readAsDataURL(audio.audioBlob);
    reader.onload = () => {
      b64 = reader.result.split(',')[1];
      resolve(b64);
    };
  }, 5000)
}


const sleep = time => new Promise(resolve => setTimeout(resolve, time));

const recordButton = document.querySelector('#record');
const pauseButton = document.querySelector('#pause')
const stopButton = document.querySelector('#stop');
const playButton = document.querySelector('#play');
const s2tResult = document.querySelector('#s2tResult');
const waitingS2T = document.querySelector('#waitingS2T');
const s2tResultEnroll = document.querySelector('#s2tResultEnroll');
const waitingS2TEnroll = document.querySelector('#waitingS2TEnroll');
const btnVoicePayment = document.querySelector('.btnVoicePayment');
const btnEnrollVoice = document.querySelector('.btnEnrollVoice');
const btnVoice = document.querySelector('#btnVoice');
const btnEnroll = document.querySelector('#btnEnroll');


let recorder;
let audio;
let purposeText;
let token_id = '';
let base64AudioMessage;

// switch to payment modal
btnVoicePayment.addEventListener('click', async () => {
  purposeText = 'voice'
  btnVoice.style.display = 'block';
  btnEnroll.style.display = 'none';
  waitingS2T.style.display = 'none';
  btnEnroll.setAttribute('disabled', true);
  btnVoice.setAttribute('disabled', true);
  s2tResult.innerHTML = "";
  s2tResultEnroll.innerHTML = "";
});

// switch to enroll modal
btnEnrollVoice.addEventListener('click', async () => {
  purposeText = 'enroll'
  btnEnroll.style.display = 'block';
  btnVoice.style.display = 'none';
  waitingS2T.style.display = 'none';
  btnEnroll.setAttribute('disabled', true);
  btnVoice.setAttribute('disabled', true);
  s2tResultEnroll.innerHTML = "";
});

// enroll voice to get voice id
btnEnroll.addEventListener('click', async () => {
  waitingS2TEnroll.style.display = "block";
  s2tResultEnroll.innerHTML = "";
  var myHeaders = new Headers();
  myHeaders.append("Content-Type", "application/json");

  var raw = JSON.stringify({
    "base_64_voice": base64AudioMessage,
    "token": token_id
  });

  var requestOptions = {
    method: 'POST',
    headers: myHeaders,
    body: raw,
    redirect: 'follow'
  };

  fetch("https://tbot1.anhph.com/enroll", requestOptions)
    .then(response => response.text())
    .then(result => {
      console.log(result);
      console.log("style: ", waitingS2TEnroll.style.display);
      waitingS2TEnroll.style.display = "none";
      console.log("style: ", waitingS2TEnroll.style.display);
      var content = JSON.parse(JSON.parse(result).result).content;
      console.log(content);
      if (content == 'Success'){
          s2tResultEnroll.innerHTML = '<div style="color: blue"> Enroll voice successfully.</div>'
          var payload = JSON.parse(JSON.parse(result).result).payload;
          var voice_id = payload.id;
          requestOptions = {
            method: 'POST',
            headers: myHeaders,
            body: JSON.stringify({
                    "item": voice_id
                  }),
            redirect: 'follow'
          };
          fetch("/save/" + voice_id, requestOptions)
          .then(response => response.json())
              .then(result => {
                  console.log("result: ", result);
              })
              .catch(error => console.log('error', error));
      } else {
        s2tResultEnroll.innerHTML = '<div style="color: red"> Failed to enroll your voice. Please speak more longer or check your connection!!!</div>';
      }
    })
    .catch(error => console.log('error', error));
});


recordButton.addEventListener('click', async () => {
  recordButton.setAttribute('disabled', true);
  pauseButton.removeAttribute('disabled');
  stopButton.removeAttribute('disabled');
  playButton.setAttribute('disabled', true);
  btnEnroll.setAttribute('disabled', true);
  btnVoice.setAttribute('disabled', true);
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

  // get token id
    var raw = "";
    console.log("before token: ", token_id);
    if (token_id == ''){
      var requestOptions = {
          method: 'POST',
          body: raw,
          redirect: 'follow'
        };

        token_id = await new Promise((resolve, reject) => {
          let t_id = '';
          fetch("https://tbot1.anhph.com/login", requestOptions)
                  .then(response => response.text())
                  .then(result => {
                    t_id = JSON.parse(JSON.parse(result).result).token;
                    resolve(t_id);
                  })
                  .catch(error => console.log('error', error));
        }, 5000)
        }
    console.log("after token: ", token_id);
  // encode voice to base64
  base64AudioMessage = await convertVoiceToB64();

  // Speech to text
  if (purposeText == 'voice'){
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
          console.log("transcript: ", JSON.parse(JSON.parse(result).result).transcript);
//        waitingS2T.style.display="None";
//        s2tResult.innerHTML = JSON.parse(JSON.parse(result).result).transcript;
//        btnVoice.removeAttribute("disabled");

        // get voice id
        fetch("/get_voice_id")
        .then(response => response.text())
        .then(voice_id => {
            var myHeaders = new Headers();
            myHeaders.append("Content-Type", "application/json");
            console.log("token: ", token_id);
            console.log("voice: ", voice_id.substring(1, voice_id.length - 1));
            var raw = JSON.stringify({
              "base_64_voice": base64AudioMessage,
              "token": token_id,
              "voice_id": voice_id.substring(1, voice_id.length - 1)
            });

            var requestOptions = {
              method: 'POST',
              headers: myHeaders,
              body: raw,
              redirect: 'follow'
            };
            fetch("https://tbot1.anhph.com/verify", requestOptions)
              .then(response => response.text())
              .then(result => {
                waitingS2T.style.display="None";
                console.log(result);
                var status = JSON.parse(JSON.parse(result).result).status;
                if (status == 2) {
                  s2tResult.innerHTML = 'Short speech'
                }
                else if (status == 3 || status == 4) {
                  s2tResult.innerHTML = JSON.parse(JSON.parse(result).result).payload.score;
                }
                btnVoice.removeAttribute("disabled");
              })
              .catch(error => console.log('error', error));
        })
        .catch(error => console.log('error', error));
    })
    .catch(error => console.log('error', error));
  }else{
    waitingS2T.style.display="None";
    s2tResult.innerHTML = '<div style="color: blue"> Ready to enroll your voice. Please click <i>Enroll</i> button to finish.</div>';
    btnEnroll.removeAttribute("disabled");
  }
});

playButton.addEventListener('click', () => {
  audio.play();
});