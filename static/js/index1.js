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
const btnVoicePayment = document.querySelectorAll('.btnVoicePayment');
const btnEnrollVoice = document.querySelectorAll('.btnEnrollVoice');
const btnVoice = document.querySelector('#btnVoice');
const btnEnroll = document.querySelector('#btnEnroll');
const infoForm = document.querySelector('#infoForm');
const bankName = document.querySelector("#bankName");
const bankAccount = document.querySelector("#bankAccount");
const username = document.querySelector("#username");
const amount = document.querySelector("#amount");


let recorder;
let audio;
let purposeText;
let token_id = '';
let is_valid = false;
let base64AudioMessage;

fetch("/get_voice_id")
.then(response => response.text())
.then(result => {
  var result = JSON.parse(result);
  token_id = result.token_id;
  is_valid = result.is_valid;
})
.catch(error => console.log(error));

// switch to payment modal
for (var i = 0; i < btnVoicePayment.length; i++){
  btnVoicePayment[i].addEventListener('click', async () => {
    purposeText = 'voice'
    btnVoice.style.display = 'block';
    btnEnroll.style.display = 'none';
    waitingS2T.style.display = 'none';
    btnEnroll.setAttribute('disabled', true);
    btnVoice.setAttribute('disabled', true);
    s2tResult.innerHTML = "";
    s2tResultEnroll.innerHTML = "";
    infoForm.style.display = 'none';
    bankName.value = '';
    bankAccount.value = '';
    username.value = '';
    amount.value = 0;
  });
}

// switch to enroll modal
for (var i = 0; i < btnEnrollVoice.length; i++){
  btnEnrollVoice[i].addEventListener('click', async () => {
    purposeText = 'enroll'
    btnEnroll.style.display = 'block';
    btnVoice.style.display = 'none';
    waitingS2T.style.display = 'none';
    btnEnroll.setAttribute('disabled', true);
    btnVoice.setAttribute('disabled', true);
    s2tResultEnroll.innerHTML = "";
    s2tResult.innerHTML = "";
    infoForm.style.display = 'none';
    bankName.value = '';
    bankAccount.value = '';
    username.value = '';
    amount.value = 0;
  });
}

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

btnVoice.addEventListener('click', async () => {
  var _bank_name = bankName.value;
  var _bank_account = bankAccount.value;
  var _username = username.value;
  var _amount = amount.value;
  var text = "";
  console.log(_bank_name);
  console.log(_bank_name.length);
  if (_bank_name.length == 0 || _bank_account.length == 0 || _username.length == 0 || _amount <= 0 ){
    text = '<div style="color:red"> Chuyển tiền thất bại. Vui lòng kiểm tra lại các thông tin </div>';
  }
  else{
    text = '<div style="color: blue"> Chuyển <b style="color:red">' + _amount + '</b>vnđ thành công đến <b>' + _username + '</b>, STK <b>' + _bank_account + '</b> tại ngân hàng <b>' + _bank_name + '</b></div>';
  }
  console.log(text);
  document.querySelector('#transferMoneyForm').innerHTML = text;
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
  console.log("before token: ", token_id);
  if (!is_valid){
    var requestOptions = {
        method: 'POST',
        body: "",
        redirect: 'follow'
      };

      token_id = await new Promise((resolve, reject) => {
        let t_id = '';
        fetch("https://tbot1.anhph.com/login", requestOptions)
                .then(response => response.text())
                .then(result => {
                  t_id = JSON.parse(JSON.parse(result).result).token;
                  resolve(t_id);
                  // update token
                  fetch("/save_token/" + t_id)
                })
                .catch(error => console.log('error', error));
      }, 5000)
      }
  console.log("after token: ", token_id);

  // encode voice to base64
  base64AudioMessage = await convertVoiceToB64();

  // Speech to text
  if (purposeText == 'voice'){

      // verify voice
      // get voice id
      fetch("/get_voice_id")
      .then(response => response.text())
      .then(result => {
          var voice_id = JSON.parse(result).voice_id;
          var myHeaders = new Headers();
          myHeaders.append("Content-Type", "application/json");
          console.log("voice id: ", voice_id);
          var raw = JSON.stringify({
            "base_64_voice": base64AudioMessage,
            "token": token_id,
            "voice_id": voice_id
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
              console.log(JSON.parse(result).result);
              var status = JSON.parse(JSON.parse(result).result).status;
              if (status == 2 || status == 4){
                waitingS2T.style.display="None";
              }
              if (status == 2) {
                s2tResult.innerHTML = '<div style="color: red">Cannot verify. Please speak more longer!</div>';
              }
              else if (status == 4){
                s2tResult.innerHTML = '<div style="color:red"> Your voice do not match!!! </div>';
              }
              else if (status == 3) {
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
                  // TODO: apply information extraction here
//                  s2tResult.innerHTML = JSON.parse(JSON.parse(result).result).transcript;
                  var transcript = JSON.parse(JSON.parse(result).result).transcript;
                  fetch("/extraction/" + transcript)
                    .then(result => result.text())
                    .then(result => {
                      result = JSON.parse(result);
                      waitingS2T.style.display="None";
                      infoForm.style.display = 'block';
                      // set attribute for element
                      bankName.value = result.bank_name;
                      bankAccount.value = result.bank_id;
                      username.value = result.name;
                      amount.value = 1000000;
                      btnVoice.removeAttribute("disabled");
                    })
                    .catch(error => console.log(error));
                 })
                .catch(error => console.log('error', error));
              }
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