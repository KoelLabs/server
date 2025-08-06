import { FeedbackGiver } from './FeedbackGiver.js';

// prettier-ignore
const target_by_words = [
    ['you', [['j', 0.24160443037974685, 0.2617381329113924], ['u', 0.2617381329113924, 0.281871835443038]]],
    ['gotta', [['ɡ', 0.34227294303797473, 0.36240664556962027], ['ɑ', 0.38254034810126586, 0.40267405063291145], ['t', 0.4429414556962025, 0.4630751582278481], ['ʌ', 0.4832088607594937, 0.5033425632911394]]],
    ['stay', [['s', 0.6845458860759495, 0.7046795886075949], ['t', 0.7650806962025317, 0.7852143987341773], ['eɪ', 0.8053481012658229, 0.8254818037974684]]],
    ['alert', [['æ', 1.0872199367088609, 1.1073536392405066], ['l', 1.369091772151899, 1.3892254746835444], ['ɜ˞', 1.4093591772151899, 1.4294928797468356], ['t', 1.6106962025316458, 1.630829905063291]]],
    ['all', [['ɔ', 2.0536376582278484, 2.073771360759494], ['l', 2.1945735759493674, 2.214707278481013]]],
    ['the', [['ð', 2.27510838607595, 2.295242088607595], ['ʌ', 2.295242088607595, 2.3153757911392407]]],
    ['time', [['t', 2.416044303797469, 2.436178006329114], ['aɪ', 2.4764454113924055, 2.4965791139240507], ['m', 2.7985846518987345, 2.8187183544303798]]],
];

// update word colors everytime the transcript updates
function color_word(el) {
  const score = parseFloat(el.dataset.pscore);
  const isCorrect = el.dataset.wordCorrect === 'true';
  if (isCorrect || score > 0.8) {
    el.style.backgroundColor = `hsl(${score * 120}, 100%, 50%)`;
  } else {
    el.style.backgroundColor = `red`;
  }
}

async function on_transcription(transcription) {
  document.getElementById('transcription').textContent = transcription.map(p => p[0]).join('');
  const [scoredWords, overall] = await feedbackGiver.getCER();
  const wordElements = document.querySelectorAll('#scored_words span');
  for (let i = 0; i < scoredWords.length; i++) {
    const wordscore = scoredWords[i];
    const score = wordscore.at(-1);
    wordElements[i].dataset.pscore = score;
    if (i < feedbackGiver.next_word_ix) {
      color_word(wordElements[i]);
    }
  }
  document.getElementById('score').textContent = Math.min(Math.round(100 * overall + 3), 100);
}

// show the current word and automatically stop recording when the last word is spoken
function on_word_spoken(words, are_words_correct, next_word_ix, percentage_correct, is_done) {
  const wordElements = document.querySelectorAll('#scored_words span');
  for (let i = 0; i < next_word_ix; i++) {
    const el = wordElements[i];
    el.dataset.wordCorrect = are_words_correct[i];
    el.style.border = 'none';
    color_word(el);
  }
  if (is_done) {
    setTimeout(() => feedbackGiver.stop(), 1000);
    document.getElementById('start').disabled = false;
    document.getElementById('stop').disabled = true;
  } else {
    const el = wordElements[next_word_ix];
    el.style.border = '3px solid red';
  }
}

const feedbackGiver = new FeedbackGiver(target_by_words, on_transcription, on_word_spoken);

// add the words to the scored words element
for (const word of feedbackGiver.words) {
  document.getElementById('scored_words').innerHTML +=
    `<span data-pscore="0" data-word-correct="false">${word}</span> `;
}

// start recording
document.getElementById('start').addEventListener('click', () => {
  const wordElements = document.querySelectorAll('#scored_words span');
  for (const wordEl of wordElements) {
    wordEl.style.backgroundColor = 'none';
    wordEl.style.border = 'none';
    wordEl.dataset.pscore = 0;
    wordEl.dataset.wordCorrect = false;
  }
  document.getElementById('start').disabled = true;
  document.getElementById('stop').disabled = false;
  feedbackGiver.start();
});

// manually stop recording
document.getElementById('stop').addEventListener('click', () => {
  feedbackGiver.stop();
  document.getElementById('start').disabled = false;
  document.getElementById('stop').disabled = true;
});

// allow listening back to the full user recording
document.getElementById('play-button').addEventListener('click', async () => {
  const button = document.getElementById('play-button');
  button.innerText = 'Playing...';
  button.disabled = true;

  await feedbackGiver.playUserAudio();
  button.innerText = 'Play';
  button.disabled = false;
});

// display feedback
window.playTimestamps = async (button, startTimestamp, endTimestamp) => {
  const originalText = button.textContent;
  button.textContent = 'Playing...';
  button.disabled = true;

  await feedbackGiver.playUserAudio(startTimestamp, endTimestamp);
  button.textContent = originalText;
  button.disabled = false;
};
document.getElementById('feedback-button').addEventListener('click', async () => {
  const feedback = await feedbackGiver.getFeedback();
  const mistakes = feedback.topk_mistakes_by_target;
  const wordTimestamps = [
    feedback.spoken_word_timestamps[0],
    ...feedback.spoken_word_timestamps,
    feedback.spoken_word_timestamps.at(-1),
  ];

  document.getElementById('feedback-area').innerHTML = /* html */ `
    <h3>Top Mistakes</h3>
    <ul>
      ${mistakes
        .map(
          mistake => /* html */ `
        <li>
          ${mistake.target} is often mispronounced as ${mistake.speech.join(', ')}, e.g., in ${mistake.words.join(', ')}<br><br>
          To make the right sound: ${mistake.target_description.explanation}<br><br>
          Clip out just the phoneme:<br>
          ${mistake.occurences_by_word.map(([word, occurences]) => occurences.map(([, speech], ix) => /* html */ `<button onclick="playTimestamps(this, ${speech[1] - 0.2}, ${speech[2] + 0.2})">${word} error #${ix} (${speech})</button>`).join('')).join('')}<br><br>
          Clip out the entire 3 word phrase:<br>
          ${mistake.occurences_by_word.map(([word, occurences], ix) => occurences.map(([, speech], num) => /* html */ `<button onclick="playTimestamps(this, ${wordTimestamps[ix][1]}, ${wordTimestamps[ix + 2][2]})">${word} error #${num} (${speech[0]})</button>`).join('')).join('')}<br><br>
        </li>
      `,
        )
        .join('')}
    </ul>
  `;
});
