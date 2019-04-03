

const HOSTED_URLS = {
  model:
      'model_js/model.json',
  metadata:
      'model_js/metadata.json'
};

const examples = {
  'example1':
      'light Blue',
  'example2':
      'green',
  'example3':
      'tensorflow orange'
};

function status(statusText) {
  console.log(statusText);
  document.getElementById('status').textContent = statusText;
}

function showMetadata(metadataJSON) {
  document.getElementById('vocabularySize').textContent =
      metadataJSON['vocabulary_size'];
  document.getElementById('maxLen').textContent =
      metadataJSON['max_len'];
}

function settextField(text, predict) {
  const textField = document.getElementById('text-entry');
  textField.value = text;
  doPredict(predict);
}

function setPredictFunction(predict) {
  const textField = document.getElementById('text-entry');
  textField.addEventListener('input', () => doPredict(predict));
}

function disableLoadModelButtons() {
  document.getElementById('load-model').style.display = 'none';
}

function doPredict(predict) {
  const textField = document.getElementById('text-entry');
  const result = predict(textField.value);
  score_string = "RGB: ";
  var RG = ["R","G","B"];
  var rgb_int = []
  console.log(result)
  for (var x in result.score) {
    score_string += RG[x] + " ->  " + Math.trunc(result.score[x]*255) + ", "
    rgb_int.push(Math.trunc(result.score[x]*255))
  }
  console.log(rgb_int[0],rgb_int[1], rgb_int[2]);

  rgb_string = "rgb("+rgb_int[0]+','+rgb_int[1]+','+rgb_int[2]+")"
  const mycolor = document.getElementById('text-color');
  mycolor.style.backgroundColor = rgb_string;
  console.log(mycolor.style.backgroundColor)

  status(
      score_string + ' elapsed: ' + result.elapsed.toFixed(3) + ' ms)');

}

function prepUI(predict) {
  setPredictFunction(predict);
  const testExampleSelect = document.getElementById('example-select');
  testExampleSelect.addEventListener('change', () => {
    settextField(examples[testExampleSelect.value], predict);
  });
  settextField(examples['example1'], predict);
}

async function urlExists(url) {
  status('Testing url ' + url);
  try {
    const response = await fetch(url, {method: 'HEAD'});
    return response.ok;
  } catch (err) {
    return false;
  }
}

async function loadHostedPretrainedModel(url) {
  status('Loading pretrained model from ' + url);
  try {
    console.log(url)
    const model = await tf.loadLayersModel(url);
    status('Done loading pretrained model.');
    disableLoadModelButtons();
    return model;
  } catch (err) {
    console.error(err);
    status('Loading pretrained model failed.');
  }
}

async function loadHostedMetadata(url) {
  status('Loading metadata from ' + url);
  try {
    const metadataJson = await fetch(url);
    const metadata = await metadataJson.json();
    status('Done loading metadata.');
    return metadata;
  } catch (err) {
    console.error(err);
    status('Loading metadata failed.');
  }
}

class Classifier {

  async init(urls) {
    this.urls = urls;
    this.model = await loadHostedPretrainedModel(urls.model);
    await this.loadMetadata();
    return this;
  }

  async loadMetadata() {
    const metadata =
        await loadHostedMetadata(this.urls.metadata);
    showMetadata(metadata);
    this.maxLen = metadata['max_len'];
    console.log('maxLen = ' + this.maxLen);
    this.wordIndex = metadata['word_index']
  }

  predict(text) {
    // Convert to lower case and remove all punctuations.
    const inputText =
        text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '');
    console.log(inputText);
    console.log(inputText.length)
    // Look up word indices.
    //const inputBuffer = tf.buffer([ this.maxLen,1], 'float32');
    const inputBuffer = tf.buffer([1, this.maxLen], 'float32');
    for (let i = 0; i < inputText.length; ++i) {
      const word = inputText[i];
      inputBuffer.set(this.wordIndex[word], 0, this.maxLen-inputText.length+i);
      console.log(i, this.maxLen-inputText.length+i,word, this.wordIndex[word], inputBuffer);
    }
    const input = inputBuffer.toTensor();
    console.log(input);

    status('Running inference');
    const beginMs = performance.now();
    console.log(input)
    const predictOut = this.model.predict(input);  //updated to take the list
    console.log(predictOut.dataSync());
    const score = predictOut.dataSync();//[0];
    predictOut.dispose();
    const endMs = performance.now();

    return {score: score, elapsed: (endMs - beginMs)};
  }
};

async function setup() {
  if (await urlExists(HOSTED_URLS.model)) {
    status('Model available: ' + HOSTED_URLS.model);
    const button = document.getElementById('load-model');

    const mycolor = document.getElementById('text-color');
    mycolor.style.backgroundColor = "white";
    // document.getElementById('text-color').style.color="rgb(255,0,0)";
    console.log(document.getElementById('text-color').style.backgroundColor);

    button.addEventListener('click', async () => {
      const predictor = await new Classifier().init(HOSTED_URLS);
      prepUI(x => predictor.predict(x));
    });
    button.style.display = 'inline-block';
  }

  status('Standing by.');
}

setup();

