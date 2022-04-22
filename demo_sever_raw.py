import io
import argparse
import torch
import numpy as np
import time
import hparams as hp
import falcon
from pypinyin import slug, Style
from wsgiref import simple_server

from models.fatchord_version import WaveRNN
from models.tacotron import Tacotron

from utils.text.symbols import symbols
from utils.paths import Paths
from utils.text import text_to_sequence
from utils.display import save_attention, simple_table
import librosa


html_body = '''<html><title>Tacotron_WaveRNN_demo</title><meta charset='utf-8'>
<style>
body {padding: 16px; font-family: sans-serif; font-size: 14px; color: #444}
input {font-size: 14px; padding: 8px 12px; outline: none; border: 1px solid #ddd}
input:focus {box-shadow: 0 1px 2px rgba(0,0,0,.15)}
p {padding: 12px}
button {background: #28d; padding: 9px 14px; margin-left: 8px; border: none; outline: none;
        color: #fff; font-size: 14px; border-radius: 4px; cursor: pointer;}
button:hover {box-shadow: 0 1px 2px rgba(0,0,0,.15); opacity: 0.9;}
button:active {background: #29f;}
button[disabled] {opacity: 0.4; cursor: default}
</style>
<body>
<form>
  <input id="text" type="text" size="40" placeholder="请输入想要合成音频的文字">
  <button id="button" name="synthesize">合成</button>
</form>
<p id="message"></p>
<audio id="audio" controls autoplay hidden></audio>
<script>
function q(selector) {return document.querySelector(selector)}
q('#text').focus()
q('#button').addEventListener('click', function(e) {
  text = q('#text').value.trim()
  if (text) {
    q('#message').textContent = '正在合成中...请稍等'
    q('#button').disabled = true
    q('#audio').hidden = true
    synthesize(text)
  }
  e.preventDefault()
  return false
})
function synthesize(text) {
  fetch('/synthesize?text=' + encodeURIComponent(text), {cache: 'no-cache'})
    .then(function(res) {
      if (!res.ok) throw Error(res.statusText)
      return res.blob()
    }).then(function(blob) {
      q('#message').textContent = ''
      q('#button').disabled = false
      q('#audio').src = URL.createObjectURL(blob)
      q('#audio').hidden = false
    }).catch(function(err) {
      q('#message').textContent = 'Error: ' + err.message
      q('#button').disabled = false
    })
}
</script></body></html>
'''


def load_wavernn(path):
    device = torch.device('cuda')
    print('Initialising WaveRNN Model...')

    # Instantiate WaveRNN Model
    voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                        fc_dims=hp.voc_fc_dims,
                        bits=hp.bits,
                        pad=hp.voc_pad,
                        upsample_factors=hp.voc_upsample_factors,
                        feat_dims=hp.num_mels,
                        compute_dims=hp.voc_compute_dims,
                        res_out_dims=hp.voc_res_out_dims,
                        res_blocks=hp.voc_res_blocks,
                        hop_length=hp.hop_length,
                        sample_rate=hp.sample_rate,
                        mode=hp.voc_mode).to(device)

    voc_model.restore(path)  # 520k
    return voc_model


def load_tacotron(path):
    print('Initialising Tacotron Model...')
    # Instantiate Tacotron Model
    device = torch.device('cuda')
    tts_model = Tacotron(embed_dims=hp.tts_embed_dims,
                         num_chars=len(symbols),
                         encoder_dims=hp.tts_encoder_dims,
                         decoder_dims=hp.tts_decoder_dims,
                         n_mels=hp.num_mels,
                         fft_bins=hp.num_mels,
                         postnet_dims=hp.tts_postnet_dims,
                         encoder_K=hp.tts_encoder_K,
                         lstm_dims=hp.tts_lstm_dims,
                         postnet_K=hp.tts_postnet_K,
                         num_highways=hp.tts_num_highways,
                         dropout=hp.tts_dropout,
                         stop_threshold=hp.tts_stop_threshold).to(device)

    tts_model.restore(path)  # 200k
    return tts_model


def synthesizer(input_text, tts_model, voc_model, batched, target, overlap):

    start_time = time.time()

    # process input pinyin
    inputs = text_to_sequence(input_text.strip(), hp.tts_cleaner_names)

    # generate mel spectrum
    _, m, attention = tts_model.generate(inputs)
    m = (m + 4) / 8
    np.clip(m, 0, 1, out=m)
    m = torch.tensor(m).unsqueeze(0)
    # generate wav
    out = io.BytesIO()
    voc_model.generate(m, out, batched, target, overlap, True)

    cost_time = time.time() - start_time
    print('synthesizing cost {} sec'.format(cost_time))

    return out.getvalue()


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(description='TTS Generator')
    parser.add_argument('--input_text', '-i', type=str, help='[string] Type in something here and TTS will generate it!')
    parser.add_argument('--batched', '-b', dest='batched', action='store_true', help='Fast Batched Generation')
    parser.add_argument('--target', '-t', type=int, help='[int] number of samples in each batch index')
    parser.add_argument('--overlap', '-o', type=int, help='[int] number of crossover samples')
    parser.add_argument('--port', type=int, default=9999)
    parser.set_defaults(input_text=None)
    parser.set_defaults(batched=hp.voc_gen_batched)
    parser.set_defaults(target=hp.voc_target)
    parser.set_defaults(overlap=hp.voc_overlap)

    args = parser.parse_args()

    input_text = args.input_text
    batched = args.batched
    target = args.target
    overlap = args.overlap

    # load Wavernn
    path = '/mnt/WaveRNN/checkpoints/biaobei_raw.wavernn2/latest_weights.pyt'
    voc_model = load_wavernn(path)

    # load tacotron
    path = '/mnt/WaveRNN/checkpoints/biaobei_lsa_smooth_attention.tacotron/latest_weights.pyt'
    tts_model = load_tacotron(path)

    # test server
    # path_wav = '/Users/lihaoqi/Desktop/WaveRNN/train_files/1_wavernn_batched_tac200k_wr406.wav'
    # f = open(path_wav,'rb')
    # out_data = f.read()

    class UIResource:
        def on_get(self, req, res):
          res.content_type = 'text/html'
          res.body = html_body

    class SynthesisResource:
        def on_get(self, req, res):
            if not req.params.get('text'):
                raise falcon.HTTPBadRequest()
            res.data = synthesizer(slug(req.params.get('text'), style=Style.TONE3, separator=' '),
                                   tts_model=tts_model,
                                   voc_model=voc_model,
                                   batched=batched,
                                   target=target, overlap=overlap)
            #res.data = out_data
            res.content_type = 'audio/wav'

    api = falcon.API()
    api.add_route('/synthesize', SynthesisResource())
    api.add_route('/', UIResource())

    print('Serving on port %d' % args.port)
    simple_server.make_server('0.0.0.0', args.port, api).serve_forever()
