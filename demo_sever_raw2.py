import torch
from models.fatchord_version import WaveRNN
from utils import hparams as hp
from utils.text.symbols import symbols
from models.tacotron import Tacotron
import argparse
from utils.text import text_to_sequence
import numpy as np
from tts_front_end import text_to_pinyin
import falcon
import time,io
from wsgiref import simple_server
from pypinyin import slug, Style


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

if __name__ == "__main__":

    # Parse Arguments
    parser = argparse.ArgumentParser(description='TTS Generator')
    parser.add_argument('--input_text', '-i', type=str, help='[string] Type in something here and TTS will generate it!')
    parser.add_argument('--tts_weights', type=str, help='[string/path] Load in different Tacotron weights')
    parser.add_argument('--save_attention', '-a', dest='save_attn', action='store_true', help='Save Attention Plots')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')

    parser.set_defaults(input_text=None)
    parser.set_defaults(weights_path=None)

    # name of subcommand goes to args.vocoder
    subparsers = parser.add_subparsers(required=True, dest='vocoder')

    wr_parser = subparsers.add_parser('wavernn', aliases=['wr'])
    wr_parser.add_argument('--batched', '-b', dest='batched', action='store_true', help='Fast Batched Generation')
    wr_parser.add_argument('--unbatched', '-u', dest='batched', action='store_false', help='Slow Unbatched Generation')
    wr_parser.add_argument('--overlap', '-o', type=int, help='[int] number of crossover samples')
    wr_parser.add_argument('--target', '-t', type=int, help='[int] number of samples in each batch index')
    wr_parser.add_argument('--voc_weights', type=str, help='[string/path] Load in different WaveRNN weights')
    wr_parser.set_defaults(batched=None)

    gl_parser = subparsers.add_parser('griffinlim', aliases=['gl'])
    gl_parser.add_argument('--iters', type=int, default=32, help='[int] number of griffinlim iterations')

    args = parser.parse_args()

    hp.configure(args.hp_file)  # Load hparams from file

    args.vocoder = 'wavernn'
    # set defaults for any arguments that depend on hparams
    if args.vocoder == 'wavernn':
        if args.target is None:
            args.target = hp.voc_target
        if args.overlap is None:
            args.overlap = hp.voc_overlap
        if args.batched is None:
            args.batched = hp.voc_gen_batched

    batched = args.batched
    target = args.target
    overlap = args.overlap
    tts_weights = args.tts_weights

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    print('\nInitialising WaveRNN Model...\n')
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

    voc_load_path = '/mnt/WaveRNN/checkpoints/biaobei_raw.wavernn/latest_weights.pyt'
    voc_model.load(voc_load_path)

    print('\nInitialising Tacotron Model...\n')

    # Instantiate Tacotron Model
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

    tts_load_path = '/mnt/WaveRNN/checkpoints/biaobei_lsa_smooth_attention.tacotron/latest_weights.pyt'
    tts_model.load(tts_load_path)


    def synthesizer(input_text, tts_model=tts_model, voc_model=voc_model, batched=batched, mu_law=hp.mu_law,
                    voc_overlap=hp.voc_overlap,voc_target=hp.voc_target):

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
        voc_model.generate(m, out, batched, voc_target, voc_overlap, mu_law)

        cost_time = time.time() - start_time
        print('synthesizing cost {} sec'.format(cost_time))

        return out.getvalue()

    class UIResource:
        def on_get(self, req, res):
          res.content_type = 'text/html'
          res.body = html_body

    class SynthesisResource:
        def on_get(self, req, res):
            if not req.params.get('text'):
                raise falcon.HTTPBadRequest()
            res.data = synthesizer(slug(req.params.get('text'), style=Style.TONE3, separator=' '))
            #res.data = out_data
            res.content_type = 'audio/wav'

    api = falcon.API()
    api.add_route('/synthesize', SynthesisResource())
    api.add_route('/', UIResource())

    print('Serving on port %d' % args.port)
    simple_server.make_server('0.0.0.0', 6006, api).serve_forever()