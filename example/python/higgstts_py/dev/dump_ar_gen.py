"""Dump a real AR prompt (token ids + reference codes) for the C# generation demo.

Uses the already-dumped reference-audio codes (dll_codes.npy) so no codec / GPU
run is needed here — only the tokenizer.

    python higgstts_py/dev/dump_ar_gen.py
"""
import os

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(_ROOT, "build.py")):
    _ROOT = os.path.dirname(_ROOT)
REFS = os.path.join(_ROOT, "data", "higgstts")
TOKENIZER_JSON = r"<local>/reader-app\models\higgs_tts_v3_tokenizer.json"

TOK_TTS = 151667
TOK_REF_TEXT = 151680
TOK_REF_AUDIO = 151679
TOK_TEXT = 151672
TOK_AUDIO = 151670
AUDIO_PLACEHOLDER = -100
N_CB = 8

REF_TEXT = "I have no doubt you will become Elden lord, may you take the throne."
TEXT = "<|style:whispering|>Hello how you doing? Are you having fun these days?"


def build_prompt(tok, text, ref_text, num_ref):
    p = [TOK_TTS]
    if ref_text:
        p.append(TOK_REF_TEXT)
        p += tok.encode(ref_text, add_special_tokens=False).ids
    p.append(TOK_REF_AUDIO)
    p += [AUDIO_PLACEHOLDER] * num_ref
    p.append(TOK_TEXT)
    p += tok.encode(text, add_special_tokens=False).ids
    p.append(TOK_AUDIO)
    return np.asarray(p, dtype=np.int32)


def main():
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(TOKENIZER_JSON)
    ref_codes = np.load(os.path.join(REFS, "dll_codes.npy")).astype(np.int32)
    num_ref = ref_codes.shape[0] + (N_CB - 1)
    prompt = build_prompt(tok, TEXT, REF_TEXT, num_ref)
    print(f"prompt L={len(prompt)}  ref_codes {ref_codes.shape}")

    prompt.tofile(os.path.join(REFS, "ar_gen_prompt_ids.i32"))
    ref_codes.tofile(os.path.join(REFS, "ar_gen_ref_codes.i32"))
    print("wrote ar_gen_prompt_ids.i32 / ar_gen_ref_codes.i32")


if __name__ == "__main__":
    main()
